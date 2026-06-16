"""Shared AlignAIR model body (port of SingleChainAlignAIR, unified)."""
import torch
import torch.nn as nn

from ..config.model_config import ModelConfig
from ..nn.embedding import TokenPositionEmbedding
from ..nn.conv import ConvResidualFeatureExtractor
from ..nn.masking import SoftCutout
from ..nn.heads import (
    SegmentationHead, AlleleClassificationHead, MutationRateHead,
    IndelCountHead, ProductivityHead,
)
from .output import AlignAIROutput

_EMBED_DIM = 32


class BaseAlignAIR(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        L = config.max_seq_length
        self.max_seq_length = L
        self.has_d_gene = config.has_d_gene

        self.embedding = TokenPositionEmbedding(max_len=L, vocab_size=6, embed_dim=_EMBED_DIM)

        def extractor(kernels):
            return ConvResidualFeatureExtractor(
                in_channels=_EMBED_DIM, filter_size=128, kernel_sizes=kernels,
                max_pool_size=2, out_features=L, activation="tanh",
            )

        seg_kernels = [3, 3, 3, 2, 5]
        cls_kernels = [3, 3, 3, 2, 2, 2, 5]

        self.meta_extractor = extractor(seg_kernels)
        self.v_seg_extractor = extractor(seg_kernels)
        self.j_seg_extractor = extractor(seg_kernels)
        self.v_cls_extractor = extractor(cls_kernels)
        self.j_cls_extractor = extractor(cls_kernels)

        self.v_start_head = SegmentationHead(L, L)
        self.v_end_head = SegmentationHead(L, L)
        self.j_start_head = SegmentationHead(L, L)
        self.j_end_head = SegmentationHead(L, L)

        self.v_mask = SoftCutout(L, k=3.0)
        self.j_mask = SoftCutout(L, k=3.0)

        self.v_allele_head = AlleleClassificationHead(
            L, config.v_latent_dim, config.v_allele_count, config.classification_mid_activation)
        self.j_allele_head = AlleleClassificationHead(
            L, config.j_latent_dim, config.j_allele_count, config.classification_mid_activation)

        self.mutation_rate_head = MutationRateHead(L, L)
        self.indel_count_head = IndelCountHead(L, L)
        self.productivity_head = ProductivityHead(L)

        if self.has_d_gene:
            self.d_seg_extractor = extractor(seg_kernels)
            self.d_cls_extractor = extractor([3, 3, 2, 2, 5])
            self.d_start_head = SegmentationHead(L, L)
            self.d_end_head = SegmentationHead(L, L)
            self.d_mask = SoftCutout(L, k=3.0)
            self.d_allele_head = AlleleClassificationHead(
                L, config.d_latent_dim, config.d_allele_count, config.classification_mid_activation)

        # Materialize all LazyLinear params (must happen before .to(device)/save).
        self._materialize_lazy_params()

    def _materialize_lazy_params(self) -> None:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros((1, self.max_seq_length), dtype=torch.long)
            self.forward(dummy)
        self.train(was_training)

    def _positions(self, device) -> torch.Tensor:
        return torch.arange(self.max_seq_length, dtype=torch.float32, device=device).unsqueeze(0)

    @staticmethod
    def _expectation(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return (probs * positions).sum(dim=-1, keepdim=True)

    def _gene_branch(self, emb, seg_extractor, start_head, end_head, mask_layer,
                     cls_extractor, allele_head, positions):
        seg_feats = seg_extractor(emb)
        start_logits = start_head(seg_feats)
        end_logits = end_head(seg_feats)
        start_exp = self._expectation(start_logits, positions)
        end_exp = self._expectation(end_logits, positions)
        mask = mask_layer(start_exp, end_exp).unsqueeze(-1)  # (B, L, 1)
        masked = emb * mask
        cls_feats = cls_extractor(masked)
        allele = allele_head(cls_feats)
        return start_logits, end_logits, start_exp, end_exp, allele

    def forward(self, tokenized_sequence: torch.Tensor) -> AlignAIROutput:
        emb = self.embedding(tokenized_sequence)  # (B, L, E)
        positions = self._positions(emb.device)

        meta = self.meta_extractor(emb)
        mutation_rate = self.mutation_rate_head(meta)
        indel_count = self.indel_count_head(meta)
        productive = self.productivity_head(meta)

        v = self._gene_branch(emb, self.v_seg_extractor, self.v_start_head, self.v_end_head,
                              self.v_mask, self.v_cls_extractor, self.v_allele_head, positions)
        j = self._gene_branch(emb, self.j_seg_extractor, self.j_start_head, self.j_end_head,
                              self.j_mask, self.j_cls_extractor, self.j_allele_head, positions)

        out = AlignAIROutput(
            v_start_logits=v[0], v_end_logits=v[1], v_start=v[2], v_end=v[3], v_allele=v[4],
            j_start_logits=j[0], j_end_logits=j[1], j_start=j[2], j_end=j[3], j_allele=j[4],
            mutation_rate=mutation_rate, indel_count=indel_count, productive=productive,
        )

        if self.has_d_gene:
            d = self._gene_branch(emb, self.d_seg_extractor, self.d_start_head, self.d_end_head,
                                  self.d_mask, self.d_cls_extractor, self.d_allele_head, positions)
            out.d_start_logits, out.d_end_logits = d[0], d[1]
            out.d_start, out.d_end, out.d_allele = d[2], d[3], d[4]

        self._meta_features = meta  # cached for multi-chain subclass
        return out

    def regularization_loss(self) -> torch.Tensor:
        """Explicit l2 penalty over conv weights (legacy intent; legacy train_step
        dropped these — see plan porting notes). Weight 0.01 matches legacy."""
        reg = torch.zeros((), device=next(self.parameters()).device)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                reg = reg + 0.01 * (module.weight ** 2).sum()
        return reg

    @torch.no_grad()
    def apply_constraints(self) -> None:
        for head in (self.mutation_rate_head, self.indel_count_head):
            head.apply_constraints()
