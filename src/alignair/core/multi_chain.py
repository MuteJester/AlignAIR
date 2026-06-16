"""Multi-chain AlignAIR model (adds a chain-type classification head)."""
from ..config.model_config import ModelConfig
from ..nn.heads import ChainTypeHead
from .base import BaseAlignAIR
from .output import AlignAIROutput


class MultiChainAlignAIR(BaseAlignAIR):
    def __init__(self, config: ModelConfig):
        if not config.number_of_chains or config.number_of_chains < 1:
            raise ValueError("MultiChainAlignAIR requires config.number_of_chains >= 1")
        super().__init__(config)
        L = config.max_seq_length
        # chain_type_head uses plain Linear layers (no LazyLinear), so no extra
        # materialization pass is needed beyond the base __init__'s.
        self.chain_type_head = ChainTypeHead(L, L, config.number_of_chains)

    def forward(self, tokenized_sequence) -> AlignAIROutput:
        out = super().forward(tokenized_sequence)
        out.chain_type = self.chain_type_head(self._meta_features)
        return out
