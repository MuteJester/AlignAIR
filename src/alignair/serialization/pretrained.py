"""PretrainedMixin: save/load AlignAIR models as state_dict bundles."""
from pathlib import Path

import torch

from .bundle import save_bundle, load_bundle


class PretrainedMixin:
    """Mixed into BaseAlignAIR. Requires ``self.config`` to be a ModelConfig."""

    def save_pretrained(self, bundle_dir, *, dataconfig=None, training_meta=None) -> None:
        save_bundle(bundle_dir, model_config=self.config, state_dict=self.state_dict(),
                    dataconfig=dataconfig, training_meta=training_meta)

    @classmethod
    def from_pretrained(cls, bundle_dir):
        model_config, _dataconfig, _meta = load_bundle(bundle_dir)
        # Choose the concrete class from the saved config (ignore the calling cls).
        from ..core.single_chain import SingleChainAlignAIR
        from ..core.multi_chain import MultiChainAlignAIR
        model_cls = MultiChainAlignAIR if model_config.is_multi_chain else SingleChainAlignAIR
        model = model_cls(model_config)
        state = torch.load(Path(bundle_dir) / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model

    @staticmethod
    def load_dataconfig(bundle_dir):
        _cfg, dataconfig, _meta = load_bundle(bundle_dir)
        return dataconfig
