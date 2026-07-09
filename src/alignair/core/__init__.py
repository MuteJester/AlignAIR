"""Faithful PyTorch AlignAIR model (port of TF src/AlignAIR/Models), unified into one class.

One model, ``AlignAIR``. Single- vs multi-chain is data, not a subclass: build the config from one
GenAIRR dataconfig (``num_chain_types == 1`` -> no chain_type head) or several (-> chain_type/locus
head). See :meth:`alignair.config.alignair_config.AlignAIRConfig.from_dataconfigs`.
"""
from .model import AlignAIR

__all__ = ["AlignAIR"]
