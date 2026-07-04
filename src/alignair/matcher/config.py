from dataclasses import dataclass


@dataclass
class MatcherConfig:
    vocab_size: int = 6          # PAD, A, C, G, T, N
    d_model: int = 256
    d_match: int = 128
    n_layers: int = 16
    kernel: int = 3              # odd; keeps length exact under dilation
    dilations: tuple = (1, 2, 4, 8, 16)
    dropout: float = 0.1
    tau: float = 0.05
    margin: float = 0.2
    max_len: int = 576
