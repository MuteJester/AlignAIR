"""Compatibility imports for the shared protein tokenizer."""
from ..data.tokenizer import (AA_TOKEN_DICT, AA_VOCAB_SIZE, AAProteinTokenizer,
                              InvalidResidueError, pad_tokenize_aa)

__all__ = ["AA_TOKEN_DICT", "AA_VOCAB_SIZE", "AAProteinTokenizer", "InvalidResidueError", "pad_tokenize_aa"]
