"""The codons and IMGT columns the reference's anchors, its novel alleles and the AIRR fields read."""
from __future__ import annotations

GAP = "."                                    # a gap column of an IMGT-gapped form
STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})
CYS_CODONS = ("TGT", "TGC")                  # the V's conserved Cys
J_ANCHOR_CODONS = ("TGG", "TTT", "TTC")      # the J's conserved Trp or Phe
CONSERVED_CODONS = {"V": CYS_CODONS, "J": J_ANCHOR_CODONS}
IMGT_CODON_104 = 309       # the first IMGT-gapped column (0-based) of V codon 104, where the junction starts
