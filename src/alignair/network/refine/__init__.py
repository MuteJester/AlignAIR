"""The chip's inference-time exact stages, no weights and no gradient: one locus per read for a reference of several
loci (``locus``), the V end and J start counted along the called alleles' germlines (``vj_boundaries``), the D call's
local alignment (``d_local``), the D edges on the called D's germline (``d_boundaries``) and the V call's banded
gapped re-score (``v_rescore``), all reading the per-base evidence of ``ratios``."""
