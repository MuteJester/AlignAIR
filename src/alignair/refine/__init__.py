"""The chip's inference-time exact stages, no weights and no gradient: one locus per read for a reference of several
loci (``locus``), the V end and J start counted along the called alleles' germlines (``boundaries``), the D call's
local alignment (``dlocal``), the D edges on the called D's germline (``drefine``) and the V call's banded gapped
re-score (``vrescore``)."""
