"""The architecture lab: experiments on AlignAIR model candidates. The product trains, the assay measures, the lab
runs experiments with both.

A development package, excluded from the wheel like ``alignair_assay``. The lab imports the product (``alignair``)
and the assay (``alignair_assay``); neither imports the lab. Its modules, each importing only those before it in this
list (``tests/alignair_lab/test_layering.py``):

- ``recipe``: the 1.0.0 training recipe every screen and record cache uses.
- ``records``: the record cache, the recipe's curriculum records written once and replayed by any number of runs.
- ``headline``: a screen's one-page headline, rendered from the assay's comparison summary.
- ``architectures``: candidate architectures, registered with ``alignair.core.components`` when imported; product
  code in shape, so it imports neither the assay nor the rest of the lab.
- ``screen``: a candidate trained from the record cache and compared by the assay with the baseline at the same step.
- ``cli``: ``python -m alignair_lab``.
"""
