"""The architecture lab: experiments on AlignAIR model candidates. The product trains, the assay measures, the lab
runs experiments with both.

A development package, excluded from the wheel like ``alignair_assay``. The lab imports the product (``alignair``)
and the assay (``alignair_assay``); neither imports the lab. Its modules, each importing only those before it in this
list (``tests/alignair_lab/test_layering.py``):

- ``recipe``: the training recipe every screen and record cache uses (the 1.0.0 settings under the product's record
  recipe, v4), and the retrain recipe.
- ``records``: the record cache, the recipe's curriculum records written once and replayed by any number of runs.
- ``noise``: run-to-run training noise: a comparison's pooled Δ, the null band of baseline screens and the decision
  rule; pure.
- ``headline``: a screen's one-page headline, rendered from the assay's comparison summary.
- ``architectures``: candidate architectures, registered with ``alignair.core.components`` when imported; product
  code in shape, so it imports neither the assay nor the rest of the lab.
- ``baselines``: what a screen is compared with: baseline checkpoints, trusted by their contents, the protocol a
  baseline screen shares with its screens, the registered baseline screen and the null band of a protocol.
- ``screen``: an architecture trained from the record cache and compared by the assay with a baseline at the same
  step: a baseline screen (lab-null-v2, an init seed's) or a milestone.
- ``calibrate``: baseline screens of several init seeds compared pair by pair into the null band of their protocol.
- ``novel``: the novel-allele screen: chip-v1 trained from a novel record cache, gated on held-out novel alleles.
- ``runs``: a live run of the retrain recipe from a data seed, no record cache (the ``train`` command).
- ``databand``: runs of several data seeds compared pair by pair into the data-seed band (the ``data-band`` command).
- ``locusscreen``: a chip shared by several loci against one chip per locus, at equal training per locus (the
  ``locus-screen`` command).
- ``cli``: ``python -m alignair_lab``.
"""
