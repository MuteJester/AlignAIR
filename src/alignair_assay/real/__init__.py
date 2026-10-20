"""Real AIRR-seq data (P3): human repertoires scored against each donor's own alleles from genomic sequencing.

Every module here is a sub-layer of its own, ``real.<module>``, with its own row in the dependency rule
(``tests/alignair_assay/test_architecture.py``): ``fetch`` makes the verified downloads under the data root (the
pilot runs' reads, at full quality from AWS ODP where it has them, and the genotype files); ``genotype`` holds the
donor-genotype model, the VDJbase loader and the mapping by sequence to the shared reference and to OGRDB's IUIS
labels. ``workflow`` and ``cli`` may import any of them, ``report`` none. This package marker imports nothing.
"""
