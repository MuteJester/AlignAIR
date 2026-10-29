"""The null candidate: conv-v1's components under a lab name.

A screen of ``lab-null-v1`` trains what the baseline trains, on the same records, from the same seed, so its
comparison with the baseline milestone at the same step must be a tie: any significant difference measures the rig
(GPU nondeterminism, a data mismatch), not an architecture.
"""
from alignair.core.components import ModelComponents

from . import register

register("lab-null-v1", ModelComponents())
