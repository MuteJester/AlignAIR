"""The headline metrics: the primary metrics of the repertoire suites, and the rows a report leads with.

They live in ``core`` so that suites (which may not import ``report``) and reports share one list.
"""
from __future__ import annotations

HEADLINE: tuple[str, ...] = ("clone_key.exact", "junction.nt_exact", "v.top1_in_set", "j.top1_in_set",
                             "v.mutation_count_err", "d.top1_in_set", "productive.correct")
