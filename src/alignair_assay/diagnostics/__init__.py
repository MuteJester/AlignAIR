"""Model diagnostics for AlignAIR: with ``core/reference.py``, the only code that imports ``alignair``.

``replay``, ``heads``, ``ablation``, ``profile`` and ``identity`` are the pieces; ``workflow.diagnose.evaluate``
orchestrates them over a run directory, with its cache, files and report.
"""
