from __future__ import annotations


def xattn_predictor(model, reference_set, *, device=None, batch_size: int = 64, **kwargs):
    """Return a predictor callable for ``XAttnAligner`` (the LLM-encoder aligner) raw-read inference."""

    from ....inference.xattn_infer import predict_reads_xattn

    def _predict(reads: list[str]):
        return predict_reads_xattn(
            model,
            reference_set,
            reads,
            device=device,
            batch_size=batch_size,
            **kwargs,
        )

    return _predict
