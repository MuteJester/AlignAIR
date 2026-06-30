from __future__ import annotations


def dnalignair_predictor(model, reference_set, *, device=None, batch_size: int = 64, **kwargs):
    """Return a predictor callable for ``DNAlignAIR`` raw-read inference."""

    from ....inference.dnalignair_infer import predict_reads

    def _predict(reads: list[str]):
        return predict_reads(
            model,
            reference_set,
            reads,
            device=device,
            batch_size=batch_size,
            **kwargs,
        )

    return _predict
