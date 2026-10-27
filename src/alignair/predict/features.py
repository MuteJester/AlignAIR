"""Observe named component inputs without changing model predictions or leaving hooks installed."""
from collections.abc import Mapping

import torch

from ..errors import ConfigurationError, InferenceError, NumericalError, error_context
from .forward import _output_keys, forward_batch, output_numpy


def component_features(model, records, input_adapter, *, components):
    """Concatenate each selected component's single two-dimensional input, in declared order."""
    names = _output_keys(components)
    if not isinstance(model, torch.nn.Module) or not callable(input_adapter):
        raise ConfigurationError("feature capture requires a torch model and callable input adapter")
    if names is None or not callable(getattr(model, "component_modules", None)):
        raise ConfigurationError("feature capture requires named model components")
    modules = model.component_modules()
    if not isinstance(modules, Mapping) or any(not isinstance(modules.get(name), torch.nn.Module) for name in names):
        raise ConfigurationError("feature capture names must identify registered torch components")
    if len({id(modules[name]) for name in names}) != len(names):
        raise ConfigurationError("feature capture requires distinct component modules")
    records = tuple(records)
    inputs = input_adapter(records)
    captured, handles = {}, []

    def capture(name):
        def hook(module, arguments):
            with error_context(component=name):
                if name in captured:
                    raise InferenceError("feature component was invoked more than once")
                value = arguments[0] if arguments else None
                if (not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != len(records)
                        or value.is_complex()):
                    raise InferenceError("component features must be a batch-by-feature tensor")
                if not torch.isfinite(value).all():
                    raise NumericalError("non-finite component features")
                captured[name] = value.detach().clone()
        return hook

    try:
        for name in names:
            handles.append(modules[name].register_forward_pre_hook(capture(name)))
        forward_batch(model, inputs, expected_rows=len(records))
        if set(captured) != set(names):
            raise InferenceError("a requested feature component was not invoked")
        return output_numpy(torch.cat([captured[name] for name in names], dim=-1))
    finally:
        for handle in handles:
            handle.remove()
