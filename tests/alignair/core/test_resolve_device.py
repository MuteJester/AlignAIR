"""A device request resolves to a backend this machine has: "auto" picks CUDA, then MPS, then the CPU, and an explicit
backend that is not available falls back to the CPU."""
from types import SimpleNamespace

import pytest
import torch

from alignair.core.runtime import resolve_device

pytestmark = pytest.mark.component


def _machine(monkeypatch, *, cuda, mps):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch, "backends", SimpleNamespace(**({"mps": SimpleNamespace(is_available=lambda: mps)}
                                                               if mps is not None else {})))


@pytest.mark.parametrize("cuda, mps, auto", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu"),
                                             (False, None, "cpu")])
def test_auto_picks_cuda_then_mps_then_the_cpu(monkeypatch, cuda, mps, auto):
    _machine(monkeypatch, cuda=cuda, mps=mps)
    assert resolve_device("auto") == resolve_device(None) == auto


@pytest.mark.parametrize("request_, cuda, mps, device", [
    ("cuda:1", True, False, "cuda:1"), ("cuda", False, False, "cpu"), ("mps", False, True, "mps"),
    ("mps", False, False, "cpu"), ("mps", False, None, "cpu"), ("cpu", True, True, "cpu")])
def test_an_explicit_backend_stands_when_available_else_the_cpu(monkeypatch, request_, cuda, mps, device):
    _machine(monkeypatch, cuda=cuda, mps=mps)
    assert resolve_device(request_) == device
