"""Per-test process state and filesystem isolation."""
import hashlib
import ipaddress
import random
import socket
import sys

import numpy as np
import pytest
import torch


def _loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(scope="session", autouse=True)
def torch_thread_budget():
    previous = torch.get_num_threads()
    environment = pytest.MonkeyPatch()
    environment.setenv("OMP_NUM_THREADS", "2")
    environment.setenv("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
        environment.undo()


@pytest.fixture(autouse=True)
def isolated_test_environment(request, monkeypatch, tmp_path):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    dtype, grad_enabled = torch.get_default_dtype(), torch.is_grad_enabled()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    seed = int.from_bytes(hashlib.sha256(request.node.nodeid.encode()).digest()[:4], "big")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    for name, directory in (
        ("ALIGNAIR_CACHE_DIR", "cache"), ("ALIGNAIR_CONFIG_DIR", "config"),
        ("XDG_CACHE_HOME", "xdg-cache"), ("XDG_CONFIG_HOME", "xdg-config"),
        ("HF_HOME", "hf"), ("HF_HUB_CACHE", "hf-hub"),
        ("HUGGINGFACE_HUB_CACHE", "hf-hub"),
    ):
        monkeypatch.setenv(name, str(tmp_path / directory))
    for name in ("ALIGNAIR_REGISTRY", "ALIGNAIR_OFFLINE", "ALIGNAIR_UPDATE_GOLDENS"):
        # Updating snapshots is a deliberate authoring operation, never inherited by CI.
        if name != "ALIGNAIR_UPDATE_GOLDENS" or not request.config.getoption("--update-goldens"):
            monkeypatch.delenv(name, raising=False)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "ALIGNAIR_NO_NETWORK"):
        monkeypatch.delenv(name, raising=False)
    hub_constants = sys.modules.get("huggingface_hub.constants")
    if hub_constants is not None:
        monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path / "hf-hub"))
        monkeypatch.setattr(hub_constants, "HF_HOME", str(tmp_path / "hf"))
    original_connect = socket.socket.connect
    original_resolve = socket.getaddrinfo

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not _loopback(address[0]):
            pytest.fail(f"External network access is forbidden in tests: {address[0]}")
        return original_connect(sock, address)

    def resolve(host, *args, **kwargs):
        if host is not None and not _loopback(host):
            pytest.fail(f"External DNS lookup is forbidden in tests: {host}")
        return original_resolve(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        torch.set_default_dtype(dtype)
        torch.set_grad_enabled(grad_enabled)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
