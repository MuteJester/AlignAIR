"""Lab chip model files to ``chip-v1``: the chip trained under a lab name, as AlignAIR's network.

A model file trained as ``lab-chip-v1`` (a 128-channel trunk) or ``lab-chipw64-v1`` (64 channels) holds the weights
``chip-v1`` builds, under the lab's name. ``convert_lab_chip`` rewrites only its configuration (``architecture``
``chip-v1``, ``trunk_channels`` the lab name's width) and its provenance (``converted_from``: the source's
architecture and SHA-256); the weights and the embedded reference are carried over byte for byte, nothing is
unpickled, and the training sections (pickled dataconfigs, optimizer state) are left out, so the result is an
inference model file. The written file is read back (reference integrity and bindings) before it replaces ``dst``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os

from ..errors import ConfigurationError
from . import container, serialize

LAB_CHIPS = {"lab-chip-v1": 128, "lab-chipw64-v1": 64}
ARCHITECTURE = "chip-v1"
_TRAINING_SECTIONS = ("dataconfig/", "train_state")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_lab_chip(src: str, dst: str) -> dict:
    """Write ``dst``, the ``chip-v1`` inference model file of the lab chip model file ``src``; returns ``{"from",
    "to", "trunk_channels", "sha256"}`` (``sha256``: of ``dst``). ConfigurationError when ``src`` is not a lab chip."""
    from . import read_model_state
    header = container.read_header(src)
    cfg = serialize.config_from_bytes(container.read_section(src, "config"))
    if cfg.architecture not in LAB_CHIPS:
        raise ConfigurationError(f"{src} is not a lab chip model file: its architecture is {cfg.architecture!r}",
                                 context={"path": src, "architecture": cfg.architecture},
                                 hint=f"Only {', '.join(sorted(LAB_CHIPS))} model files convert to {ARCHITECTURE}.")
    converted = dataclasses.replace(cfg, architecture=ARCHITECTURE, trunk_channels=LAB_CHIPS[cfg.architecture])
    sections, formats = {}, {}
    for name, section in header["sections"].items():
        if name.startswith(_TRAINING_SECTIONS):
            continue
        payload = (serialize.config_to_bytes(converted) if name == "config"
                   else container.read_section(src, name))                     # read_section verifies checksums
        sections[name] = (payload, section["codec"])
        formats[name] = section.get("format", "bytes")
    new_header = {key: value for key, value in header.items() if not key.startswith("_") and key != "sections"}
    reference = dict(header["reference"])
    reference["dataconfigs"] = [dict(entry, section=None) for entry in reference.get("dataconfigs", [])]
    new_header["reference"] = reference
    new_header["provenance"] = {**header.get("provenance", {}),
                                "converted_from": {"architecture": cfg.architecture, "sha256": _sha256(src)}}
    new_header["_formats"] = formats
    tmp = f"{dst}.tmp.{os.getpid()}"
    try:
        container.write_container(tmp, new_header, sections)
        read_model_state(tmp)                                                  # integrity and bindings, no build
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return {"from": cfg.architecture, "to": ARCHITECTURE, "trunk_channels": converted.trunk_channels,
            "sha256": _sha256(dst)}
