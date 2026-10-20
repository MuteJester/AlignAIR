"""The ``all`` command's flags: how they map onto ``RunConfig``, and ``--quick``."""
from pathlib import Path

import alignair_assay.cli.all as all_command
from alignair_assay.cli import main
from alignair_assay.workflow.orchestrate import RunConfig


def _parse(monkeypatch, *flags):
    calls = []
    monkeypatch.setattr(all_command, "run_all", lambda cfg, layout, *, force=False: calls.append((cfg, layout, force))
                        or 0)
    assert main(["all", "--run-dir", "run", "--suites", "S1, S10", "--tools", "igblast,alignair", *flags]) == 0
    (call,) = calls
    return call


def test_all_defaults_and_quick(monkeypatch):
    cfg, layout, force = _parse(monkeypatch)
    assert cfg == RunConfig(suites=("S1", "S10"), tools=("igblast", "alignair"))
    assert layout.root == Path.cwd() / "run" and not force
    # --quick is scale 0.02; an explicit --scale wins over it.
    assert _parse(monkeypatch, "--quick")[0].scale == 0.02
    assert _parse(monkeypatch, "--quick", "--scale", "0.5")[0].scale == 0.5


def test_all_flags_map_onto_the_config(monkeypatch):
    cfg, _, force = _parse(monkeypatch, "--split", "test", "--reference", "HUMAN_IGH_OGRDB", "--seed", "3",
                           "--scale", "0.1", "--workers", "2", "--threads", "4", "--igblast-dir", "/opt/igblast",
                           "--alignair-image", "img", "--alignair-model", "m@1", "--alignair-checkpoint", "x.alignair",
                           "--device", "cuda", "--force")
    assert cfg == RunConfig(suites=("S1", "S10"), tools=("igblast", "alignair"), split="test", seed=3, scale=0.1,
                            workers=2, threads=4, igblast_dir="/opt/igblast", alignair_image="img",
                            alignair_model="m@1", alignair_checkpoint="x.alignair", device="cuda")
    assert force
