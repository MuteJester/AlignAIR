"""Pinned, hardened container runs (``core.container``): Docker is never started; ``subprocess.run`` is replaced by a
fake that answers as Docker would."""
from __future__ import annotations

import json
import os

import pytest

from alignair_assay.core import container
from alignair_assay.core.container import (
    ContainerError,
    Mount,
    check_image,
    container_name,
    hardened_run,
    pinned_repo_digest,
    run_container,
)

DIGEST = "sha256:" + "a" * 64
IMAGE = f"quay.io/biocontainers/samtools:1.0--h0_0@{DIGEST}"


class _Completed:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def _inspected(*repo_digests: str) -> str:
    return json.dumps([{"Id": "sha256:" + "1" * 64, "RepoDigests": list(repo_digests)}])


def test_pinned_repo_digest_drops_the_tag_but_not_a_registry_port():
    assert pinned_repo_digest(IMAGE) == f"quay.io/biocontainers/samtools@{DIGEST}"
    assert pinned_repo_digest(f"host:5000/tools/x:2.1@{DIGEST}") == f"host:5000/tools/x@{DIGEST}"
    assert pinned_repo_digest(f"ncbi/sra-tools@{DIGEST}") == f"ncbi/sra-tools@{DIGEST}"
    with pytest.raises(ValueError, match="digest"):
        pinned_repo_digest("quay.io/biocontainers/samtools:1.0--h0_0")


def test_check_image_finds_the_pinned_digest(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return _Completed(_inspected(f"quay.io/biocontainers/samtools@{DIGEST}"))

    monkeypatch.setattr(container.subprocess, "run", run)
    found = check_image(IMAGE)
    assert found == {"image": IMAGE, "id": "sha256:" + "1" * 64,
                     "repo_digests": [f"quay.io/biocontainers/samtools@{DIGEST}"]}
    assert calls == [["docker", "image", "inspect", IMAGE]]


def test_check_image_pulls_an_image_that_is_not_local(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[:3] == ["docker", "image", "inspect"] and len(calls) == 1:
            return _Completed(returncode=1, stderr="No such image")
        return _Completed(_inspected(f"quay.io/biocontainers/samtools@{DIGEST}"))

    monkeypatch.setattr(container.subprocess, "run", run)
    assert check_image(IMAGE)["image"] == IMAGE
    assert calls[1] == ["docker", "pull", IMAGE] and len(calls) == 3


def test_check_image_refuses_another_digest_a_failed_pull_and_no_docker(monkeypatch):
    other = f"quay.io/biocontainers/samtools@sha256:{'f' * 64}"
    monkeypatch.setattr(container.subprocess, "run", lambda argv, **kwargs: _Completed(_inspected(other)))
    with pytest.raises(ContainerError, match="lack the pinned digest"):
        check_image(IMAGE)

    monkeypatch.setattr(container.subprocess, "run",
                        lambda argv, **kwargs: _Completed(returncode=1, stderr="denied"))
    with pytest.raises(ContainerError, match="cannot pull"):
        check_image(IMAGE)

    def missing(argv, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(container.subprocess, "run", missing)
    with pytest.raises(ContainerError, match="docker is not available"):
        check_image(IMAGE)
    with pytest.raises(ValueError, match="digest"):
        check_image("quay.io/biocontainers/samtools:1.0--h0_0")


@pytest.mark.parametrize("stdout", ["", "not json", "[]", "{}", '["x"]', "null"])
def test_check_image_refuses_an_inspect_answer_it_cannot_read(monkeypatch, stdout):
    monkeypatch.setattr(container.subprocess, "run", lambda argv, **kwargs: _Completed(stdout))
    with pytest.raises(ContainerError, match="cannot read docker's inspection"):
        check_image(IMAGE)


def test_hardened_run_is_offline_unprivileged_and_mounts_only_what_it_is_given(tmp_path):
    argv = hardened_run(IMAGE, ["samtools", "view", "/data/x.bam"], name="alignair-t-1",
                        mounts=[Mount(tmp_path, "/data")])
    assert argv[:3] == ["docker", "run", "--rm"]
    assert argv[argv.index("--name") + 1] == "alignair-t-1"
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert argv[argv.index("--security-opt") + 1] == "no-new-privileges"
    assert argv[argv.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert "--cpus" not in argv
    assert "-v" not in argv and "--volume" not in argv
    assert _mounts(argv) == [f"type=bind,source={tmp_path},target=/data,readonly"]
    assert argv[argv.index(IMAGE) + 1:] == ["samtools", "view", "/data/x.bam"]
    writable = hardened_run(IMAGE, ["true"], name="n", mounts=[Mount(tmp_path, "/data", read_only=False)],
                            workdir="/data")
    assert _mounts(writable) == [f"type=bind,source={tmp_path},target=/data"]
    assert writable[writable.index("-w") + 1] == "/data"
    limited = hardened_run(IMAGE, ["true"], name="n", mounts=[], cpus=8)
    assert limited[limited.index("--cpus") + 1] == "8" and limited.index("--cpus") < limited.index(IMAGE)
    with pytest.raises(ValueError, match="digest"):
        hardened_run("samtools:latest", ["true"], name="n", mounts=[])


def _mounts(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, part in enumerate(argv) if part == "--mount"]


def test_a_mount_keeps_a_path_with_a_colon_or_a_comma_whole(tmp_path):
    """``-v src:dst:ro`` splits a path at ``:``; ``--mount`` takes comma-separated fields, so a field holding a comma
    or a quote is quoted as CSV, as Docker parses it."""
    colon = tmp_path / "run:2026-10-20"
    comma = tmp_path / 'a,b"c'
    argv = hardened_run(IMAGE, ["true"], name="n", mounts=[Mount(colon, "/in"), Mount(comma, "/out", False)])
    assert _mounts(argv) == [f"type=bind,source={colon},target=/in,readonly",
                             f'type=bind,"source={tmp_path}/a,b""c",target=/out']
    with pytest.raises(ValueError, match="absolute"):
        hardened_run(IMAGE, ["true"], name="n", mounts=[Mount(tmp_path, "data")])


@pytest.mark.skipif(os.environ.get("ALIGNAIR_ASSAY_DOCKER") != "1", reason="set ALIGNAIR_ASSAY_DOCKER=1")
def test_docker_refuses_a_missing_mount_source_instead_of_creating_it(tmp_path):
    """With ``-v`` Docker would create the missing directory, as root; with ``--mount`` the run fails."""
    from alignair_assay.real.locus import SAMTOOLS_IMAGE

    check_image(SAMTOOLS_IMAGE)
    missing = tmp_path / "missing"
    name = container_name("test", "mount")
    argv = hardened_run(SAMTOOLS_IMAGE, ["true"], name=name, mounts=[Mount(missing, "/data")])
    with pytest.raises(ContainerError, match="failed"):
        run_container(argv, name=name, timeout=120, what="a run on a missing mount")
    assert not missing.exists()


def test_container_name_is_a_valid_docker_name_per_process():
    name = container_name("locus", "P25_I58_S1")
    assert name == f"alignair-locus-P25_I58_S1-{os.getpid()}"
    assert container_name("locus", "a b/c:d") == f"alignair-locus-a-b-c-d-{os.getpid()}"


def test_run_container_returns_the_output_and_refuses_a_failure(monkeypatch):
    monkeypatch.setattr(container.subprocess, "run", lambda argv, **kwargs: _Completed("out\n", stderr="note"))
    assert run_container(["docker", "run", "x"], name="n", timeout=5, what="samtools view").stdout == "out\n"
    monkeypatch.setattr(container.subprocess, "run",
                        lambda argv, **kwargs: _Completed(returncode=1, stderr="[E::hts_open] fail"))
    with pytest.raises(ContainerError, match=r"samtools view failed with return code 1: \[E::hts_open\] fail"):
        run_container(["docker", "run", "x"], name="n", timeout=5, what="samtools view")


def test_a_timeout_kills_the_container_by_name(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["docker", "run"]:
            raise container.subprocess.TimeoutExpired(argv, kwargs.get("timeout"))
        return _Completed()

    monkeypatch.setattr(container.subprocess, "run", run)
    with pytest.raises(ContainerError, match="did not finish within 5 s.*alignair-x-1 was killed"):
        run_container(["docker", "run", "--name", "alignair-x-1", "img"], name="alignair-x-1", timeout=5,
                      what="samtools view")
    assert calls[1] == ["docker", "kill", "alignair-x-1"]

    def missing(argv, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(container.subprocess, "run", missing)
    with pytest.raises(ContainerError, match="did not run"):
        run_container(["docker", "run", "img"], name="n", timeout=5, what="samtools view")
