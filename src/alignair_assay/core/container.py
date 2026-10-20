"""Pinned, hardened container runs, shared by the steps that run a tool in Docker.

- ``check_image``: the image, pulled by digest when it is not local, and refused unless Docker lists the pinned
  digest among its RepoDigests.
- ``hardened_run``: the ``docker run`` argv of a step. The container is removed when it exits and named, so a run
  that times out can be killed. It has no network and no capabilities, cannot gain privileges, runs as the current
  user, and sees only the directories it is given (``--mount`` binds, which must exist), read-only unless a mount
  says otherwise.
- ``run_container``: runs that argv with a timeout. On a timeout the container is killed by name: killing the
  ``docker`` client does not stop it.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
from pathlib import Path
from typing import NamedTuple

PULL_TIMEOUT = 1800.0  # seconds a ``docker pull`` may take
INSPECT_TIMEOUT = 120.0
KILL_TIMEOUT = 60.0
_NAME_INVALID = re.compile(r"[^A-Za-z0-9_.-]+")


class ContainerError(RuntimeError):
    """Docker is missing, the image is not the pinned one, or a container run failed or timed out."""


class Mount(NamedTuple):
    """A host directory mounted into the container at ``container`` (an absolute path in it)."""

    host: str | Path
    container: str
    read_only: bool = True


def pinned_repo_digest(image: str) -> str:
    """``<repository>@<digest>`` of a pinned image reference (``<repository>[:<tag>]@<digest>``), as Docker lists it
    among the image's RepoDigests. ValueError for a reference without a digest."""
    name, _, digest = image.partition("@")
    if not digest:
        raise ValueError(f"image {image!r} is not pinned by digest (<repository>[:<tag>]@sha256:...)")
    repository = name.rsplit(":", 1)[0] if ":" in name.rsplit("/", 1)[-1] else name
    return f"{repository}@{digest}"


def check_image(image: str, *, pull_timeout: float = PULL_TIMEOUT) -> dict:
    """Pinned image ``image``, pulled by digest when it is not local: ``{"image", "id", "repo_digests"}``.
    ContainerError when Docker is not available, the image cannot be pulled, or its RepoDigests lack the pinned
    digest; ValueError for an image that is not pinned by digest."""
    pinned = pinned_repo_digest(image)
    inspect = ["docker", "image", "inspect", image]
    try:
        completed = subprocess.run(inspect, capture_output=True, text=True, timeout=INSPECT_TIMEOUT)
        if completed.returncode != 0:
            pulled = subprocess.run(["docker", "pull", image], capture_output=True, text=True, timeout=pull_timeout)
            if pulled.returncode != 0:
                raise ContainerError(f"docker cannot pull {image}: {(pulled.stdout + pulled.stderr).strip()[-500:]}")
            completed = subprocess.run(inspect, capture_output=True, text=True, timeout=INSPECT_TIMEOUT)
    except FileNotFoundError as error:
        raise ContainerError(f"docker is not available, and {image} runs in Docker: install Docker or start its "
                             "daemon") from error
    except subprocess.TimeoutExpired as error:
        raise ContainerError(f"docker did not answer for {image} within {error.timeout:g} s") from error
    if completed.returncode != 0:
        raise ContainerError(f"docker cannot inspect {image}: {completed.stderr.strip()[-500:]}")
    try:
        found = json.loads(completed.stdout)[0]
        repo_digests = list(found.get("RepoDigests") or [])
    except (ValueError, LookupError, TypeError, AttributeError) as error:  # not JSON, no image, not an object
        raise ContainerError(f"cannot read docker's inspection of {image} ({type(error).__name__}): "
                             f"{completed.stdout.strip()[:200]!r}") from error
    if pinned not in repo_digests:
        raise ContainerError(f"image {image}: its RepoDigests {repo_digests} lack the pinned digest "
                             f"{image.partition('@')[2]}")
    return {"image": image, "id": found.get("Id"), "repo_digests": repo_digests}


def container_name(*parts: str) -> str:
    """``alignair-<parts>-<pid>``, with each run of characters Docker does not allow in a name made ``-``: one name
    per step and process, so a container that times out can be killed by it."""
    return "-".join(["alignair", *(_NAME_INVALID.sub("-", part) for part in parts), str(os.getpid())])


def hardened_run(image: str, args: list[str], *, name: str, mounts: list[Mount], workdir: str | None = None,
                 env: dict[str, str] | None = None) -> list[str]:
    """The ``docker run`` argv that runs ``args`` in pinned image ``image`` as the module docstring describes.
    ValueError for an image that is not pinned by digest or a mount whose container path is not absolute."""
    pinned_repo_digest(image)
    argv = ["docker", "run", "--rm", "--name", name, "--network", "none", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--user", f"{os.getuid()}:{os.getgid()}"]
    for key, value in (env or {}).items():
        argv += ["-e", f"{key}={value}"]
    for mount in mounts:
        argv += ["--mount", mount_option(mount)]
    if workdir:
        argv += ["-w", workdir]
    return [*argv, image, *args]


def mount_option(mount: Mount) -> str:
    """The ``--mount`` value of ``mount``: a bind of its absolute host path, ``readonly`` unless it says otherwise.
    Unlike ``-v``, a bind mount whose source does not exist fails the run rather than being created (as root), and
    a ``:`` in a path is kept; a field holding a comma or a quote is quoted as CSV, which is how Docker splits the
    value. ValueError for a container path that is not absolute."""
    if not mount.container.startswith("/"):
        raise ValueError(f"container path {mount.container!r} is not absolute")
    fields = ["type=bind", f"source={Path(mount.host).absolute()}", f"target={mount.container}"]
    if mount.read_only:
        fields.append("readonly")
    line = io.StringIO()
    csv.writer(line, lineterminator="").writerow(fields)
    return line.getvalue()


def run_container(argv: list[str], *, name: str, timeout: float, what: str) -> subprocess.CompletedProcess:
    """Run container argv ``argv`` (named ``name``) and return its completed process, text captured. ContainerError,
    naming ``what``, when it does not run, returns non-zero, or does not finish within ``timeout`` seconds: the
    container is then killed by name."""
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        kill(name)  # the docker client was killed; its container was not
        raise ContainerError(f"{what} did not finish within {timeout:g} s; its container {name} was killed") \
            from error
    except FileNotFoundError as error:
        raise ContainerError(f"{what} did not run: {error}") from error
    if completed.returncode != 0:
        report = completed.stderr.strip() or completed.stdout.strip()  # a tool's output may be its data
        raise ContainerError(f"{what} failed with return code {completed.returncode}: {report[-1000:]}")
    return completed


def kill(name: str) -> None:
    """``docker kill`` container ``name``, best effort: it may have stopped by itself."""
    try:
        subprocess.run(["docker", "kill", name], capture_output=True, text=True, timeout=KILL_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        pass
