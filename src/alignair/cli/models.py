"""`alignair models …` — download and manage shipped models from configurable registries."""
from __future__ import annotations

from ..registry import cache, sources
from ._options import add_model_source, add_registry, model_sources, resolve_model_path


def register(sub) -> None:
    p = sub.add_parser("models", help="download / manage shipped AlignAIR models")
    ms = p.add_subparsers(dest="models_cmd", required=True)

    lp = ms.add_parser("list", help="list available models + install/update status")
    add_registry(lp)
    lp.set_defaults(func=_list)
    for name, fn, help_ in (("get", _get, "download a model (id[@version]) into the cache"),
                            ("path", _path, "print a model's local path (id[@version])")):
        sp = ms.add_parser(name, help=help_)
        sp.add_argument("model", help="model id or id@version")
        add_model_source(sp)
        sp.set_defaults(func=fn)
    for name, fn, help_ in (("update", _update, "update installed models to the latest version"),
                            ("verify", _verify, "re-hash installed models against the registry")):
        sp = ms.add_parser(name, help=help_)
        sp.add_argument("model", nargs="?", help="model id (default: all installed)")
        add_registry(sp)
        sp.set_defaults(func=fn)
    prn = ms.add_parser("prune", help="remove old cached model versions")
    prn.add_argument("--keep", type=int, default=1, help="the newest versions to keep per model (default 1)")
    prn.set_defaults(func=_prune)


def _list(args) -> int:
    installed = cache.installed_models()
    seen = set()
    for src in model_sources(args):
        try:
            reg = sources.load_registry(src, offline=args.offline)
        except Exception as e:  # noqa: BLE001 - a registry that cannot be read is listed as unreachable
            print(f"# registry {src} unreachable: {e}")
            continue
        for mid, m in reg.get("models", {}).items():
            if mid in seen:
                continue
            seen.add(mid)
            latest, inst = m.get("latest", "?"), installed.get(mid, [])
            status = "installed" if latest in inst else ("update available" if inst else "available")
            print(f"{mid:22s} {latest:10s} {status:18s} {m.get('description', '')}")
    if not seen:
        print("no models found in the configured registries")
    return 0


def _get(args) -> int:
    print(f"cached -> {resolve_model_path(args)}")
    return 0


def _path(args) -> int:
    print(resolve_model_path(args))
    return 0


def _update(args) -> int:
    targets = [args.model] if args.model else list(cache.installed_models())
    if not targets:
        print("no models installed")
        return 0
    for mid in targets:
        p = cache.resolve_model(mid, sources=model_sources(args), offline=args.offline)   # latest -> download if new
        print(f"{mid}: {p.name}")
    return 0


def _verify(args) -> int:
    rows = cache.verify_installed(model_sources(args), offline=args.offline, model_id=args.model)
    if not rows:
        print("no models installed")
        return 0
    bad = 0
    for mid, ver, ok in rows:
        state = "OK" if ok else ("UNKNOWN (not in registry)" if ok is None else "MISMATCH")
        bad += ok is False
        print(f"{mid}@{ver}: {state}")
    return 1 if bad else 0


def _prune(args) -> int:
    removed = cache.prune(keep=args.keep)
    print(f"removed {len(removed)} old cached version(s)")
    return 0
