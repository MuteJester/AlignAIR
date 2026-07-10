"""`alignair models …` — download and manage shipped models from configurable registries."""
from __future__ import annotations

from ..registry import cache, sources


def register(sub) -> None:
    p = sub.add_parser("models", help="download / manage shipped AlignAIR models")
    ms = p.add_subparsers(dest="models_cmd", required=True)

    def net(sp):
        sp.add_argument("--registry", action="append", help="registry url (repeatable)")
        sp.add_argument("--offline", action="store_true", help="never touch the network")
        return sp

    net(ms.add_parser("list", help="list available models + install/update status")).set_defaults(func=_list)
    for name, fn, help_ in (("get", _get, "download a model (id[@version]) into the cache"),
                            ("path", _path, "print a model's local path (id[@version])"),
                            ("info", _info, "print a model's card (id[@version])")):
        sp = net(ms.add_parser(name, help=help_))
        sp.add_argument("model", help="model id or id@version")
        sp.set_defaults(func=fn)
    for name, fn, help_ in (("update", _update, "update installed models to the latest version"),
                            ("verify", _verify, "re-hash installed models against the registry")):
        sp = net(ms.add_parser(name, help=help_))
        sp.add_argument("model", nargs="?", help="model id (default: all installed)")
        sp.set_defaults(func=fn)
    prn = ms.add_parser("prune", help="remove old cached model versions")
    prn.add_argument("--keep", type=int, default=1)
    prn.set_defaults(func=_prune)


def _srcs(args):
    return sources.resolve_sources(getattr(args, "registry", None))


def _list(args) -> int:
    installed = cache.installed_models()
    seen = set()
    for src in _srcs(args):
        try:
            reg = sources.load_registry(src, offline=args.offline)
        except Exception as e:                                     # unreachable registry: note + continue
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
    p = cache.resolve_model(args.model, sources=_srcs(args), offline=args.offline)
    print(f"cached -> {p}")
    return 0


def _path(args) -> int:
    print(cache.resolve_model(args.model, sources=_srcs(args), offline=args.offline))
    return 0


def _info(args) -> int:
    from ..model_file import read_metadata
    p = cache.resolve_model(args.model, sources=_srcs(args), offline=args.offline)
    md = read_metadata(str(p))
    print(f"{md.get('model_id') or '(local)'} {md.get('model_version') or ''}  "
          f"(format v{md.get('model_format_version')}, alignair {md.get('created_by_alignair')})")
    print(f"  created:  {md.get('created')}")
    print(f"  species/locus: {md.get('species')} / {md.get('locus')}")
    ac = md.get("model", {}).get("allele_counts", {})
    print(f"  alleles:  V={ac.get('v')} D={ac.get('d')} J={ac.get('j')}")
    print(f"  reference: allele_order_sha256={md.get('reference', {}).get('allele_order_sha256', '')[:12]}…")
    return 0


def _update(args) -> int:
    targets = [args.model] if args.model else list(cache.installed_models())
    if not targets:
        print("no models installed")
        return 0
    for mid in targets:
        p = cache.resolve_model(mid, sources=_srcs(args), offline=args.offline)   # latest -> download if new
        print(f"{mid}: {p.name}")
    return 0


def _verify(args) -> int:
    rows = cache.verify_installed(_srcs(args), offline=args.offline, model_id=args.model)
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
