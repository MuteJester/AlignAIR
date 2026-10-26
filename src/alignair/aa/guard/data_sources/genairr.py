"""GenAIRR simulated positives for the AA guard's validation cohort."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from itertools import islice
from pathlib import Path

from alignair.aa.guard.hydration import SourceResult, atomic_write_text, filter_disjoint
from alignair.errors import (
    AlignAIRError, DataContractError, InternalError, OperationCancelledError, OperationIOError, report_error,
)

# Retained settings from the original validation study.
START_SEED = 1_100_000_000
BUDGET = 4000
WANT = 120
RATE = 0.05


def _first(call):
    return (call or "").split(",")[0]


def collect_genairr(denylist, *, store, census_dir, record_stream=None):
    """Collect simulated positives and write the existing data/census filenames.

    The target is applied before denylist filtering, as in the original study;
    excluded records are not replaced. An injected stream must start at
    ``START_SEED`` and preserve simulator record order. Without one, the original
    recombination/S5F experiment is created lazily.
    """
    if record_stream is None:
        import GenAIRR as ga

        experiment = ga.Experiment.on("HUMAN_IGH_OGRDB").recombine().mutate(model="s5f", rate=RATE)
        record_stream = experiment.stream_records(n=BUDGET, seed=START_SEED)

    records, generated = [], 0
    rejected = {"not_productive": 0, "no_aa": 0, "stop_codon": 0, "no_frame": 0, "bad_len": 0}
    for index, record in enumerate(islice(record_stream, BUDGET)):
        generated += 1
        seed = START_SEED + index
        if not isinstance(record, Mapping):
            raise DataContractError("Simulator records must be mappings.", context={"record_index": index})
        if not record.get("productive"):
            rejected["not_productive"] += 1
            continue
        sequence = record.get("sequence_aa")
        if not sequence:
            rejected["no_aa"] += 1
            continue
        if not isinstance(sequence, str):
            raise DataContractError("Simulator sequence_aa must be text.", context={"record_index": index})
        if "*" in sequence:
            rejected["stop_codon"] += 1
            continue
        if record.get("junction_start") is None:
            rejected["no_frame"] += 1
            continue
        if not 100 <= len(sequence) <= 160:
            rejected["bad_len"] += 1
            continue
        records.append({
            "id": f"pos_sim_{seed}", "aa": sequence.upper(), "group": "positive_sim",
            "category": "pos_igh_simulator", "lineage_key": f"genairr_fresh_seed_{seed}", "seed": seed,
            "v_call": _first(record.get("v_call")), "j_call": _first(record.get("j_call")),
            "source": "genairr_sim", "verification": "simulator_construction_truth",
        })
        if len(records) >= WANT:
            break
    kept, dropped = filter_disjoint(records, denylist=denylist, phase3=({}, {}))
    lengths = [len(record["aa"]) for record in kept]
    census = {
        "source": "genairr_sim",
        "provenance": {"dataconfig": "HUMAN_IGH_OGRDB", "engine": "GenAIRR recombine+s5f", "rate": RATE,
                       "seed_range": [START_SEED, START_SEED + generated - 1],
                       "held_out": "seeds >= 1e9 (training used < 1e9); disjoint from Gate-3 sim (~1e9)",
                       "completeness": "simulator construction truth (independent of guard)"},
        "generated_scanned": generated, "rejected": rejected, "kept_after_denylist": len(kept),
        "denylist_dropped": dropped, "distinct_seeds": len({record["seed"] for record in kept}),
        "aa_len": {"min": min(lengths, default=0), "max": max(lengths, default=0)},
        "distinct_v_calls": len({record["v_call"] for record in kept}),
        "distinct_j_calls": len({record["j_call"] for record in kept}),
    }
    atomic_write_text(Path(store) / "genairr_sim.normalized.json", json.dumps(kept))
    atomic_write_text(Path(census_dir) / "genairr_sim.census.json", json.dumps(census, indent=2, sort_keys=True))
    return SourceResult("genairr_sim", kept, census)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--error-format", choices=("text", "json"), default="text")
    for option in ("cleaned", "fingerprint", "phase3-gold", "store", "census-dir"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from alignair.aa.guard.disjoint import build_gate3_global_denylist

        if not args.phase3_gold.is_file():
            raise OperationIOError("Phase-3 digest replay labels are missing.", context={"path": args.phase3_gold})
        fingerprint = json.loads(args.fingerprint.read_text())
        if not isinstance(fingerprint, dict) or not isinstance(fingerprint.get("record_digest", {}), dict):
            raise DataContractError("Pre-split fingerprint requires a record_digest mapping.")
        denylist = build_gate3_global_denylist(args.cleaned, fingerprint, phase3_gold_path=args.phase3_gold)
        result = collect_genairr(denylist, store=args.store, census_dir=args.census_dir)
        print(json.dumps(result.census, indent=2, sort_keys=True))
        return 0
    except AlignAIRError as exc:
        error = exc
    except KeyboardInterrupt:
        error = OperationCancelledError("GenAIRR collection interrupted.")
    except OSError as exc:
        error = OperationIOError("Cannot access GenAIRR collection inputs or outputs.", context={"reason": str(exc)})
        error.__cause__ = exc
    except json.JSONDecodeError as exc:
        error = DataContractError("Invalid JSON in GenAIRR collection input.", context={"line": exc.lineno})
        error.__cause__ = exc
    except Exception as exc:
        error = InternalError("Unexpected GenAIRR collection failure.", context={"cause_type": type(exc).__name__})
        error.__cause__ = exc
    return report_error(error, debug=args.debug, json_output=args.error_format == "json")


if __name__ == "__main__":
    raise SystemExit(main())
