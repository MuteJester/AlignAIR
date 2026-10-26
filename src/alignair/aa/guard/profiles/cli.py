"""Command adapter for building a profile bank from pinned local sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignair.errors import (
    AlignAIRError, DataContractError, InternalError, MissingDependencyError,
    OperationCancelledError, OperationIOError, report_error,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True, help="JSON source specifications; relative file paths resolve here")
    parser.add_argument("--out", type=Path, required=True, help="new directory for HMMs, MSAs, and the profile manifest")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--error-format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    try:
        from .builder import build_profile_bank

        sources = json.loads(args.sources.read_text())
        if not isinstance(sources, dict):
            raise DataContractError("Profile source specifications must be a mapping.")
        for spec in sources.values():
            if isinstance(spec, dict) and isinstance(spec.get("path"), str):
                spec["path"] = str(args.sources.parent / spec["path"])
        manifest = build_profile_bank(sources, output_dir=args.out)
        print(json.dumps({"bank_checksum": manifest["bank_checksum"], "output": str(args.out)}, sort_keys=True))
        return 0
    except AlignAIRError as exc:
        error = exc
    except KeyboardInterrupt:
        error = OperationCancelledError("Profile bank build interrupted.")
    except OSError as exc:
        error = OperationIOError("Cannot access profile bank inputs or outputs.", context={"reason": str(exc)})
        error.__cause__ = exc
    except (json.JSONDecodeError, UnicodeError) as exc:
        error = DataContractError("Invalid JSON or text encoding in profile build input.")
        error.__cause__ = exc
    except ModuleNotFoundError as exc:
        error = MissingDependencyError("Profile building requires the optional HMM research toolchain.", context={"module": exc.name})
        error.__cause__ = exc
    except Exception as exc:
        error = InternalError("Unexpected profile bank build failure.", context={"cause_type": type(exc).__name__})
        error.__cause__ = exc
    return report_error(error, debug=args.debug, json_output=args.error_format == "json")
