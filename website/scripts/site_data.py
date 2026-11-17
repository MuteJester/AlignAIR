"""Generate the website's data files (``website/src/data/*.json``) from the code, so the option reference, the column
dictionary, the error codes, the Python API, the model list, the comparison with IgBLAST and the search index cannot
drift from AlignAIR. ``tests/website/test_site_data.py`` fails when a committed file differs from what this writes.

Run from the repository root, with AlignAIR importable:

    python website/scripts/site_data.py                     # the files generated from the code
    python website/scripts/site_data.py --models DIR        # also models.json, from the seven model files in DIR
    python website/scripts/site_data.py --comparison A.json [B.json ...] [--real REAL.json]
                                                            # also comparison.json, from a locked test-split run
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "website" / "src" / "data"
PAGES = ROOT / "website" / "src" / "pages" / "docs"

MODEL_IDS = {"IGH": "alignair4-igh-human", "IGK": "alignair4-igk-human", "IGL": "alignair4-igl-human",
             "TRA": "alignair4-tra-human", "TRB": "alignair4-trb-human", "TRD": "alignair4-trd-human",
             "TRG": "alignair4-trg-human"}
# The model files of the seven models, under the --models directory.
MODEL_FILES = {"IGH": "IGH-v4-seed0-cool/milestones/model.step320000.alignair",
               "IGK": "IGK-v4-seed0-cool/milestones/model.step320000.alignair",
               "IGL": "IGL-v4-seed0-cool/milestones/model.step320000.alignair",
               "TRA": "TRA-v4-seed0-cool/milestones/model.step320000.alignair",
               "TRB": "TRB-v4-seed0-invd/milestones/model.step340000.alignair",
               "TRD": "TRD-v4-seed0-invd/milestones/model.step340000.alignair",
               "TRG": "TRG-v4-seed0-cool/milestones/model.step320000.alignair"}
# The locus a germline set's name gives (HUMAN_TCRB_IMGT -> TRB).
_SET_LOCUS = {"IGH": "IGH", "IGK": "IGK", "IGL": "IGL", "TCRA": "TRA", "TCRB": "TRB", "TCRD": "TRD", "TCRG": "TRG"}
PYTHON_PACKAGES = ("alignair", "alignair.io", "alignair.registry", "alignair.reference", "alignair.train")

# What a user does about each error code.
ACTIONS = {
    "ALIGNAIR_ERROR": "Read the message; rerun with --debug for the details.",
    "ALIGNAIR_INPUT_INVALID": "Fix the input the message names (file format, sequence column, ids).",
    "ALIGNAIR_CONFIG_INVALID": "Change the option or setting the message names; its hint gives the allowed values.",
    "ALIGNAIR_DATA_INVALID": "A record broke an internal contract: rerun with --debug and report it with the traceback.",
    "ALIGNAIR_MODEL_INVALID": "Use an AlignAIR 4 model file, or check that the path names a model file.",
    "ALIGNAIR_MODEL_FORMAT_INVALID": "The file is not an AlignAIR model, or is damaged or truncated: download it again.",
    "ALIGNAIR_MODEL_NOT_FOUND": "Check the path or model id (alignair models list) and, for a Hugging Face repo, its "
                                "revision and file name.",
    "ALIGNAIR_ARTIFACT_INTEGRITY_FAILED": "The file does not match its checksum: download it again from a trusted "
                                          "source.",
    "ALIGNAIR_MODEL_INTEGRITY_FAILED": "A section of the model file is damaged: download the model again.",
    "ALIGNAIR_MODEL_SECTION_MISSING": "The model file lacks a section this command needs: use a complete model file.",
    "ALIGNAIR_REGISTRY_FAILED": "Check the registry setting (--registry, ALIGNAIR_REGISTRY) and, for a gated Hugging "
                                "Face repo, your token.",
    "ALIGNAIR_REGISTRY_UNAVAILABLE": "Check your network connection and try again; it may be a temporary outage.",
    "ALIGNAIR_REGISTRY_INVALID": "The registry's catalog is malformed: fix registry.json or use another registry.",
    "ALIGNAIR_MODEL_UNAVAILABLE": "You are offline and the model is not cached: run once online (alignair models get "
                                  "ID) or drop --offline.",
    "ALIGNAIR_DOWNLOAD_INTEGRITY_FAILED": "The download does not match its checksum: download it again; never bypass "
                                          "the check.",
    "ALIGNAIR_TRAINING_FAILED": "Read the message; rerun with --debug for the details.",
    "ALIGNAIR_TRAINING_DATA_INVALID": "The training reference or records lack required fields: check the germline "
                                      "FASTA or dataconfig.",
    "ALIGNAIR_TRAINING_NUMERICS_INVALID": "Training diverged: lower --lr or set --grad-clip, then train again.",
    "ALIGNAIR_INFERENCE_FAILED": "Rerun with --debug and report it with the traceback.",
    "ALIGNAIR_AIRR_ASSEMBLY_FAILED": "One read's AIRR fields could not be built: the row is tagged; rerun with --debug "
                                     "and report the read if it repeats.",
    "ALIGNAIR_NUMERICS_INVALID": "The model produced non-finite values: report it with the model id and the read.",
    "ALIGNAIR_DEPENDENCY_MISSING": "Install the package the message names (the hint gives the pip command).",
    "ALIGNAIR_IO_FAILED": "Check the path, its permissions and the free disk space (errno in the context).",
    "ALIGNAIR_GENOTYPE_ALLELE_NOT_FOUND": "Name alleles the model's reference holds (OGRDB names for the IG models), "
                                          "or add their sequences with --reference.",
    "ALIGNAIR_PREDICTION_COUNT_MISMATCH": "Report it with the traceback (--debug): ids, reads and predictions "
                                          "disagree.",
    "ALIGNAIR_METADATA_ID_INVALID": "The metadata table repeats an id: choose a unique --metadata-id-column.",
    "ALIGNAIR_INTERNAL_ERROR": "An unexpected failure: rerun with --debug and report it with the traceback.",
    "ALIGNAIR_INTERRUPTED": "The run was stopped (Ctrl-C); run it again.",
}


def _anchor(text: str) -> str:
    """A heading or row anchor: lower case, runs of other characters as one dash."""
    return re.sub(r"[^a-z0-9_*@.]+", "-", text.lower()).strip("-")


# --------------------------------------------------------------------------------------------- CLI options
def _option(parser, action) -> dict:
    import argparse
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse._CountAction)):
        kind = "flag"
    else:
        kind = getattr(action.type, "__name__", "text") if action.type else "text"
    default = None if action.default in (None, argparse.SUPPRESS) or kind == "flag" else action.default
    help_text = parser._get_formatter()._expand_help(action) if action.help else ""
    flags = list(action.option_strings) or [action.metavar or action.dest]
    return {"flags": flags, "metavar": action.metavar if isinstance(action.metavar, str) else None, "kind": kind,
            "default": default if isinstance(default, (str, int, float, bool)) else None,
            "choices": list(action.choices) if action.choices else None, "required": bool(action.required),
            "repeatable": isinstance(action, argparse._AppendAction), "help": help_text}


def cli_options() -> dict:
    """``{command: {"help", "groups": [{"title", "options": [...]}]}}`` for every command of ``alignair``; each
    option carries its ``anchor`` on the CLI reference page."""
    import argparse

    from alignair.cli.main import build_parser
    out: dict = {}

    def walk(parser, path: tuple, help_text: str) -> None:
        if path:
            command = " ".join(path)
            groups = []
            for group in parser._action_groups:
                options = [_option(parser, action) for action in group._group_actions
                           if not isinstance(action, (argparse._HelpAction, argparse._SubParsersAction))]
                for option in options:
                    option["anchor"] = _anchor(f"{command} {option['flags'][0].lstrip('-')}")
                if options:
                    groups.append({"title": group.title, "options": options})
            out[command] = {"help": help_text or parser.description or "", "groups": groups}
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                helps = {choice.dest: choice.help for choice in action._choices_actions}
                for name, child in action.choices.items():
                    walk(child, (*path, name), helps.get(name) or "")
    walk(build_parser(), (), "")
    return out


# ------------------------------------------------------------------------------------- output columns
def output_columns() -> list:
    """Every column of the AIRR table AlignAIR writes: its kind, unit, meaning and the presets that include it."""
    from alignair.io.airr import COLUMN_PRESETS
    from alignair.io.columns_doc import COLUMN_DOCS
    return [{"name": name, "kind": doc.kind, "unit": doc.unit, "meaning": doc.meaning,
             "presets": [preset for preset, columns in COLUMN_PRESETS.items() if name in columns]}
            for name, doc in COLUMN_DOCS.items()]


# ---------------------------------------------------------------------------------------- error codes
def _example(cls):
    try:
        return cls("example")
    except TypeError:
        return cls("example", ValueError("example"))


def error_codes() -> list:
    """Every error class of ``alignair.errors``: its code, category, exit code, whether a retry may succeed, what it
    means (its docstring) and what to do (``ACTIONS``)."""
    from alignair import errors
    entries = []
    for name in errors.__all__:
        cls = getattr(errors, name)
        if not (inspect.isclass(cls) and issubclass(cls, errors.AlignAIRError)):
            continue
        meaning = (inspect.getdoc(cls) or "").split("\n\n")[0].replace("\n", " ")
        entries.append({"code": cls.code, "class": name, "category": cls.category, "exit_code": int(cls.exit_code),
                        "retryable": bool(_example(cls).retryable), "meaning": meaning,
                        "action": ACTIONS.get(cls.code, ""), "anchor": _anchor(cls.code)})
    return sorted(entries, key=lambda entry: (entry["code"], entry["class"]))


# ----------------------------------------------------------------------------------------- Python API
def _signature(obj) -> str:
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return ""
    parts, star = [], False
    for parameter in signature.parameters.values():
        if parameter.name in ("self", "cls"):
            continue
        if parameter.kind is parameter.KEYWORD_ONLY and not star:
            parts.append("*")
            star = True
        prefix = {parameter.VAR_POSITIONAL: "*", parameter.VAR_KEYWORD: "**"}.get(parameter.kind, "")
        text = prefix + parameter.name
        if parameter.VAR_POSITIONAL == parameter.kind:
            star = True
        if parameter.annotation is not parameter.empty:
            text += f": {parameter.annotation if isinstance(parameter.annotation, str) else inspect.formatannotation(parameter.annotation)}"
        if parameter.default is not parameter.empty:
            default = parameter.default
            text += " = " + (repr(default) if isinstance(default, (str, int, float, bool, type(None), tuple))
                             else default.__name__ if callable(default) and hasattr(default, "__name__")
                             else type(default).__name__ + "()")
        parts.append(text)
    returns = signature.return_annotation
    tail = "" if returns is signature.empty else f" -> {returns if isinstance(returns, str) else inspect.formatannotation(returns)}"
    return f"({', '.join(parts)}){tail}"


def python_api() -> list:
    """The public Python API: each public name of the user packages, its signature and docstring, and the public
    methods and properties of its classes."""
    entries = []
    for package_name in PYTHON_PACKAGES:
        package = importlib.import_module(package_name)
        for name in package.__all__:
            value = getattr(package, name)
            qualified = f"{package_name}.{name}"
            if inspect.isclass(value):
                members = []
                for attribute, member in vars(value).items():
                    if attribute.startswith("_") or attribute == "forward":
                        continue
                    function = member.fget if isinstance(member, property) else getattr(member, "__func__", member)
                    if inspect.isfunction(function):
                        members.append({"name": attribute, "kind": "property" if isinstance(member, property)
                                        else "method", "signature": "" if isinstance(member, property)
                                        else _signature(function), "doc": inspect.getdoc(function) or ""})
                entries.append({"name": qualified, "kind": "class", "signature": _signature(value),
                                "doc": inspect.getdoc(value) or "", "members": members, "anchor": _anchor(qualified)})
            elif inspect.isfunction(value):
                entries.append({"name": qualified, "kind": "function", "signature": _signature(value),
                                "doc": inspect.getdoc(value) or "", "members": [], "anchor": _anchor(qualified)})
            elif isinstance(value, (str, int, float)):
                entries.append({"name": qualified, "kind": "constant", "signature": f" = {value!r}", "doc": "",
                                "members": [], "anchor": _anchor(qualified)})
    return entries


# --------------------------------------------------------------------------------------------- models
def models(directory: Path) -> list:
    """The seven pretrained models, from their model files under ``directory`` (``MODEL_FILES``)."""
    from alignair.model_file import embedded_allele_counts, read_metadata
    out = []
    for locus, model_id in MODEL_IDS.items():
        path = directory / MODEL_FILES[locus]
        metadata = read_metadata(path)
        dataconfig = metadata["reference"]["dataconfigs"][0]
        counts = embedded_allele_counts(path)
        out.append({"id": model_id, "locus": locus, "chain": dataconfig["chain_type"],
                    "species": dataconfig["species"], "germline_set": dataconfig["name"],
                    "naming": "OGRDB" if dataconfig["name"].endswith("OGRDB") else "IMGT",
                    "alleles": {"V": counts.get("v", 0), "D": counts.get("d", 0), "J": counts.get("j", 0)},
                    "training_steps": metadata["training"]["steps"], "licence": metadata["license"],
                    "anchor": _anchor(model_id)})
    return out


# ------------------------------------------------------------------------------------------ comparison
def comparison(simulated: list[Path], real: Path | None) -> dict:
    """The comparison with IgBLAST from a locked test-split run: the assay's run reports (IgBLAST against AlignAIR,
    one per locus) and, optionally, its real-data report. A ValueError for a development-split run or another pair
    of tools: development numbers are never published."""
    loci = {}
    for path in simulated:
        report = json.loads(Path(path).read_text())
        if report.get("kind") != "alignair_assay.run_report":
            raise ValueError(f"{path}: not an assay run report")
        if report.get("a") != "igblast" or not str(report.get("b", "")).startswith("alignair"):
            raise ValueError(f"{path}: compares {report.get('a')} with {report.get('b')}, not igblast with alignair")
        if report["config"].get("split") != "test":
            raise ValueError(f"{path}: a {report['config'].get('split')!r} split run; only the locked test split "
                             "is published")
        reference = report["config"]["reference"]
        locus = _SET_LOCUS[reference.split("_")[1]]
        suites = []
        for suite in report["suites"]:
            arm = suite.get("arms", {}).get("shared", {})
            if not arm.get("compared"):
                continue
            rows = [{"measure": measure, "igblast": row["a"]["mean"], "alignair": row["b"]["mean"],
                     "delta": row["delta"], "ci95": row["ci95"], "verdict": row["verdict"]}
                    for measure, row in arm["rows"].items()]
            suites.append({"name": suite["name"], "title": suite.get("title", ""), "cases": suite.get("cases"),
                           "rows": rows})
        loci[locus] = {"reference": reference, "suites": suites}
    out = {"status": "ready", "split": "test", "loci": loci}
    if real is not None:
        report = json.loads(Path(real).read_text())
        if report.get("config", {}).get("split") not in (None, "test"):
            raise ValueError(f"{real}: not a test-split real-data report")
        out["real"] = report
    return out


# ---------------------------------------------------------------------------------------- search index
_PAGE = re.compile(r'slug:\s*"([^"]+)"')
_FIELD = r'{}:\s*"([^"]+)"'
_HEADING = re.compile(r'<H[23]\s+id="([^"]+)"\s*>(.*?)</H[23]>', re.DOTALL)


def _text(markup: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>|[{}]", "", markup)).strip()


def search_index() -> list:
    """What the search box finds: every docs page and its headings (from the page sources), every CLI option,
    output column, model and error code; each with the link it opens."""
    entries = []
    for source in sorted(PAGES.glob("pages-*.tsx")):
        text = source.read_text()
        starts = [match.start() for match in _PAGE.finditer(text)] + [len(text)]
        for begin, end in itertools.pairwise(starts):
            chunk = text[begin:end]
            slug = _PAGE.match(chunk).group(1)
            title = re.search(_FIELD.format("title"), chunk)
            section = re.search(_FIELD.format("section"), chunk)
            title = title.group(1) if title else slug
            entries.append({"kind": "page", "title": title, "text": section.group(1) if section else "",
                            "href": f"#/docs/{slug}"})
            entries += [{"kind": "heading", "title": _text(heading), "text": title,
                         "href": f"#/docs/{slug}?h={anchor}"} for anchor, heading in _HEADING.findall(chunk)]
    for command, entry in cli_options().items():
        for group in entry["groups"]:
            entries += [{"kind": "option", "title": ", ".join(option["flags"]),
                         "text": f"alignair {command}: {option['help'][:140]}",
                         "href": f"#/docs/cli?h={option['anchor']}"} for option in group["options"]]
    entries += [{"kind": "column", "title": column["name"], "text": column["meaning"][:140],
                 "href": f"#/docs/column-dictionary?h={column['name']}"} for column in output_columns()]
    models_file = DATA / "models.json"
    if models_file.exists():
        entries += [{"kind": "model", "title": model["id"], "text": f"{model['locus']} · {model['germline_set']}",
                     "href": f"#/docs/model-cards?h={model['anchor']}"} for model in json.loads(models_file.read_text())]
    entries += [{"kind": "error", "title": error["code"], "text": error["meaning"][:140],
                 "href": f"#/docs/error-codes?h={error['anchor']}"} for error in error_codes()]
    return entries


GENERATED = {"cli-options": cli_options, "output-columns": output_columns, "error-codes": error_codes,
             "python-api": python_api, "search-index": search_index}


def _json(value) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def render(name: str) -> str:
    """The text of the generated data file ``name`` (``GENERATED``)."""
    return _json(GENERATED[name]())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", type=Path, help="the directory holding the seven model files (MODEL_FILES)")
    parser.add_argument("--comparison", type=Path, nargs="+", help="the locked test-split run reports")
    parser.add_argument("--real", type=Path, help="the locked test-split real-data report")
    args = parser.parse_args(argv)
    DATA.mkdir(parents=True, exist_ok=True)
    if args.models:
        (DATA / "models.json").write_text(_json(models(args.models)))
    if args.comparison:
        (DATA / "comparison.json").write_text(_json(comparison(args.comparison, args.real)))
    elif not (DATA / "comparison.json").exists():
        (DATA / "comparison.json").write_text(_json({"status": "pending"}))
    for name in GENERATED:                                # the search index last: it reads models.json
        (DATA / f"{name}.json").write_text(render(name))
        print(f"wrote {DATA / name}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
