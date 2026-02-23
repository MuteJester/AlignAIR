from __future__ import annotations
import os
import warnings
# 0 = all logs | 1 = filter INFO | 2 = filter WARNING | 3 = filter ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
import pathlib
from typing import Optional
import typer
from rich import box
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import yaml
import psutil
import platform
import tensorflow as tf

from AlignAIR.Hub import AVAILABLE_MODELS, list_available_models, get_model_path, download_model, is_model_cached
from AlignAIR.PredictObject.PredictObject import PredictObject
from AlignAIR.Step.Step import Step
from AlignAIR.Preprocessing.Steps.batch_processing_steps import BatchProcessingStep
from AlignAIR.Preprocessing.Steps.dataconfig_steps import ConfigLoadStep
from AlignAIR.Preprocessing.Steps.file_steps import FileNameExtractionStep, FileSampleCounterStep
from AlignAIR.Preprocessing.Steps.model_loading_steps import ModelLoadingStep
from AlignAIR.PostProcessing.Steps.clean_up_steps import CleanAndArrangeStep
from AlignAIR.PostProcessing.Steps.segmentation_correction_steps import SegmentCorrectionStep
from AlignAIR.PostProcessing.Steps.correct_likelihood_for_genotype_step import GenotypeBasedLikelihoodAdjustmentStep
from AlignAIR.PostProcessing.Steps.allele_threshold_step import MaxLikelihoodPercentageThresholdApplicationStep
from AlignAIR.PostProcessing.Steps.translate_to_imgt_step import TranslationStep
from AlignAIR.PostProcessing.Steps.germline_alignment_steps import AlleleAlignmentStep
from AlignAIR.PostProcessing.Steps.finalization_and_packaging_steps import FinalizationStep
from AlignAIR.PostProcessing.Steps.airr_finalization_and_packaging_steps import AIRRFinalizationStep

app = typer.Typer(add_completion=False)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
tf.get_logger().setLevel("ERROR")

# # ANSI color codes for styling
BLUE = "\033[94m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"
ASCII_ART = f"""
{BLUE}` .     ..  .                            .                      .       .
`                    ..   .                                    ..  .     .   ..
                                           .   .    .
                                  . .                                  .   . .
                     .                     ..    .              . .     .  .`
      .       .   .            .          .      .                    .       .
          .           /.         .      .               /     .
               ... .%#//%.    ..                   .  ##//%*   .              .
     .       ..*#/////////%,       .                %#/////////%       .      .
..          .  %(///////////%*.           .       %#///////////%*    .
   .       .#  .%#////////////%/              ..%(////////////%.   /*   .
 .. .  (#////#% ..%#////////////%(          ..%(////////////%.   (%/////%  .
 .  .  .%//////(% .  %#////////////%(       .%/////////////%.   #%//////#/
  ..   /#////////(%   %#////////////%#   ,%/////////////%,   ##////////(#    .
    . (#////////////%. .%(///////////%*  %////////////%, . %#///////////(%. .  .
 .      ##////////////%. .%(/////////%*  %//////////%*...%#///////////(%.
      ..  ##////////////%, /#////////%*  %/////////%  .%(///////////(%. ..
 .      ..  #%///////////#%/#////////%,  %/////////% (%///////////(%          ..
             .(%///////#%. /#////////%,  %/////////%   /%///////#% ..        . .
                (%///#%    /#////////%,  %/////////%     /%///#% .      .  .
          .       /%#      /#////////%.  %/////////%       /%%       .  .
    .  .                   /#////////%.  %/////////%     .               .
     .                     /#////////%.  %/////////%.      .   .  .    .      .
. .                .  .    /#////////%.  %/////////%     .  ..
.  .  .     ..  .. .. .    /#////////%.  %/////////%                        .
     .       ..            /#////////%.  %/////////%              .
. . ....                  ./#////////%. .%/////////%           ...    .      .
                 .    .    /#////////%.  %/////////%          ..
  . ...  .           ..    *%////////%.  ##///////(#..  ...               ..  .
 ..             ..         ..%%%%%%%%  . .#%%%%%%%(           . .    .     .
    .      .       .         . ..%            #,        .      .
  .    .   ..   ..   .       .   %            #,.                    .
   . .   .                .      %        .   #, .
         ..          .          .%.           #,        .             .
    .    .  .   .     . ..  .   .% ..      .. #,     .
{RESET}
"""
# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def _human(bytes_: int, prec: int = 2) -> str:
    return f"{bytes_ / 1024 ** 3:.{prec}f} GB"


def header() -> None:
    panel = Panel("AlignAIR v2.0", subtitle="Unified Multi-Chain Architecture", style="bold cyan", expand=False)
    console.print(Panel.fit(ASCII_ART, title="[bold cyan]AlignAIR v2.0 - Unified Architecture", border_style="cyan"))
    console.print("[bold yellow]🔬 v2.0 Features:[/bold yellow] [green]SingleChainAlignAIR & MultiChainAlignAIR with GenAIRR Integration[/green]")


def system_stats() -> None:
    tbl = Table(box=box.SIMPLE, highlight=True)
    tbl.add_column("Key", style="bold")
    tbl.add_column("Value")
    tbl.add_row("Processes", str(len(psutil.pids())))
    tbl.add_row("Free RAM", _human(psutil.virtual_memory().available))
    gpus = tf.config.list_physical_devices("GPU")
    tbl.add_row("GPUs", ", ".join(d.name for d in gpus) or "None")
    cpus = tf.config.list_physical_devices("CPU")
    tbl.add_row("CPUs", ", ".join(d.name for d in cpus))
    tbl.add_row("OS", platform.platform())
    tbl.add_row("Python", platform.python_version())
    console.print(tbl)


def _build_steps(airr: bool):
    steps = [
        ConfigLoadStep("Load Config"),
        FileNameExtractionStep("Get File Name"),
        FileSampleCounterStep("Count Samples"),
        ModelLoadingStep("Load Models"),
        BatchProcessingStep("Predict Batches"),
        CleanAndArrangeStep("Clean Raw Predictions"),
        GenotypeBasedLikelihoodAdjustmentStep("Adjust Likelihoods for Genotype"),
        SegmentCorrectionStep("Correct Segments"),
        MaxLikelihoodPercentageThresholdApplicationStep("Apply Thresholds"),
        AlleleAlignmentStep("Align With Germline"),
    ]
    if airr:
        steps.append(AIRRFinalizationStep("Finalize (AIRR)"))
    else:
        steps.extend([
            TranslationStep("Translate to IMGT"),
            FinalizationStep("Finalize Results"),
        ])
    return steps


def _run_pipeline(cfg):
    logger = logging.getLogger("AlignAIR")
    Step.set_logger(logger)
    po = PredictObject(cfg, logger=logger)
    for s in _build_steps(cfg.airr_format):
        po = s.execute(po)
    logger.info("✅ Alignment finished – results stored at %s", cfg.save_path)

# ────────────────────────────────────────────────────────────────────────────────
# Commands
# ────────────────────────────────────────────────────────────────────────────────

@app.command()
def doctor():
    """Print system diagnostics and exit."""
    header()
    system_stats()


@app.command()
def run(
    # General
    config_file: Optional[pathlib.Path] = typer.Option(None, help="YAML with parameters (flags override)"),
    model_dir: Optional[str] = typer.Option(None, help="Path to a pretrained bundle directory (preferred). If provided, overrides --model-checkpoint."),
    model_checkpoint: Optional[str] = typer.Option(None, help="Path to saved weights"),
    save_path: Optional[str] = typer.Option(None, help="Where to write alignment output"),
    sequences: Optional[str] = typer.Option(None, help="CSV/TSV/FASTA containing sequences ('sequence' column). For multi-chain analysis, use comma-separated paths."),
    genairr_dataconfig: Optional[str] = typer.Option(
        None,
        help=(
            "GenAIRR DataConfig identifier(s). "
            "Single: 'HUMAN_IGH_OGRDB' (or 'HUMAN_IGH_EXTENDED'). "
            "Multi-chain: comma-separated (e.g. 'HUMAN_IGL_OGRDB,HUMAN_IGK_OGRDB'). "
            "Available: HUMAN_IGH_OGRDB, HUMAN_IGH_EXTENDED, HUMAN_IGK_OGRDB, HUMAN_IGL_OGRDB, HUMAN_TCRB_IMGT"
        ),
    ),
    # Performance
    batch_size: Optional[int] = typer.Option(None),
    # Thresholds
    v_allele_threshold: Optional[float] = typer.Option(None),
    d_allele_threshold: Optional[float] = typer.Option(None),
    j_allele_threshold: Optional[float] = typer.Option(None),
    v_cap: Optional[int] = typer.Option(None),
    d_cap: Optional[int] = typer.Option(None),
    j_cap: Optional[int] = typer.Option(None),
    # Flags
    translate_to_asc: bool = typer.Option(False),
    airr_format: bool = typer.Option(False),
    fix_orientation: bool = typer.Option(True),
    # Misc paths
    custom_orientation_pipeline_path: Optional[str] = typer.Option(None),
    custom_genotype: Optional[str] = typer.Option(None),
    save_predict_object: bool = typer.Option(False),
):
    """Run the AlignAIR v2.0 pipeline with unified single/multi-chain architecture."""
    header()
    system_stats()

    cfg_dict = {}
    if config_file:
        cfg_dict.update(yaml.safe_load(config_file.read_text()))

    defaults = {
        "model_dir": None,
        "model_checkpoint": None,
        "save_path": None,
        "sequences": None,
        "genairr_dataconfig": "HUMAN_IGH_OGRDB",  # Default single heavy chain
        "batch_size": 2048,
    "v_allele_threshold": 0.75,
    "d_allele_threshold": 0.30,
    "j_allele_threshold": 0.80,
        "v_cap": 3,
        "d_cap": 3,
        "j_cap": 3,
        "translate_to_asc": False,
        "airr_format": False,
        "fix_orientation": True,
        "custom_orientation_pipeline_path": None,
        "custom_genotype": None,
        "save_predict_object": False,
    }

    # Build dict of non‑None overrides from function locals excluding cfg_dict/config_file
    overrides = {
        k: v for k, v in locals().items()
        if k not in {"cfg_dict", "config_file"} and v is not None
    }

    # Merge: defaults < config_file < CLI overrides
    cfg_dict = {**defaults, **cfg_dict, **overrides}

    # Prefer bundle directory over legacy checkpoint
    if not cfg_dict.get("model_dir") and not cfg_dict.get("model_checkpoint"):
        raise typer.BadParameter("One of --model-dir (preferred) or --model-checkpoint must be provided")
    if cfg_dict.get("model_dir"):
        model_ref = cfg_dict["model_dir"]
        p = pathlib.Path(model_ref).resolve()

        # If not a local directory, check if it's a known HuggingFace model name
        if not p.is_dir() and model_ref in AVAILABLE_MODELS:
            console.print(f"[blue]Resolving model name '{model_ref}' from HuggingFace Hub...[/blue]")
            p = get_model_path(model_ref)
            console.print(f"[green]📦 Model resolved:[/green] {p}")
            cfg_dict["model_dir"] = str(p)

        # Map to model_checkpoint for ModelLoadingStep compatibility
        cfg_dict["model_checkpoint"] = cfg_dict["model_dir"]
        # Confirm bundle structure if possible
        try:
            p = pathlib.Path(cfg_dict["model_dir"]).resolve()
            if p.is_dir() and (p / "config.json").exists():
                console.print(f"[green]📦 Detected pretrained bundle:[/green] {p}")
                # When loading from a bundle, the DataConfig inside the bundle is the source of truth
                if cfg_dict.get("genairr_dataconfig"):
                    console.print("[yellow]⚠️  Ignoring --genairr_dataconfig: bundle provides its own DataConfig.[/yellow]")
            else:
                console.print(f"[yellow]⚠️  --model-dir provided but no config.json found at:[/yellow] {p} — falling back to legacy loader")
        except Exception:
            pass

    # v2.0 validation and warnings
    _validate_v2_config(cfg_dict)

    # SimpleNamespace keeps it lightweight and attribute‑style
    from types import SimpleNamespace
    cfg = SimpleNamespace(**cfg_dict)

    _run_pipeline(cfg)


@app.command()
def download(
    model_name: Optional[str] = typer.Argument(None, help="Model bundle name to download (e.g. IGH_S5F_576)"),
    all_models: bool = typer.Option(False, "--all", help="Download all available pretrained models"),
    list_models: bool = typer.Option(False, "--list", help="List available models without downloading"),
    revision: Optional[str] = typer.Option(None, help="HuggingFace repo revision (tag/branch/commit)"),
    cache_dir: Optional[pathlib.Path] = typer.Option(None, help="Custom local cache directory (default: ~/.alignair/models)"),
    force: bool = typer.Option(False, help="Re-download even if already cached"),
):
    """Download pretrained AlignAIR model bundles from HuggingFace Hub."""
    header()

    if list_models:
        tbl = Table(box=box.SIMPLE_HEAVY)
        tbl.add_column("Model", style="bold cyan")
        tbl.add_column("Cached", style="bold")
        for name in list_available_models():
            cached = is_model_cached(name, revision, cache_dir)
            status = "[green]Yes[/green]" if cached else "[dim]No[/dim]"
            tbl.add_row(name, status)
        console.print(tbl)
        return

    if all_models:
        names = list_available_models()
    elif model_name:
        names = [model_name]
    else:
        console.print("[red]Provide a model name, --all, or --list.[/red]")
        console.print(f"[blue]Available models:[/blue] {', '.join(list_available_models())}")
        raise typer.Exit(code=1)

    for name in names:
        try:
            if not force and is_model_cached(name, revision, cache_dir):
                console.print(f"[green]Already cached:[/green] {name}")
                continue
            console.print(f"[blue]Downloading:[/blue] {name} ...")
            path = download_model(name, revision=revision, cache_dir=cache_dir, force=force)
            console.print(f"[green]Ready:[/green] {name} -> {path}")
        except Exception as exc:
            console.print(f"[red]Failed to download {name}:[/red] {exc}")

    console.print("[bold green]Done.[/bold green]")


@app.command("list-pretrained")
def list_pretrained(
    root: pathlib.Path = typer.Option(pathlib.Path("/app/pretrained_models"), help="Directory containing pretrained model bundles"),
    show_files: bool = typer.Option(False, help="Show expected key files for each bundle"),
    json_output: bool = typer.Option(False, help="Emit machine-readable JSON instead of a table"),
):
    """List available pretrained model bundles baked into the image (or a custom root).

    A valid bundle minimally contains: config.json, dataconfig.pkl, VERSION, fingerprint.txt, saved_model/.
    """
    header()
    if not root.exists() or not root.is_dir():
        console.print(f"[red]No pretrained models found: directory does not exist[/red] {root}")
        raise typer.Exit(code=1)

    bundles = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        # quick structural validation
        required = ["config.json", "dataconfig.pkl", "VERSION", "fingerprint.txt", "saved_model"]
        present = {name: (p / name).exists() for name in required}
        is_valid = all(present.values()) and (p / "saved_model").is_dir()
        meta = {
            "name": p.name,
            "path": str(p.resolve()),
            "valid": is_valid,
            "missing": [k for k, ok in present.items() if not ok],
        }
        # Try to parse config for quick summary
        cfg_path = p / "config.json"
        if cfg_path.exists():
            try:
                import json as _json
                raw = _json.loads(cfg_path.read_text())
                meta.update({
                    "model_type": raw.get("model_type"),
                    "max_seq_length": raw.get("max_seq_length"),
                    "format_version": raw.get("format_version"),
                    "chains": raw.get("chain_types"),
                })
            except Exception:  # pragma: no cover - best effort
                pass
        bundles.append(meta)

    if json_output:
        import json as _json
        console.print(_json.dumps(bundles, indent=2))
        return

    tbl = Table(box=box.SIMPLE_HEAVY)
    tbl.add_column("Bundle", style="bold cyan")
    tbl.add_column("Type")
    tbl.add_column("SeqLen")
    tbl.add_column("Chains")
    tbl.add_column("Status")
    for b in bundles:
        status = "[green]OK[/green]" if b["valid"] else f"[red]Missing: {','.join(b['missing'])}[/red]"
        chains_display = ",".join(b.get("chains") or []) if b.get("chains") else "-"
        tbl.add_row(b["name"], str(b.get("model_type") or "?"), str(b.get("max_seq_length") or "-"), chains_display, status)
    console.print(tbl)

    if show_files:
        for b in bundles:
            console.print(f"\n[bold]{b['name']}[/bold] -> {b['path']}")
            for fname in ["config.json", "dataconfig.pkl", "VERSION", "fingerprint.txt", "saved_model/"]:
                exists = (pathlib.Path(b['path']) / fname.rstrip('/')).exists()
                console.print(f"  {'[green]✓[/green]' if exists else '[red]✗[/red]'} {fname}")

    console.print(f"[blue]Found {len(bundles)} bundle(s) in {root}[/blue]")


def _validate_v2_config(cfg_dict):
    """Validate configuration: check known built-in GenAIRR dataconfig names and print helpful info."""
    # Validate genairr_dataconfig
    valid_configs = {
        'HUMAN_IGH_OGRDB', 'HUMAN_IGH_EXTENDED',
        'HUMAN_IGK_OGRDB', 'HUMAN_IGL_OGRDB', 'HUMAN_TCRB_IMGT'
    }
    
    genairr_config = cfg_dict.get('genairr_dataconfig', '')
    if genairr_config:
        configs = [c.strip() for c in genairr_config.split(',')]
        for config in configs:
            if config not in valid_configs and not config.endswith('.pkl'):
                console.print(f"[yellow]⚠️  Warning:[/yellow] '{config}' is not a recognized built-in GenAIRR dataconfig.")
                console.print(f"[yellow]   Available built-in configs: {', '.join(valid_configs)}[/yellow]")
                console.print(f"[yellow]   Or provide a path to a custom .pkl dataconfig file.[/yellow]")
        
        # Show info about multi-chain detection
        if len(configs) > 1:
            console.print(f"[green]🔗 Multi-chain analysis detected:[/green] {len(configs)} dataconfigs")
            console.print(f"[green]   Will use MultiChainAlignAIR model automatically[/green]")
        else:
            console.print(f"[blue]🔍 Single-chain analysis:[/blue] {configs[0]}")
            console.print(f"[blue]   Will use SingleChainAlignAIR model automatically[/blue]")

# ────────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        app()
    except Exception as exc:  # pragma: no cover
        console.print(f"[bold red]🛑  Error:[/bold red] {exc}")
        raise


