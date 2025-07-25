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

from AlignAIR.PredictObject.PredictObject import PredictObject
from AlignAIR.Step.Step import Step
from AlignAIR.Preprocessing.Steps.batch_processing_steps import BatchProcessingStep
from AlignAIR.Preprocessing.Steps.dataconfig_steps import ConfigLoadStep
from AlignAIR.Preprocessing.Steps.file_steps import FileNameExtractionStep, FileSampleCounterStep
from AlignAIR.Preprocessing.Steps.model_loading_steps import ModelLoadingStep
from AlignAIR.PostProcessing.Steps.clean_up_steps import CleanAndArrangeStep
from AlignAIR.PostProcessing.Steps.segmentation_correction_steps import SegmentCorrectionStep
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
    model_checkpoint: Optional[str] = typer.Option(None, help="Path to saved weights"),
    save_path: Optional[str] = typer.Option(None, help="Where to write alignment output"),
    sequences: Optional[str] = typer.Option(None, help="CSV/TSV/FASTA containing sequences ('sequence' column). For multi-chain analysis, use comma-separated paths."),
    genairr_dataconfig: Optional[str] = typer.Option(None, help="GenAIRR DataConfig identifier(s). Single: 'HUMAN_IGH_OGRDB'. Multi-chain: 'HUMAN_IGH_OGRDB,HUMAN_IGK_OGRDB,HUMAN_IGL_OGRDB'. Available: HUMAN_IGH_OGRDB, HUMAN_IGK_OGRDB, HUMAN_IGL_OGRDB, HUMAN_TCRB_IMGT"),
    # Performance
    batch_size: Optional[int] = typer.Option(None),
    max_input_size: Optional[int] = typer.Option(None),
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
    # Legacy support (deprecated - use genairr_dataconfig instead)
    chain_type: Optional[str] = typer.Option(None, help="[DEPRECATED] Use genairr_dataconfig instead. heavy / light"),
    lambda_data_config: Optional[str] = typer.Option(None, help="[DEPRECATED] Use genairr_dataconfig instead"),
    kappa_data_config: Optional[str] = typer.Option(None, help="[DEPRECATED] Use genairr_dataconfig instead"),
    heavy_data_config: Optional[str] = typer.Option(None, help="[DEPRECATED] Use genairr_dataconfig instead"),
    # Misc paths
    custom_orientation_pipeline_path: Optional[str] = typer.Option(None),
    custom_genotype: Optional[str] = typer.Option(None),
    finetuned_model_params_yaml: Optional[str] = typer.Option(None),
    save_predict_object: bool = typer.Option(False),
):
    """Run the AlignAIR v2.0 pipeline with unified single/multi-chain architecture."""
    header()
    system_stats()

    cfg_dict = {}
    if config_file:
        cfg_dict.update(yaml.safe_load(config_file.read_text()))

    defaults = {
        "model_checkpoint": None,
        "save_path": None,
        "sequences": None,
        "genairr_dataconfig": "HUMAN_IGH_OGRDB",  # Default single heavy chain
        "batch_size": 2048,
        "max_input_size": 576,
        "v_allele_threshold": 0.1,
        "d_allele_threshold": 0.1,
        "j_allele_threshold": 0.1,
        "v_cap": 3,
        "d_cap": 3,
        "j_cap": 3,
        "translate_to_asc": False,
        "airr_format": False,
        "fix_orientation": True,
        # Legacy support - deprecated in v2.0 (will be converted to genairr_dataconfig)
        "chain_type": None,
        "lambda_data_config": "D",
        "kappa_data_config": "D", 
        "heavy_data_config": "D",
        "custom_orientation_pipeline_path": None,
        "custom_genotype": None,
        "finetuned_model_params_yaml": None,
        "save_predict_object": False,
    }

    # Build dict of non‑None overrides from function locals excluding cfg_dict/config_file
    overrides = {
        k: v for k, v in locals().items()
        if k not in {"cfg_dict", "config_file"} and v is not None
    }

    # Merge: defaults < config_file < CLI overrides
    cfg_dict = {**defaults, **cfg_dict, **overrides}

    # v2.0 validation and warnings
    _validate_v2_config(cfg_dict)

    # SimpleNamespace keeps it lightweight and attribute‑style
    from types import SimpleNamespace
    cfg = SimpleNamespace(**cfg_dict)

    _run_pipeline(cfg)


def _validate_v2_config(cfg_dict):
    """Validate v2.0 configuration and show deprecation warnings."""
    # Show deprecation warnings for legacy parameters
    legacy_params = {
        'chain_type': 'genairr_dataconfig',
        'heavy_data_config': 'genairr_dataconfig', 
        'kappa_data_config': 'genairr_dataconfig',
        'lambda_data_config': 'genairr_dataconfig'
    }
    
    for old_param, new_param in legacy_params.items():
        if cfg_dict.get(old_param) and cfg_dict[old_param] not in [None, "D"]:
            warnings.warn(
                f"Parameter '{old_param}' is deprecated in AlignAIR v2.0. "
                f"Use '{new_param}' instead for better performance and multi-chain support.",
                DeprecationWarning,
                stacklevel=3
            )
    
    # Validate genairr_dataconfig
    valid_configs = {
        'HUMAN_IGH_OGRDB', 'HUMAN_IGK_OGRDB', 
        'HUMAN_IGL_OGRDB', 'HUMAN_TCRB_IMGT'
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


