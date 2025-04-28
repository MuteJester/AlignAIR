from __future__ import annotations
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings(
    "ignore",
    category=InconsistentVersionWarning,
)
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
    panel = Panel("AlignAIR", subtitle="CLI", style="bold cyan", expand=False)
    console.print(Panel.fit(ASCII_ART, title="[bold cyan]AlignAIR", border_style="cyan"))


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
    chain_type: Optional[str] = typer.Option(None, help="heavy / light"),
    sequences: Optional[str] = typer.Option(None, help="CSV/TSV/FASTA containing sequences ('sequence' column)"),
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
    # Misc paths
    lambda_data_config: Optional[str] = typer.Option(None),
    kappa_data_config: Optional[str] = typer.Option(None),
    heavy_data_config: Optional[str] = typer.Option(None),
    custom_orientation_pipeline_path: Optional[str] = typer.Option(None),
    custom_genotype: Optional[str] = typer.Option(None),
    finetuned_model_params_yaml: Optional[str] = typer.Option(None),
    save_predict_object: bool = typer.Option(False),
):
    """Run the AlignAIR pipeline."""
    header()
    system_stats()

    cfg_dict = {}
    if config_file:
        cfg_dict.update(yaml.safe_load(config_file.read_text()))

    defaults = {
        "model_checkpoint": None,
        "save_path": None,
        "chain_type": None,
        "sequences": None,
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

    # SimpleNamespace keeps it lightweight and attribute‑style
    from types import SimpleNamespace
    cfg = SimpleNamespace(**cfg_dict)

    _run_pipeline(cfg)

# ────────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        app()
    except Exception as exc:  # pragma: no cover
        console.print(f"[bold red]🛑  Error:[/bold red] {exc}")
        raise


