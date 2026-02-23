"""AlignAIR 3.0 Pipeline — CLI entry point."""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

from AlignAIR.Pipeline.Models.config import (
    PipelineConfig,
    AlleleThresholdConfig,
    OrientationConfig,
    MemoryConfig,
    ReproducibilityConfig,
)
from AlignAIR.Pipeline.Models.enums import OutputFormat
from AlignAIR.Pipeline.Runner.runner import AlignAIRPipeline
from AlignAIR.Pipeline.Runner.assembly import assemble_pipeline
from AlignAIR.Pipeline.Observability.logger import PipelineLogger
from AlignAIR.Pipeline.Reproducibility.determinism import set_deterministic
from AlignAIR.Pipeline.Reproducibility.environment import EnvironmentFingerprint
from AlignAIR.Pipeline.Reproducibility.provenance import RunProvenance, file_sha256


def build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AlignAIR 3.0 - Immunoglobulin Sequence Alignment Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_dir", type=str, required=True, help="Path to model bundle directory")
    p.add_argument("--sequences", type=str, required=True, help="Path to input sequences file")
    p.add_argument("--save_path", type=str, required=True, help="Output directory or file path")
    p.add_argument("--batch_size", type=int, default=2048, help="Batch size for inference")
    p.add_argument("--v_allele_threshold", type=float, default=0.1)
    p.add_argument("--d_allele_threshold", type=float, default=0.1)
    p.add_argument("--j_allele_threshold", type=float, default=0.1)
    p.add_argument("--v_cap", type=int, default=3)
    p.add_argument("--d_cap", type=int, default=3)
    p.add_argument("--j_cap", type=int, default=3)
    p.add_argument("--translate_to_asc", action="store_true", default=True)
    p.add_argument("--fix_orientation", type=bool, default=True)
    p.add_argument("--custom_orientation_pipeline_path", type=str, default=None)
    p.add_argument("--custom_genotype", type=str, default=None)
    p.add_argument("--airr_format", action="store_true", help="Output in AIRR format")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_provenance", action="store_true", help="Save provenance JSON alongside output")
    return p


def config_from_cli(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        model_dir=args.model_dir,
        sequences_path=args.sequences,
        save_path=args.save_path,
        output_format=OutputFormat.AIRR if args.airr_format else OutputFormat.CSV,
        translate_to_asc=args.translate_to_asc,
        custom_genotype_path=args.custom_genotype,
        thresholds=AlleleThresholdConfig(
            v_threshold=args.v_allele_threshold,
            d_threshold=args.d_allele_threshold,
            j_threshold=args.j_allele_threshold,
            v_cap=args.v_cap,
            d_cap=args.d_cap,
            j_cap=args.j_cap,
        ),
        orientation=OrientationConfig(
            enabled=args.fix_orientation,
            custom_model_path=args.custom_orientation_pipeline_path,
        ),
        memory=MemoryConfig(batch_size=args.batch_size),
        reproducibility=ReproducibilityConfig(seed=args.seed),
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    log = logging.getLogger("AlignAIR.Pipeline")

    parser = build_cli_parser()
    args = parser.parse_args()

    config = config_from_cli(args)

    # Lock determinism
    det_settings = set_deterministic(
        seed=config.reproducibility.seed,
        tf_deterministic=config.reproducibility.deterministic_ops,
    )

    # Capture environment
    env = EnvironmentFingerprint.capture()
    log.info("Environment fingerprint: %s", env.fingerprint)

    # Assemble and run
    pipeline_logger = PipelineLogger()
    stages = assemble_pipeline(config)
    pipeline = AlignAIRPipeline(config, stages, logger=pipeline_logger)

    started = datetime.datetime.now(datetime.timezone.utc).isoformat()

    try:
        store = pipeline.run()
        output_path = store.get("output_path")
        run_report = store.get("run_report")

        log.info("Pipeline complete. Output: %s", output_path)

        # Build provenance if requested
        if getattr(args, 'save_provenance', False) and output_path:
            finished = datetime.datetime.now(datetime.timezone.utc).isoformat()

            provenance = RunProvenance(
                run_id=run_report.get("run_id", ""),
                started_utc=started,
                finished_utc=finished,
                wall_seconds=run_report.get("total_wall_seconds", 0.0),
                environment=env.to_dict(),
                input_path=config.sequences_path,
                input_sha256=file_sha256(config.sequences_path),
                model_dir=config.model_dir,
                model_fingerprint=store.get("model").bundle_fingerprint if store.has("model") else "",
                config_sha256=config.sha256(),
                determinism_settings=det_settings,
                stages_executed=pipeline_logger.stages_executed,
                stage_timings=pipeline_logger.stage_timings,
                output_path=output_path,
                output_sha256=file_sha256(output_path),
            )
            provenance.compute_reproducibility_hash()

            prov_path = Path(output_path).with_suffix(".provenance.json")
            provenance.save(prov_path)
            log.info("Provenance saved: %s", prov_path)

        return 0
    except Exception as e:
        log.error("Pipeline failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
