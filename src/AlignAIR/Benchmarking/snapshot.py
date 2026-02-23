"""
Create and load reproducibility snapshots for AlignAIR models.

A snapshot captures a model's complete behavior on a fixed evaluation dataset:
- Raw prediction outputs (allele probabilities, positions, scalars)
- Latent representations (V, D, J gene internal representations)
- Quality metrics computed against ground truth
- Full pipeline CSV output (optional)
"""

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelSnapshot:
    """Creates and loads reproducibility snapshots."""

    @staticmethod
    def create(
        model_dir: str,
        eval_data_path: str,
        output_dir: str,
        batch_size: int = 64,
        max_sequences: Optional[int] = None,
        include_latent: bool = True,
        include_pipeline: bool = True,
        ground_truth_columns: Optional[dict] = None,
    ) -> Path:
        """
        Create a full snapshot of a model's behavior on evaluation data.

        Args:
            model_dir: Path to the model bundle directory.
            eval_data_path: Path to evaluation CSV. Must have a 'sequence' column.
                For metrics, must also have ground truth columns (v_call, j_call, etc.).
            output_dir: Where to save the snapshot.
            batch_size: Batch size for inference.
            max_sequences: Limit number of sequences (None = use all).
            include_latent: Whether to extract and save latent representations.
            include_pipeline: Whether to run the full pipeline and save CSV output.
            ground_truth_columns: Mapping of ground truth column names. Defaults to
                standard names: v_call, d_call, j_call, v_sequence_start, etc.

        Returns:
            Path to the created snapshot directory.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_path = Path(model_dir)
        config = _load_bundle_config(model_path)

        # Load evaluation data
        eval_df = pd.read_csv(eval_data_path)
        if max_sequences is not None:
            eval_df = eval_df.head(max_sequences)
        sequences = eval_df['sequence'].tolist()
        logger.info("Loaded %d sequences from %s", len(sequences), eval_data_path)

        # Tokenize sequences
        tokenized = _tokenize_sequences(sequences, config.get('max_seq_length', 576))

        # --- 1. Raw predictions ---
        logger.info("Running raw model predictions...")
        predictions = _run_predictions(model_path, tokenized, batch_size)
        pred_dir = output_path / "predictions"
        pred_dir.mkdir(exist_ok=True)
        np.savez_compressed(pred_dir / "raw_outputs.npz", **predictions)
        logger.info("Saved raw outputs with keys: %s", list(predictions.keys()))

        # --- 2. Latent representations ---
        if include_latent:
            logger.info("Extracting latent representations...")
            latent_dir = output_path / "latent"
            latent_dir.mkdir(exist_ok=True)
            latent_reps = _extract_latent_representations(
                model_path, config, tokenized, batch_size
            )
            for gene, latent in latent_reps.items():
                np.save(latent_dir / f"{gene}_latent.npy", latent)
                logger.info("Saved %s latent: shape %s", gene, latent.shape)

        # --- 3. Pipeline output ---
        if include_pipeline:
            logger.info("Running full pipeline...")
            pipeline_csv = _run_pipeline(model_path, eval_data_path, output_path, batch_size)
            if pipeline_csv:
                logger.info("Pipeline output saved to %s", pipeline_csv)
            else:
                logger.warning("Pipeline execution failed or produced no output.")

        # --- 4. Metrics ---
        metrics_dir = output_path / "metrics"
        metrics_dir.mkdir(exist_ok=True)

        # Copy training meta from bundle if available
        training_meta_src = model_path / "training_meta.json"
        if training_meta_src.exists():
            import shutil
            shutil.copy2(training_meta_src, metrics_dir / "training_meta.json")

        # Compute quality metrics if ground truth is available
        gt_cols = ground_truth_columns or _default_ground_truth_columns()
        metrics = _compute_metrics(predictions, eval_df, config, gt_cols)
        with open(metrics_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=_json_serialize)
        logger.info("Computed %d metric categories", len(metrics))

        # --- 5. Metadata ---
        metadata = _build_metadata(model_path, eval_data_path, config, len(sequences))
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # --- 6. Snapshot config ---
        snapshot_config = {
            "model_dir": str(model_path),
            "eval_data_path": str(eval_data_path),
            "num_sequences": len(sequences),
            "max_sequences": max_sequences,
            "batch_size": batch_size,
            "include_latent": include_latent,
            "include_pipeline": include_pipeline,
        }
        with open(output_path / "config.json", "w") as f:
            json.dump(snapshot_config, f, indent=2)

        logger.info("Snapshot created at %s", output_path)
        return output_path

    @staticmethod
    def load(snapshot_dir: str) -> dict:
        """
        Load all snapshot artifacts into a dict.

        Returns:
            dict with keys: 'metadata', 'config', 'predictions', 'latent', 'metrics',
                            'pipeline_output', 'training_meta'
        """
        path = Path(snapshot_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Snapshot directory not found: {path}")

        result = {}

        # Metadata
        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                result['metadata'] = json.load(f)

        # Config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                result['config'] = json.load(f)

        # Predictions
        pred_path = path / "predictions" / "raw_outputs.npz"
        if pred_path.exists():
            npz = np.load(pred_path)
            result['predictions'] = {k: npz[k] for k in npz.files}

        # Latent representations
        latent_dir = path / "latent"
        if latent_dir.is_dir():
            latent = {}
            for f in latent_dir.glob("*_latent.npy"):
                gene = f.stem.replace("_latent", "")
                latent[gene] = np.load(f)
            result['latent'] = latent

        # Metrics
        metrics_path = path / "metrics" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                result['metrics'] = json.load(f)

        # Training meta
        tm_path = path / "metrics" / "training_meta.json"
        if tm_path.exists():
            with open(tm_path) as f:
                result['training_meta'] = json.load(f)

        # Pipeline output
        pipeline_dir = path / "predictions"
        pipeline_csv = pipeline_dir / "pipeline_output.csv"
        if pipeline_csv.exists():
            result['pipeline_output'] = pd.read_csv(pipeline_csv)

        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_bundle_config(model_path: Path) -> dict:
    """Load config.json from a model bundle."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {model_path}")
    with open(config_path) as f:
        return json.load(f)


def _tokenize_sequences(sequences: list, max_seq_length: int) -> np.ndarray:
    """Tokenize and center-pad sequences."""
    from AlignAIR.Data.tokenizers.center_padded_sequence_tokenizer import CenterPaddedSequenceTokenizer
    tokenizer = CenterPaddedSequenceTokenizer(max_length=max_seq_length)
    tokenized, _ = tokenizer.encode_and_pad_center(sequences)
    return tokenized


def _run_predictions(model_path: Path, tokenized: np.ndarray, batch_size: int) -> dict:
    """Run model inference and return all outputs as numpy arrays."""
    from AlignAIR.Models.SingleChainAlignAIR.SingleChainAlignAIR import SingleChainAlignAIR

    wrapper = SingleChainAlignAIR.from_pretrained(str(model_path))
    n = len(tokenized)
    all_outputs = {}

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = tokenized[start:end]
        outputs = wrapper.predict({"tokenized_sequence": batch}, verbose=0)
        for key, val in outputs.items():
            if key not in all_outputs:
                all_outputs[key] = []
            all_outputs[key].append(val)

    # Concatenate batches
    return {k: np.concatenate(v, axis=0) for k, v in all_outputs.items()}


def _extract_latent_representations(
    model_path: Path, config: dict, tokenized: np.ndarray, batch_size: int
) -> dict:
    """
    Extract latent representations by reconstructing the full Keras model.
    Requires checkpoint.weights.h5 in the bundle.
    """
    import pickle
    import tensorflow as tf

    weights_path = model_path / "checkpoint.weights.h5"
    if not weights_path.exists():
        logger.warning(
            "No checkpoint.weights.h5 found in %s. "
            "Latent extraction requires the full Keras weights. Skipping.",
            model_path
        )
        return {}

    # Load dataconfig
    dc_path = model_path / "dataconfig.pkl"
    with open(dc_path, "rb") as f:
        dataconfig = pickle.load(f)

    # Reconstruct the full model
    from AlignAIR.Models.SingleChainAlignAIR.SingleChainAlignAIR import SingleChainAlignAIR

    model = SingleChainAlignAIR(
        max_seq_length=config['max_seq_length'],
        dataconfig=dataconfig,
        v_allele_latent_size=config.get('v_allele_latent_size'),
        d_allele_latent_size=config.get('d_allele_latent_size'),
        j_allele_latent_size=config.get('j_allele_latent_size'),
    )

    # Build model with a dummy forward pass
    dummy_input = {"tokenized_sequence": tf.zeros((1, config['max_seq_length']), dtype=tf.int32)}
    _ = model(dummy_input, training=False)

    # Load weights
    model.load_weights(str(weights_path))
    logger.info("Loaded Keras weights from %s", weights_path)

    # Extract latent representations
    genes = ['V', 'J']
    if config.get('has_d_gene', False):
        genes.append('D')

    latent_reps = {}
    n = len(tokenized)
    for gene in genes:
        gene_latents = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = {"tokenized_sequence": tf.constant(tokenized[start:end], dtype=tf.int32)}
            latent = model.get_latent_representation(batch, gene)
            gene_latents.append(latent.numpy())
        latent_reps[gene.lower()] = np.concatenate(gene_latents, axis=0)

    return latent_reps


def _run_pipeline(model_path: Path, eval_data_path: str, output_path: Path, batch_size: int) -> Optional[str]:
    """Run the full AlignAIR prediction pipeline via CLI and save the output."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, "-m", "AlignAIR.API.AlignAIRRPredict",
            "--model_dir", str(model_path),
            "--save_path", tmpdir + "/",
            "--sequences", str(eval_data_path),
            "--batch_size", str(batch_size),
            "--translate_to_asc",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            logger.error("Pipeline failed:\nSTDOUT: %s\nSTDERR: %s", result.stdout, result.stderr)
            return None

        # Find the output CSV
        output_csvs = list(Path(tmpdir).glob("*_alignairr_results.csv"))
        if not output_csvs:
            logger.error("No pipeline output CSV found")
            return None

        # Copy to snapshot
        import shutil
        dest = output_path / "predictions" / "pipeline_output.csv"
        shutil.copy2(output_csvs[0], dest)
        return str(dest)


def _default_ground_truth_columns() -> dict:
    """Standard ground truth column name mapping."""
    return {
        "v_call": "v_call",
        "d_call": "d_call",
        "j_call": "j_call",
        "v_sequence_start": "v_sequence_start",
        "v_sequence_end": "v_sequence_end",
        "d_sequence_start": "d_sequence_start",
        "d_sequence_end": "d_sequence_end",
        "j_sequence_start": "j_sequence_start",
        "j_sequence_end": "j_sequence_end",
        "mutation_rate": "mutation_rate",
        "productive": "productive",
    }


def _compute_metrics(
    predictions: dict,
    eval_df: pd.DataFrame,
    config: dict,
    gt_cols: dict,
) -> dict:
    """Compute quality metrics from predictions vs ground truth."""
    metrics = {}

    # Allele classification metrics
    for gene in ['v', 'j', 'd']:
        allele_key = f"{gene}_allele"
        gt_col = gt_cols.get(f"{gene}_call")

        if allele_key not in predictions or gt_col not in eval_df.columns:
            continue

        probs = predictions[allele_key]  # (N, num_alleles)

        # Top-1 accuracy: does the argmax match any of the ground truth alleles?
        # For now, store prediction statistics (actual accuracy needs allele name mapping)
        metrics[f"{gene}_allele"] = {
            "mean_max_probability": float(np.mean(np.max(probs, axis=1))),
            "std_max_probability": float(np.std(np.max(probs, axis=1))),
            "mean_entropy": float(np.mean(-np.sum(
                np.where(probs > 1e-10, probs * np.log(probs + 1e-10), 0), axis=1
            ))),
            "num_alleles": int(probs.shape[1]),
        }

    # Position metrics
    for gene in ['v', 'j', 'd']:
        for pos in ['start', 'end']:
            pred_key = f"{gene}_{pos}"
            gt_col_name = gt_cols.get(f"{gene}_sequence_{pos}")

            if pred_key not in predictions or gt_col_name not in eval_df.columns:
                continue

            pred_pos = predictions[pred_key].flatten()
            gt_pos = eval_df[gt_col_name].values.astype(float)

            # Filter valid entries (non-NaN)
            valid = ~np.isnan(gt_pos)
            if valid.sum() == 0:
                continue

            pred_valid = pred_pos[valid]
            gt_valid = gt_pos[valid]

            mae = float(np.mean(np.abs(pred_valid - gt_valid)))
            exact_match = float(np.mean(np.abs(pred_valid - gt_valid) < 0.5))
            within_1nt = float(np.mean(np.abs(pred_valid - gt_valid) < 1.5))

            metrics[f"{gene}_{pos}"] = {
                "mae": mae,
                "exact_match_rate": exact_match,
                "within_1nt_rate": within_1nt,
            }

    # Scalar metrics
    for key, gt_col_name in [("mutation_rate", gt_cols.get("mutation_rate")),
                              ("productive", gt_cols.get("productive"))]:
        if key not in predictions:
            continue
        pred_vals = predictions[key].flatten()

        if gt_col_name and gt_col_name in eval_df.columns:
            gt_vals = eval_df[gt_col_name].values.astype(float)
            valid = ~np.isnan(gt_vals)
            if valid.sum() > 0:
                metrics[key] = {
                    "mae": float(np.mean(np.abs(pred_vals[valid] - gt_vals[valid]))),
                    "pred_mean": float(np.mean(pred_vals)),
                    "pred_std": float(np.std(pred_vals)),
                }
                continue

        # No ground truth — just record prediction stats
        metrics[key] = {
            "pred_mean": float(np.mean(pred_vals)),
            "pred_std": float(np.std(pred_vals)),
        }

    return metrics


def _build_metadata(model_path: Path, eval_data_path: str, config: dict, num_sequences: int) -> dict:
    """Build metadata dict for the snapshot."""
    # Try to get git commit
    git_commit = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()
    except Exception:
        pass

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_path),
        "model_config": config,
        "eval_data_path": str(eval_data_path),
        "num_sequences": num_sequences,
        "git_commit": git_commit,
        "python_version": sys.version,
    }


def _json_serialize(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
