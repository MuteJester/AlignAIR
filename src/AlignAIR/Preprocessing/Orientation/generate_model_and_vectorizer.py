#! /usr/bin/env python
"""
Custom DNA-orientation detector – compact edition
-------------------------------------------------
Generates synthetic AIRR-seq datasets, trains a char-ngram logistic-regression
classifier to recognise sequence orientation, and saves the fitted sklearn
pipeline plus evaluation plots.

Thomas Konstantinovsky · 2025-05-15
"""
# ────────────────────────── std lib ──────────────────────────
import argparse
import logging
import pickle
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Optional

# ────────────────────────── 3rd-party ──────────────────────────
import matplotlib.pyplot as plt
import numpy as np
from GenAIRR.dataconfig import DataConfig
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm
from GenAIRR.mutation import S5F
try:  # type: ignore
    from GenAIRR.mutation.uniform import Uniform  # type: ignore
except Exception:  # pragma: no cover - fallback for older versions
    from GenAIRR.mutation import Uniform  # type: ignore
from GenAIRR.pipeline import (
    AugmentationPipeline,
)
from GenAIRR.steps import (
    SimulateSequence,
    FixVPositionAfterTrimmingIndexAmbiguity,
    FixDPositionAfterTrimmingIndexAmbiguity,
    FixJPositionAfterTrimmingIndexAmbiguity,
    CorrectForVEndCut,
    CorrectForDTrims,
    CorruptSequenceBeginning,
    InsertNs,
    InsertIndels,
    ShortDValidation,
    DistillMutationRate
)
from GenAIRR.steps import AugmentationStep
from AlignAIR.Preprocessing.Orientation import (
    reverse_sequence,
    complement_sequence,
    reverse_complement_sequence,
)
from enum import Enum, auto
# ────────────────────────── logging ──────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)




class SliceType:

    types = ['Only V', 'Only J', 'VJ']

    def __init__(self, dataconfig: DataConfig):
        self.dataconfig = dataconfig


        if self.dataconfig.metadata.has_d:
            self.types += ['Only D', 'VD', 'DJ']

    def __iter__(self):
        for item in self.types:
            yield item

    def sample(self):
        return random.choice(self.types)



ORIENTATIONS = {
    "Normal": lambda s: s,
    "Reversed": reverse_sequence,
    "Complement": complement_sequence,
    "Reverse Complement": reverse_complement_sequence,
}

DEFAULT_TRAIN_SIZE = 300_000



# ────────────────────────── helpers ──────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("AlignAIR orientation detector")
    p.add_argument("--save_path", required=True, help="Folder for pipeline pickle")
    p.add_argument("--plot_save_path", required=True, help="Folder for confusion plots")
    p.add_argument("--dataconfig", help="built in GenAIRR DataConfig or Path to pkl file", required=True)
    p.add_argument("--train_dataset_size", type=int, default=DEFAULT_TRAIN_SIZE)
    p.add_argument("--uniform_test_dataset_size", type=int, default=DEFAULT_TRAIN_SIZE)
    p.add_argument("--partial_test_dataset_size", type=int, default=DEFAULT_TRAIN_SIZE)

    # mutation model and min max
    p.add_argument("--min_mutation_rate", type=float, default=0.003)
    p.add_argument("--max_mutation_rate", type=float, default=0.25)
    p.add_argument('--mutation_model', choices=['S5F', 'Uniform'], default='S5F', help="Mutation model to use for training data generation")
    return p.parse_args()


def build_augmentation_pipeline(mutation_model, has_d: bool) -> AugmentationPipeline:
    """Create an AugmentationPipeline with the standard post-processing steps."""
    steps = [
        # Newer API supports passing the mutation model positionally
        SimulateSequence(mutation_model, productive=True),
        FixVPositionAfterTrimmingIndexAmbiguity(),
        FixJPositionAfterTrimmingIndexAmbiguity(),
        CorrectForVEndCut(),
        # Align to current signature used across the repo (see GenerateStrictlyBalancedTrainDataset)
        CorruptSequenceBeginning(0.7, [0.4, 0.4, 0.2], 576, 210, 310, 50),
        InsertNs(0.02, 0.5),
        InsertIndels(0.5, 5, 0.5, 0.5),
        DistillMutationRate(),
    ]
    if has_d:
        # Insert D-specific fixes in appropriate places
        steps.insert(2, FixDPositionAfterTrimmingIndexAmbiguity())  # after V fix
        steps.insert(4, CorrectForDTrims())  # after V-end correction
        steps.insert(6, ShortDValidation(5))  # before indels
    return AugmentationPipeline(steps)


def _to_record(obj: Any) -> Dict[str, Any]:
    """Normalize pipeline output to a dict for downstream use."""
    if hasattr(obj, 'get_dict'):
        rec = obj.get_dict()  # type: ignore[attr-defined]
    else:
        rec = obj
    if isinstance(rec, list) and len(rec) == 1 and isinstance(rec[0], dict):
        rec = rec[0]
    if not isinstance(rec, dict):
        raise RuntimeError("Simulation output is not a dict-like record")
    return rec


def make_sequences(
        n: int,
        dataconfig: DataConfig,
        mutation_model,
        return_partials: bool = False
                ) -> Tuple[List[str], List[str], Optional[List[str]]]:
    """
    Generate `n` sequences plus orientation labels.
    If `return_partials` is True, also return the partial-label list.
    """
    # Updated API: chain_type is derived from dataconfig metadata; no explicit constant required
    AugmentationStep.set_dataconfig(dataconfig)

    has_d = bool(getattr(getattr(dataconfig, 'metadata', None), 'has_d', False))
    pipe = build_augmentation_pipeline(mutation_model, has_d)
    seqs, orient_labels, partial_labels = [], [], []
    slice_enum = SliceType(dataconfig)

    for _ in tqdm(range(n), desc="Simulating sequences"):
        sample_raw = pipe.execute()
        sample = _to_record(sample_raw)
        # Optionally cut to partial region
        if return_partials:
            sample["sequence"], partial = _extract_partial(sample, slice_enum)
            partial_labels.append(partial)

        # Apply random orientation
        orient, fn = random.choice(list(ORIENTATIONS.items()))
        seqs.append(fn(sample["sequence"]))
        orient_labels.append(orient)

    return seqs, orient_labels, partial_labels if return_partials else None


def _extract_partial(
    sim: Dict[str, Any], slice_types: SliceType
) -> Tuple[str, str]:  # heavy-/light-aware
    """Return a subsequence plus its slice label."""
    choice = slice_types.sample()
    def _mk_slice(start_key: str, end_key: str) -> Optional[slice]:
        if start_key in sim and end_key in sim and sim[start_key] is not None and sim[end_key] is not None:
            return slice(sim[start_key], sim[end_key])
        return None

    v = _mk_slice("v_sequence_start", "v_sequence_end")
    d = _mk_slice("d_sequence_start", "d_sequence_end")
    j = _mk_slice("j_sequence_start", "j_sequence_end")

    seq = sim["sequence"]
    joined: Dict[str, str] = {}
    if v is not None:
        joined['Only V'] = seq[v]
    if d is not None:
        joined["Only D"] = seq[d]
    if j is not None:
        joined["Only J"] = seq[j]
    if v is not None and d is not None:
        joined["VD"] = seq[v] + seq[d]
    if v is not None and j is not None:
        joined["VJ"] = seq[v] + seq[j]
    if d is not None and j is not None:
        joined["DJ"] = seq[d] + seq[j]

    if choice not in joined:
        # Fallback: if chosen slice unavailable (e.g., light chain picked D category), prefer VJ if possible
        for alt in ("VJ", "Only V", "Only J"):
            if alt in joined:
                return joined[alt], alt
        # As a last resort, return full sequence
        return seq, "Full"
    return joined[choice], choice


def build_sklearn_pipeline(
    min_k: int = 3, max_k: int = 5, max_features: int = 576
) -> Pipeline:
    vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=(min_k, max_k),
        max_features=max_features,
        binary=True,
    )
    model = LogisticRegression(random_state=42, max_iter=1_000)
    return Pipeline([("vectorizer", vectorizer), ("model", model)])


def eval_and_plot(
    pipe: Pipeline,
    X: List[str],
    y: List[str],
    title: str,
    out_path: Path,
) -> None:
    preds = pipe.predict(X)
    labels = list(ORIENTATIONS.keys())
    cm = confusion_matrix(y, preds, labels=labels, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]*100:.1f}%", ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Row-normalized')
    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logging.info("Saved confusion matrix → %s", out_path)


def dump_pipeline(pipe: Pipeline, dest: Path) -> None:
    with dest.open("wb") as f:
        pickle.dump(pipe, f)
    logging.info("Saved sklearn pipeline → %s", dest)


# ────────────────────────── main ──────────────────────────
def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_path).expanduser()
    plot_dir = Path(args.plot_save_path).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ─── datasets ───
    if args.mutation_model == 'Uniform':
        mutation_model = Uniform(args.min_mutation_rate, args.max_mutation_rate)
    else:
        mutation_model = S5F(args.min_mutation_rate, args.max_mutation_rate)


    from GenAIRR.data import _CONFIG_NAMES
    dataconfig_path = args.dataconfig
    if dataconfig_path in _CONFIG_NAMES:
        import importlib
        dataconfig = getattr(importlib.import_module("GenAIRR.data"), dataconfig_path)()
    elif Path(dataconfig_path).is_file():
        with open(dataconfig_path, "rb") as h:
            dataconfig = pickle.load(h)
        if not isinstance(dataconfig, DataConfig):
            raise ValueError("Provided dataconfig file does not contain a DataConfig object")


    train_X, train_y, _ = make_sequences(
        args.train_dataset_size, dataconfig, mutation_model
    )



    # Add Uniform half
    uni_X, uni_y, _ = make_sequences(
        args.train_dataset_size, dataconfig, Uniform(0.003, 0.25)  # type: ignore[arg-type]
    )
    train_X.extend(uni_X)
    train_y.extend(uni_y)

    test_X, test_y, _ = make_sequences(
        args.uniform_test_dataset_size, dataconfig, Uniform(0.003, 0.25)  # type: ignore[arg-type]
    )

    ptest_X, ptest_y, _ = make_sequences(
        args.partial_test_dataset_size,
        dataconfig,
        Uniform(0.003, 0.25),  # type: ignore[arg-type]
        return_partials=True,
    )

    # ─── model ───
    clf = build_sklearn_pipeline()
    logging.info("Training classifier …")
    clf.fit(train_X, train_y)
    logging.info("Training done.")

    # ─── evaluation ───
    eval_and_plot(
        clf,
        test_X,
        test_y,
        "Uniform Mutation Model – full sequences",
        plot_dir / "uniform_mm_confmat.png",
    )
    eval_and_plot(
        clf,
        ptest_X,
        ptest_y,
        "Uniform Mutation Model – partial sequences",
        plot_dir / "partial_uniform_mm_confmat.png",
    )

    # ─── persist ───
    dump_pipeline(clf, save_dir / "Custom_DNA_Orientation_Pipeline.pkl")


if __name__ == "__main__":
    main()
