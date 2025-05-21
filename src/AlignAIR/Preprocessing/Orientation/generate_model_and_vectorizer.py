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
from typing import Callable, Dict, List, Tuple

# ────────────────────────── 3rd-party ──────────────────────────
import matplotlib.pyplot as plt
import numpy as np
from scikitplot.metrics import plot_confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

# ────────────────────────── GenAIRR / AlignAIR ──────────────────────────
from GenAIRR.data import (
    builtin_heavy_chain_data_config,
    builtin_kappa_chain_data_config,
    builtin_lambda_chain_data_config,
    builtin_tcrb_data_config,
)
from GenAIRR.mutation import S5F, Uniform
from GenAIRR.pipeline import (
    AugmentationPipeline,
    CHAIN_TYPE_BCR_HEAVY,
    CHAIN_TYPE_BCR_LIGHT_KAPPA,
    CHAIN_TYPE_BCR_LIGHT_LAMBDA,
    CHAIN_TYPE_TCR_BETA,
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

# ────────────────────────── logging ──────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ────────────────────────── constants ──────────────────────────
CHAIN_SPEC = {
    "heavy": (builtin_heavy_chain_data_config(), CHAIN_TYPE_BCR_HEAVY),
    "tcrb": (builtin_tcrb_data_config(), CHAIN_TYPE_TCR_BETA),
    "kappa": (builtin_kappa_chain_data_config(), CHAIN_TYPE_BCR_LIGHT_KAPPA),
    "lambda": (builtin_lambda_chain_data_config(), CHAIN_TYPE_BCR_LIGHT_LAMBDA),
}

ORIENTATIONS = {
    "Normal": lambda s: s,
    "Reversed": reverse_sequence,
    "Complement": complement_sequence,
    "Reverse Complement": reverse_complement_sequence,
}

PARTIAL_SLICES_HEAVY = ["Only V", "Only D", "Only J", "VD", "VJ", "DJ"]
PARTIAL_SLICES_LIGHT = ["Only V", "Only J", "VJ"]

# ────────────────────────── helpers ──────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("AlignAIR orientation detector")
    p.add_argument("--save_path", required=True, help="Folder for pipeline pickle")
    p.add_argument("--plot_save_path", required=True, help="Folder for confusion plots")
    p.add_argument("--chain_type", required=True, choices=CHAIN_SPEC, help="heavy / …")
    p.add_argument("--train_dataset_size", type=int, default=100_000)
    p.add_argument("--uniform_test_dataset_size", type=int, default=100_000)
    p.add_argument("--partial_test_dataset_size", type=int, default=100_000)
    return p.parse_args()


def build_augmentation_pipeline(mutation_model) -> AugmentationPipeline:
    """Create an AugmentationPipeline with the standard post-processing steps."""
    steps = [
        SimulateSequence(mutation_model=mutation_model, productive=True),
        FixVPositionAfterTrimmingIndexAmbiguity(),
        FixDPositionAfterTrimmingIndexAmbiguity(),
        FixJPositionAfterTrimmingIndexAmbiguity(),
        CorrectForVEndCut(),
        CorrectForDTrims(),
        CorruptSequenceBeginning(
            corruption_probability=0.7,
            corrupt_events_proba=[0.4, 0.4, 0.2],
            max_sequence_length=576,
            nucleotide_add_coefficient=210,
            nucleotide_remove_coefficient=310,
            nucleotide_add_after_remove_coefficient=50,
            random_sequence_add_proba=1,
            single_base_stream_proba=0,
            duplicate_leading_proba=0,
            random_allele_proba=0,
        ),
        InsertNs(n_ratio=0.02, proba=0.5),
        ShortDValidation(short_d_length=5),
        InsertIndels(indel_probability=0.5, max_indels=5,insertion_proba=0.5, deletion_proba=0.5),
        DistillMutationRate(),
    ]
    return AugmentationPipeline(steps)


def make_sequences(
    n: int, chain_type: str, mutation_model, return_partials: bool = False
):
    """
    Generate `n` sequences plus orientation labels.
    If `return_partials` is True, also return the partial-label list.
    """
    dataconfig, chain_const = CHAIN_SPEC[chain_type]
    AugmentationStep.set_dataconfig(dataconfig, chain_type=chain_const)

    pipe = build_augmentation_pipeline(mutation_model)
    seqs, orient_labels, partial_labels = [], [], []

    slice_choices = (
        PARTIAL_SLICES_HEAVY
        if chain_type in {"heavy", "tcrb"}
        else PARTIAL_SLICES_LIGHT
    )

    for _ in tqdm(range(n), desc="Simulating sequences"):
        sample = pipe.execute()
        # Optionally cut to partial region
        if return_partials:
            sample["sequence"], partial = _extract_partial(sample, slice_choices)
            partial_labels.append(partial)

        # Apply random orientation
        orient, fn = random.choice(list(ORIENTATIONS.items()))
        seqs.append(fn(sample["sequence"]))
        orient_labels.append(orient)

    return seqs, orient_labels, partial_labels if return_partials else None


def _extract_partial(
    sim: Dict, choices: List[str]
) -> Tuple[str, str]:  # heavy-/light-aware
    """Return a subsequence plus its slice label."""
    choice = random.choice(choices)
    v = slice(sim["v_sequence_start"], sim["v_sequence_end"])
    d = slice(sim["d_sequence_start"], sim["d_sequence_end"])
    j = slice(sim["j_sequence_start"], sim["j_sequence_end"])
    joined = {
        "Only V": sim["sequence"][v],
        "Only D": sim["sequence"][d],
        "Only J": sim["sequence"][j],
        "VD": sim["sequence"][v] + sim["sequence"][d],
        "VJ": sim["sequence"][v] + sim["sequence"][j],
        "DJ": sim["sequence"][d] + sim["sequence"][j],
    }
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
    plot_confusion_matrix(y, preds, normalize=True, x_tick_rotation=90)
    plt.title(title)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
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
    mutation_model = S5F(0.003, 0.25) if args.chain_type in {"heavy", "light"} else Uniform(0.003, 0.01)
    train_X, train_y, _ = make_sequences(
        args.train_dataset_size, args.chain_type,mutation_model
    )
    # Add Uniform half
    uni_X, uni_y, _ = make_sequences(
        args.train_dataset_size, args.chain_type, Uniform(0.003, 0.25)
    )
    train_X.extend(uni_X)
    train_y.extend(uni_y)

    test_X, test_y, _ = make_sequences(
        args.uniform_test_dataset_size, args.chain_type, Uniform(0.003, 0.25)
    )

    ptest_X, ptest_y, _ = make_sequences(
        args.partial_test_dataset_size,
        args.chain_type,
        Uniform(0.003, 0.25),
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
