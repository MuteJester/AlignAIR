"""LoadModelStage — merges ConfigLoad + FileNameExtraction + FileSampleCounter + ModelLoading.

This is the atomic "setup" stage. Everything needed before inference
is loaded here, validated, and returned as frozen dataclasses.
"""
from __future__ import annotations

import logging
import os
import pathlib
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

from AlignAIR.Pipeline.Stage.protocol import Stage, StageContext
from AlignAIR.Pipeline.Models.slots import FileInfo, LoadedModel
from AlignAIR.Pipeline.Errors.exceptions import ModelLoadError, DataConfigError

logger = logging.getLogger("AlignAIR.Pipeline")


class LoadModelStage(Stage):
    """Loads the model bundle, dataconfig, sequences, and file metadata."""

    reads = frozenset({"config"})
    writes = frozenset({"model", "file_info", "sequences"})

    def run(self, context: StageContext) -> Dict[str, Any]:
        config = context.config

        # 1. Load bundle config
        bundle_path = pathlib.Path(config.model_dir)
        if not bundle_path.is_dir() or not (bundle_path / "config.json").exists():
            raise ModelLoadError(f"Model bundle not found or invalid at: {config.model_dir}")

        from AlignAIR.Serialization.io import load_bundle
        try:
            bundle_config, dataconfig, _meta = load_bundle(bundle_path)
        except Exception as e:
            raise ModelLoadError(f"Failed to load bundle from {config.model_dir}: {e}") from e

        model_type = getattr(bundle_config, "model_type", "single_chain")

        # 2. Load SavedModel inference wrapper
        try:
            if model_type == "multi_chain":
                from AlignAIR.Models.MultiChainAlignAIR.MultiChainAlignAIR import MultiChainAlignAIR
                wrapper = MultiChainAlignAIR.from_pretrained(config.model_dir)
            else:
                from AlignAIR.Models.SingleChainAlignAIR.SingleChainAlignAIR import SingleChainAlignAIR
                wrapper = SingleChainAlignAIR.from_pretrained(config.model_dir)
        except Exception as e:
            raise ModelLoadError(f"Failed to load model from {config.model_dir}: {e}") from e

        logger.info("Loaded SavedModel from %s", config.model_dir)

        # 3. Wrap dataconfig if needed
        from AlignAIR.Data import MultiDataConfigContainer
        if not isinstance(dataconfig, MultiDataConfigContainer):
            dataconfig = MultiDataConfigContainer([dataconfig])

        # Update from model's own dataconfig if available
        model_dc = getattr(wrapper, 'dataconfigs', None) or getattr(wrapper, 'dataconfig', None)
        if model_dc is not None:
            if not isinstance(model_dc, MultiDataConfigContainer):
                model_dc = MultiDataConfigContainer([model_dc])
            dataconfig = model_dc

        # 4. Load orientation pipeline
        orientation = None
        if config.orientation.enabled:
            orientation = self._load_orientation(config, dataconfig)
            logger.info("Orientation pipeline loaded")

        # 5. Fit kmer density extractor
        candidate_extractor = self._fit_kmer_extractor(
            dataconfig, getattr(bundle_config, 'max_seq_length', 576)
        )

        # 6. Read sequences and build file info
        sequences, file_info = self._load_sequences(config)
        logger.info("Loaded %d sequences from %s", len(sequences), config.sequences_path)

        # 7. Orientation is handled by the consumer_producer worker during
        #    tokenization — do NOT apply it here to avoid double-application.

        # 8. Determine chain type
        has_d = getattr(bundle_config, 'has_d_gene', True)
        chain_type = self._infer_chain_type(bundle_config, dataconfig)

        # 9. Read fingerprint
        fingerprint = self._read_fingerprint(config.model_dir)

        model = LoadedModel(
            inference_wrapper=wrapper,
            dataconfig=dataconfig,
            max_seq_length=getattr(bundle_config, 'max_seq_length', 576),
            has_d_gene=has_d,
            v_allele_count=getattr(bundle_config, 'v_allele_count', 0),
            j_allele_count=getattr(bundle_config, 'j_allele_count', 0),
            d_allele_count=getattr(bundle_config, 'd_allele_count', None),
            orientation_pipeline=orientation,
            bundle_fingerprint=fingerprint,
            chain_type=chain_type,
            candidate_extractor=candidate_extractor,
        )

        return {"model": model, "file_info": file_info, "sequences": sequences}

    def _load_orientation(self, config, dataconfig):
        """Load orientation pipeline — custom or builtin."""
        if config.orientation.custom_model_path:
            with open(config.orientation.custom_model_path, 'rb') as h:
                return pickle.load(h)

        from AlignAIR.PretrainedComponents import builtin_orientation_classifier
        from GenAIRR.dataconfig.enums import ChainType as CT

        try:
            chain_types = dataconfig.chain_types()
            chain_type_val = chain_types[0] if chain_types else None
        except Exception:
            chain_type_val = None

        if isinstance(chain_type_val, CT):
            return builtin_orientation_classifier(chain_type_val)

        s = str(chain_type_val).lower() if chain_type_val else ""
        if 'kappa' in s or s in {'igk', 'bcr_light_kappa'}:
            return builtin_orientation_classifier(CT.BCR_LIGHT_KAPPA)
        elif 'lambda' in s or s in {'igl', 'bcr_light_lambda'}:
            return builtin_orientation_classifier(CT.BCR_LIGHT_LAMBDA)
        elif 'tcrb' in s or 'beta' in s:
            return builtin_orientation_classifier(CT.TCR_BETA)
        else:
            return builtin_orientation_classifier(CT.BCR_HEAVY)

    def _fit_kmer_extractor(self, dataconfig, max_seq_length):
        """Fit a FastKmerDensityExtractor on reference alleles."""
        from AlignAIR.Preprocessing.LongSequence.FastKmerDensityExtractor import FastKmerDensityExtractor

        ref_alleles = []
        for dc in dataconfig:
            for gene in ['v', 'j']:
                for a in dc.allele_list(gene):
                    seq = getattr(a, 'ungapped_seq', '')
                    if isinstance(seq, str) and seq:
                        ref_alleles.append(seq.upper())
            if dc.metadata.has_d:
                for a in dc.allele_list('d'):
                    seq = getattr(a, 'ungapped_seq', '')
                    if isinstance(seq, str) and seq:
                        ref_alleles.append(seq.upper())

        ref_alleles = list(dict.fromkeys(ref_alleles))  # deduplicate

        extractor = FastKmerDensityExtractor(11, max_length=max_seq_length, allowed_mismatches=0)
        extractor.fit(ref_alleles)
        return extractor

    def _load_sequences(self, config):
        """Load sequences from file and return (sequences_list, FileInfo)."""
        from AlignAIR.Utilities.file_processing import FILE_ROW_COUNTERS

        file_path = config.sequences_path
        file_name = pathlib.Path(file_path).stem
        file_type = pathlib.Path(file_path).suffix.lstrip('.').lower()
        file_size = os.path.getsize(file_path)

        # Count rows
        if file_type in FILE_ROW_COUNTERS:
            n_sequences = FILE_ROW_COUNTERS[file_type](file_path)
        else:
            # Fallback: count lines minus header
            with open(file_path) as f:
                n_sequences = sum(1 for _ in f) - 1

        # Read sequences
        sequences = self._read_sequences(file_path, file_type)

        file_info = FileInfo(
            file_path=file_path,
            file_name=file_name,
            file_type=file_type,
            n_sequences=len(sequences),
            file_size_bytes=file_size,
        )

        return sequences, file_info

    def _read_sequences(self, file_path: str, file_type: str) -> List[str]:
        """Read sequences from a CSV/TSV/FASTA file."""
        if file_type in ('csv', 'tsv'):
            import pandas as pd
            sep = ',' if file_type == 'csv' else '\t'
            df = pd.read_csv(file_path, sep=sep)
            if 'sequence' in df.columns:
                return df['sequence'].tolist()
            # Try first column
            return df.iloc[:, 0].tolist()
        elif file_type in ('fasta', 'fa'):
            from Bio import SeqIO
            return [str(record.seq) for record in SeqIO.parse(file_path, 'fasta')]
        else:
            raise ModelLoadError(f"Unsupported file type: {file_type}")

    def _infer_chain_type(self, bundle_config, dataconfig) -> str:
        """Infer chain type string from bundle and dataconfig."""
        model_type = getattr(bundle_config, 'model_type', 'single_chain')
        if model_type == 'multi_chain':
            return 'multi'

        has_d = getattr(bundle_config, 'has_d_gene', True)
        if has_d:
            return 'heavy'
        return 'light'

    def _read_fingerprint(self, model_dir: str) -> str:
        """Read bundle fingerprint."""
        fp_path = pathlib.Path(model_dir) / "fingerprint.txt"
        if fp_path.exists():
            return fp_path.read_text().strip()
        return "unknown"
