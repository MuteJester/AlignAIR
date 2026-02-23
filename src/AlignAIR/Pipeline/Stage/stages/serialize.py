"""Serialize stages — CSV and AIRR output formatters."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from AlignAIR.Pipeline.Stage.protocol import Stage, StageContext

logger = logging.getLogger("AlignAIR.Pipeline")


def _format_likelihoods(likelihood_list: List[np.ndarray]) -> List[str]:
    """Convert per-sequence likelihood arrays to parseable semicolon-delimited strings.

    Before: [0.7512059  0.6461257  0.49320796]  (numpy repr, unparseable)
    After:  0.751206;0.646126;0.493208           (semicolon-delimited, 6 decimal places)
    """
    return [';'.join(f'{x:.6f}' for x in arr) for arr in likelihood_list]


def _clean_chain_type(raw) -> str:
    """Strip enum prefix from chain type values.

    Before: ChainType.BCR_HEAVY
    After:  BCR_HEAVY
    """
    if hasattr(raw, 'value'):
        return str(raw.value)
    s = str(raw)
    if '.' in s:
        return s.split('.', 1)[1]
    return s


class CSVSerializeStage(Stage):
    """Serializes pipeline results to CSV format.

    Output format (Pipeline 3.0):
    - sequence_id: 0-based index for linking back to input
    - sequence: the input sequence
    - Columns grouped by gene: V block, D block (if applicable), J block
    - Likelihoods as semicolon-delimited floats (6 decimal places)
    - chain_type as clean string (e.g. BCR_HEAVY, not ChainType.BCR_HEAVY)
    - indels rounded to integer
    - Enrichment fields: locus, junction, v_identity, stop_codon, vj_in_frame, etc.
    """

    reads = frozenset({
        "config", "file_info", "model", "sequences",
        "processed_predictions", "selected_allele_calls",
        "likelihoods_of_selected_alleles", "germline_alignments",
    })
    writes = frozenset({"output_path"})

    def run(self, context: StageContext) -> Dict[str, Any]:
        config = context.config
        file_info = context["file_info"]
        model = context["model"]
        sequences = context["sequences"]
        preds = context["processed_predictions"]
        allele_calls = context["selected_allele_calls"]
        likelihoods = context["likelihoods_of_selected_alleles"]
        germline = context["germline_alignments"]
        has_d = model.has_d_gene

        logger.info("Finalizing results and saving to CSV...")

        # Build columns in logical order
        columns = {}

        # 1. Identity
        columns['sequence_id'] = list(range(len(sequences)))
        columns['sequence'] = sequences

        # 2. V gene block
        columns['v_call'] = [','.join(calls) for calls in allele_calls['v']]
        columns['v_sequence_start'] = [m['start_in_seq'] for m in germline['v']]
        columns['v_sequence_end'] = [m['end_in_seq'] for m in germline['v']]
        columns['v_germline_start'] = [max(0, m['start_in_ref']) for m in germline['v']]
        columns['v_germline_end'] = [m['end_in_ref'] for m in germline['v']]
        columns['v_likelihoods'] = _format_likelihoods(likelihoods['v'])

        # 3. D gene block (if applicable)
        if has_d:
            columns['d_call'] = [','.join(calls) for calls in allele_calls['d']]
            columns['d_sequence_start'] = [m['start_in_seq'] for m in germline['d']]
            columns['d_sequence_end'] = [m['end_in_seq'] for m in germline['d']]
            columns['d_germline_start'] = [abs(m['start_in_ref']) for m in germline['d']]
            columns['d_germline_end'] = [m['end_in_ref'] for m in germline['d']]
            columns['d_likelihoods'] = _format_likelihoods(likelihoods['d'])

        # 4. J gene block
        columns['j_call'] = [','.join(calls) for calls in allele_calls['j']]
        columns['j_sequence_start'] = [m['start_in_seq'] for m in germline['j']]
        columns['j_sequence_end'] = [m['end_in_seq'] for m in germline['j']]
        columns['j_germline_start'] = [max(0, m['start_in_ref']) for m in germline['j']]
        columns['j_germline_end'] = [m['end_in_ref'] for m in germline['j']]
        columns['j_likelihoods'] = _format_likelihoods(likelihoods['j'])

        # 5. Scalars
        columns['mutation_rate'] = preds['mutation_rate'].reshape(-1)
        columns['indels'] = np.rint(preds['indel_count'].reshape(-1)).astype(int)
        columns['productive'] = preds['productive'].reshape(-1)

        # 6. Chain type
        if has_d:
            columns['chain_type'] = _clean_chain_type(model.dataconfig.metadata.chain_type)

        # Handle multi-chain type column
        if 'type_' in preds:
            from AlignAIR.Data.encoders import ChainTypeOneHotEncoder
            chaintype_ohe = ChainTypeOneHotEncoder(chain_types=model.dataconfig.chain_types())
            decoded = chaintype_ohe.decode(preds['type_'])
            columns['chain_type'] = [_clean_chain_type(ct) for ct in decoded]

        # 7. Enrichment fields from AIRR module
        from AlignAIR.Pipeline.AIRR import build_csv_enrichment
        enrichment = build_csv_enrichment(
            sequences, allele_calls, germline, likelihoods,
            preds, model.dataconfig,
        )
        columns['sequence_length'] = enrichment['sequence_length']
        columns['locus'] = enrichment['locus']
        columns['v_sequence'] = enrichment['v_sequence']
        if has_d:
            columns['d_sequence'] = enrichment['d_sequence']
        columns['j_sequence'] = enrichment['j_sequence']
        columns['np1_length'] = enrichment['np1_length']
        columns['np2_length'] = enrichment['np2_length']
        columns['junction'] = enrichment['junction']
        columns['junction_aa'] = enrichment['junction_aa']
        columns['junction_length'] = enrichment['junction_length']
        columns['v_identity'] = enrichment['v_identity']
        columns['stop_codon'] = enrichment['stop_codon']
        columns['vj_in_frame'] = enrichment['vj_in_frame']

        final_csv = pd.DataFrame(columns)

        # Determine output path
        save_path = config.save_path
        path_obj = Path(save_path)
        if path_obj.suffix.lower() == '.csv':
            final_csv_path = path_obj
            os.makedirs(final_csv_path.parent, exist_ok=True)
        else:
            os.makedirs(path_obj, exist_ok=True)
            file_to_save = f"{file_info.file_name}_alignairr_results.csv"
            final_csv_path = path_obj / file_to_save

        final_csv.to_csv(final_csv_path, index=False)
        logger.info("Results saved at %s", final_csv_path)

        return {"output_path": str(final_csv_path)}


class AIRRSerializeStage(Stage):
    """Serializes pipeline results to AIRR TSV format."""

    reads = frozenset({
        "config", "file_info", "model", "sequences",
        "processed_predictions", "selected_allele_calls",
        "likelihoods_of_selected_alleles", "germline_alignments",
    })
    writes = frozenset({"output_path"})

    def run(self, context: StageContext) -> Dict[str, Any]:
        config = context.config
        file_info = context["file_info"]
        model = context["model"]
        sequences = context["sequences"]
        preds = context["processed_predictions"]
        allele_calls = context["selected_allele_calls"]
        likelihoods = context["likelihoods_of_selected_alleles"]
        germline = context["germline_alignments"]

        logger.info("Finalizing results in AIRR format...")

        from AlignAIR.Pipeline.AIRR import build_airr_dataframe

        df = build_airr_dataframe(
            sequences=sequences,
            allele_calls=allele_calls,
            germline_alignments=germline,
            likelihoods=likelihoods,
            processed_predictions=preds,
            dataconfig=model.dataconfig,
        )

        # Determine output path
        save_path = config.save_path
        path_obj = Path(save_path)
        if path_obj.suffix.lower() == '.tsv':
            final_path = path_obj
            os.makedirs(final_path.parent, exist_ok=True)
        else:
            os.makedirs(path_obj, exist_ok=True)
            file_to_save = f"{file_info.file_name}_alignairr_results.tsv"
            final_path = path_obj / file_to_save

        df.to_csv(final_path, sep='\t', index=False)
        logger.info("AIRR results saved at %s", final_path)

        return {"output_path": str(final_path)}
