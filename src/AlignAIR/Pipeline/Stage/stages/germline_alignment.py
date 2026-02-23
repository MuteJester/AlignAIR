"""GermlineAlignmentStage — align predicted segments with germline references."""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from AlignAIR.Pipeline.Stage.protocol import Stage, StageContext

logger = logging.getLogger("AlignAIR.Pipeline")


class GermlineAlignmentStage(Stage):
    """Aligns predicted sequence segments with germline reference alleles."""

    reads = frozenset({"model", "sequences", "processed_predictions", "selected_allele_calls"})
    writes = frozenset({"germline_alignments"})

    def run(self, context: StageContext) -> Dict[str, Any]:
        model = context["model"]
        sequences = context["sequences"]
        preds = context["processed_predictions"]
        allele_calls = context["selected_allele_calls"]
        has_d = model.has_d_gene
        dataconfig = model.dataconfig

        logger.info("Aligning with germline alleles...")

        from AlignAIR.PostProcessing import HeuristicReferenceMatcher

        # Build reference maps
        reference_map = {}
        reference_map['v'] = {a.name: a.ungapped_seq.upper() for a in dataconfig.allele_list('v')}
        reference_map['j'] = {a.name: a.ungapped_seq.upper() for a in dataconfig.allele_list('j')}
        if has_d:
            reference_map['d'] = {a.name: a.ungapped_seq.upper() for a in dataconfig.allele_list('d')}
            reference_map['d']['Short-D'] = ''

        # Build segments dict
        genes = ['v', 'j']
        if has_d:
            genes.append('d')

        segments = {}
        for gene in genes:
            segments[gene] = (preds[f'{gene}_start'], preds[f'{gene}_end'])

        indel_counts = preds['indel_count'].round().astype(int)

        # Align each gene
        germline_alignments = {}
        for gene in genes:
            ref_alleles = reference_map[gene]
            starts, ends = segments[gene]
            mapper = HeuristicReferenceMatcher(ref_alleles)
            mappings = mapper.match(
                sequences=sequences,
                starts=starts,
                ends=ends,
                alleles=[calls[0] for calls in allele_calls[gene]],
                _gene=gene,
                indel_counts=indel_counts,
            )
            germline_alignments[gene] = mappings

        return {"germline_alignments": germline_alignments}
