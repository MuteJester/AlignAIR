"""GenotypeAdjustmentStage — conditionally adjusts allele likelihoods for genotype."""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from tqdm.auto import tqdm

from AlignAIR.Pipeline.Stage.protocol import ConditionalStage, StageContext

logger = logging.getLogger("AlignAIR.Pipeline")


class GenotypeAdjustmentStage(ConditionalStage):
    """Zeroes out allele likelihoods for alleles not in the genotype.

    This is a ConditionalStage — skipped entirely if no custom genotype is provided.
    """

    reads = frozenset({"config", "processed_predictions", "model"})
    writes = frozenset({"processed_predictions"})

    def should_run(self, context: StageContext) -> bool:
        return context.config.custom_genotype_path is not None

    def _run(self, context: StageContext) -> Dict[str, Any]:
        config = context.config
        preds = context["processed_predictions"]
        model = context["model"]
        dataconfig = model.dataconfig

        logger.info("Adjusting likelihoods given genotype...")

        # Build allele name/index mappings
        alleles_dict = {}
        for gene in ['v', 'd', 'j']:
            names = sorted(a.name for a in dataconfig.allele_list(gene))
            if gene == 'd':
                names.append('Short-D')
            alleles_dict[gene] = names

        name_to_idx = {gene: {name: i for i, name in enumerate(names)} for gene, names in alleles_dict.items()}
        idx_to_name = {gene: {i: name for i, name in enumerate(names)} for gene, names in alleles_dict.items()}

        # Build genotype checker
        all_allele_names = set()
        for gene in ['v', 'd', 'j']:
            all_allele_names.update(a.name for a in dataconfig.allele_list(gene))

        def is_in_genotype(allele_name):
            return allele_name in all_allele_names

        def bounded_redistribution(likelihoods_dict):
            geno = {k: v for k, v in likelihoods_dict.items() if is_in_genotype(k)}
            non_geno = {k: v for k, v in likelihoods_dict.items() if not is_in_genotype(k)}
            total_geno = sum(geno.values())
            total_non_geno = sum(non_geno.values())
            if total_geno > 0:
                factor = total_non_geno / total_geno
                geno = {k: min(1, v + v * factor) for k, v in geno.items()}
            return geno

        # Apply adjustment per gene
        adjusted_preds = dict(preds)  # shallow copy
        for gene in ['v', 'd', 'j']:
            key = f'{gene}_allele'
            if adjusted_preds.get(key) is None:
                continue

            processed = []
            arr = adjusted_preds[key]
            for row in tqdm(arr, desc=f"Processing {gene} allele likelihoods"):
                dict_form = {idx_to_name[gene][i]: row[i] for i in range(len(row))}
                redistributed = bounded_redistribution(dict_form)
                processed.append(redistributed)

            # Convert back to numpy
            adjusted_preds[key] = np.vstack([list(d.values()) for d in processed])

        return {"processed_predictions": adjusted_preds}
