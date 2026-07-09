"""Shared CLI option constants."""
from __future__ import annotations

from ..core import artifact_contract_catalog
from ..evaluation import MULTIPLE_COMPARISON_CORRECTIONS, comparison_policy_catalog

PREDICTION_FORMATS = ("jsonl", "airr", "airr-tsv", "airr-csv")
READINESS_PROFILE_CHOICES = ("smoke", "development", "assay", "allele_complete", "allele_stratified")
SUITE_READINESS_PROFILE_CHOICES = ("smoke", "development", "assay")
COMPARISON_POLICY_CHOICES = tuple(row["name"] for row in comparison_policy_catalog())
ARTIFACT_KIND_CHOICES = tuple(row["kind"] for row in artifact_contract_catalog())
