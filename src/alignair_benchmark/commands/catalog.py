"""CLI handlers for printing benchmark catalogs and contracts."""
from __future__ import annotations

import json

from ..core import artifact_contract_catalog, criteria_catalog, metric_spec_catalog, scenario_axes_catalog
from ..evaluation import comparison_policy_catalog, prediction_contract, scoring_manifest_catalog
from ..generation import measurement_scenario_catalog, validate_measurement_scenario_catalog


def register_catalog_commands(subparsers) -> None:
    """Register catalog-printing commands."""

    criteria_parser = subparsers.add_parser("criteria", help="print the benchmark criteria catalog")
    criteria_parser.set_defaults(func=criteria)

    contract_parser = subparsers.add_parser("contract", help="print the normalized prediction contract")
    contract_parser.set_defaults(func=contract)

    metrics_parser = subparsers.add_parser("metrics", help="print the benchmark metric registry")
    metrics_parser.set_defaults(func=metrics)

    scoring_manifest_parser = subparsers.add_parser(
        "scoring-manifest",
        help="print the scoring component manifest",
    )
    scoring_manifest_parser.set_defaults(func=scoring_manifest)

    measurement_scenarios_parser = subparsers.add_parser(
        "measurement-scenarios",
        help="print measurement-aligned GenAIRR scenario mappings",
    )
    measurement_scenarios_parser.set_defaults(func=measurement_scenarios)

    artifact_contracts_parser = subparsers.add_parser(
        "artifact-contracts",
        help="print registered benchmark artifact contracts",
    )
    artifact_contracts_parser.set_defaults(func=artifact_contracts)

    comparison_policies_parser = subparsers.add_parser(
        "comparison-policies",
        help="print built-in comparison policy templates",
    )
    comparison_policies_parser.set_defaults(func=comparison_policies)


def criteria(args) -> None:
    payload = {"criteria": criteria_catalog(), "scenario_axes": scenario_axes_catalog()}
    print(json.dumps(payload, indent=2, sort_keys=True))


def contract(args) -> None:
    print(json.dumps({"prediction_contract": prediction_contract()}, indent=2, sort_keys=True))


def metrics(args) -> None:
    print(json.dumps({"metric_registry": metric_spec_catalog()}, indent=2, sort_keys=True))


def scoring_manifest(args) -> None:
    print(json.dumps({"scoring_manifest": scoring_manifest_catalog()}, indent=2, sort_keys=True))


def measurement_scenarios(args) -> None:
    print(
        json.dumps(
            {
                "measurement_scenarios": measurement_scenario_catalog(),
                "validation": validate_measurement_scenario_catalog(),
            },
            indent=2,
            sort_keys=True,
        )
    )


def artifact_contracts(args) -> None:
    print(json.dumps({"artifact_contracts": artifact_contract_catalog()}, indent=2, sort_keys=True))


def comparison_policies(args) -> None:
    print(json.dumps({"comparison_policies": comparison_policy_catalog()}, indent=2, sort_keys=True))
