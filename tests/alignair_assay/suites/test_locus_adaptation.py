"""Suites adapt to their reference: every recipe starts from ``recipes.experiment_on`` (GenAIRR's curatable refdata
allowed, as AlignAIR's trainer allows it), SHM is skipped on a TCR reference (``recipes.mutated``), and a stratum
that does not apply to the reference (``Stratum.applies``) is skipped and recorded; a suite none of whose strata
applies is not applicable to that reference (``engine.SuiteNotApplicable``)."""
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("GenAIRR")

import alignair_assay.suites as suites_package  # noqa: E402
from alignair_assay.suites import get_suite  # noqa: E402
from alignair_assay.suites.base import Stratum, Suite, has_d  # noqa: E402
from alignair_assay.suites.engine import SuiteNotApplicable, generate_suite  # noqa: E402
from alignair_assay.suites.recipes import experiment_on  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.genairr_build]


def _generate(suite, reference, **overrides):
    options = dict(reference_name=reference, split="dev", seed=7, scale=0.02, workers=1) | overrides
    return generate_suite(get_suite(suite) if isinstance(suite, str) else suite, **options)


def _plain(ctx):
    return experiment_on(ctx.config).recombine()


TOY = Suite(name="T_LOCUS", slug="toy_locus", title="Toy", purpose="A stratum that needs a D beside one that does not.",
            primary=("v.top1_in_set",),
            strata=(Stratum("plain", 2, _plain), Stratum("needs_d", 2, _plain, applies=has_d)))
ONLY_D = Suite(name="T_ONLY_D", slug="toy_only_d", title="Toy", purpose="Every stratum needs a D.",
               primary=("v.top1_in_set",), strata=(Stratum("needs_d", 2, _plain, applies=has_d),))


def test_every_recipe_starts_from_experiment_on():
    package = Path(suites_package.__file__).parent
    direct = [f"{path.relative_to(package)}:{n}" for path in sorted(package.rglob("*.py"))
              for n, line in enumerate(path.read_text().splitlines(), 1)
              if re.search(r"Experiment\.on\(", line) and "def experiment_on" not in line
              and not (path.name == "recipes.py" and "return Experiment.on(config)" in line)]
    assert direct == []


def test_a_tcr_reference_builds_unmutated_reads():
    generated = _generate("S12", "HUMAN_TCRB_IMGT")
    assert generated.cases and all(case["record"]["n_mutations"] == 0 for case in generated.cases)


def test_a_stratum_that_does_not_apply_is_skipped_and_recorded():
    generated = _generate(TOY, "HUMAN_IGK_OGRDB", scale=1.0)
    assert generated.skipped == ("needs_d",)
    assert {case["stratum"] for case in generated.cases} == {"plain"} and list(generated.stats) == ["plain"]
    assert _generate(TOY, "HUMAN_IGH_OGRDB", scale=1.0).skipped == ()


def test_a_suite_none_of_whose_strata_applies_is_not_applicable():
    with pytest.raises(SuiteNotApplicable, match="HUMAN_IGK_OGRDB"):
        _generate(ONLY_D, "HUMAN_IGK_OGRDB", scale=1.0)


def test_build_skips_a_suite_that_does_not_apply(tmp_path, capsys, monkeypatch):
    """``build`` reports a suite that does not apply to the reference, writes nothing for it and builds the others
    (S6 is made not applicable here; Task-3 markings make it so on a light chain)."""
    import alignair_assay.cli.build as cli_build
    from alignair_assay.cli import main
    real = cli_build.build_suite

    def build_suite(name, layout, **options):
        if name == "S6":
            raise SuiteNotApplicable("suite S6 is not applicable to HUMAN_IGK_OGRDB: none of its strata applies")
        return real(name, layout, **options)

    monkeypatch.setattr(cli_build, "build_suite", build_suite)
    assert main(["build", "--suites", "S6,S12", "--run-dir", str(tmp_path), "--reference", "HUMAN_IGK_OGRDB",
                 "--scale", "0.02", "--workers", "1"]) == 0
    assert "not applicable to HUMAN_IGK_OGRDB: S6" in capsys.readouterr().out
    assert not (tmp_path / "suites" / "S6").exists()
    assert json.loads((tmp_path / "suites" / "S12" / "build.json").read_text())["skipped_strata"] == []
