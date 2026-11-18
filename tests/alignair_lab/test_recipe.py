"""The retrain recipe: the product run's settings (``chip_full_w64_5272fc63``'s launcher and ``launch.json``) with
novel alleles in its banks, and its cooldown."""
import pytest

from alignair.data.flanks import FlankPolicy
from alignair.train.records.novel import NovelAlleles
from alignair_lab.recipe import cooldown_config, retrain_config

pytestmark = pytest.mark.component


def test_the_retrain_recipe_is_the_product_runs_with_novel_alleles():
    cfg = retrain_config()
    assert (cfg.dataconfigs, cfg.steps, cfg.batch_size, cfg.lr, cfg.seed, cfg.init_seed, cfg.val_every,
            cfg.data_workers, dict(cfg.model_options), cfg.schedule) == (
        ("HUMAN_IGH_OGRDB",), 300_000, 64, 3e-4, 0, None, 2000, 7, {}, None)
    assert cfg.flanks == FlankPolicy() and cfg.novel_alleles == NovelAlleles()


def test_the_cooldown_resumes_the_run_to_320k_at_a_tenth_of_the_rate(tmp_path):
    checkpoint = str(tmp_path / "model.step300000.alignair")
    cool = cooldown_config(retrain_config(), checkpoint)
    assert (cool.steps, cool.lr, cool.resume_path, cool.novel_alleles) == (320_000, 3e-5, checkpoint, NovelAlleles())


def test_a_retrain_of_several_loci_names_its_dataconfigs():
    names = ("HUMAN_IGH_OGRDB", "HUMAN_IGK_OGRDB", "HUMAN_TCRB_IMGT")
    assert retrain_config(dataconfigs=names).dataconfigs == names
