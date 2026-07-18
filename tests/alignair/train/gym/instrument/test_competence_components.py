from alignair.train.gym.instrument.competence import CompetenceMetric


def _rec(v, j, region, errs):
    return {"v_call_correct": v, "j_call_correct": j, "region_acc": region,
            "coord_errs": errs}


def test_components_separates_allele_coords_region():
    m = CompetenceMetric(coord_tol=1.0)
    recs = [
        _rec(1, 1, 1.0, [0.0, 0.0]),   # all perfect
        _rec(0, 0, 0.5, [5.0, 5.0]),   # allele wrong, coords outside tol, region 0.5
    ]
    comp = m.components(recs)
    # allele: rec0 = 1.0, rec1 = 0.0 -> mean 0.5
    assert abs(comp["allele"]["S"] - 0.5) < 1e-9
    # coords: rec0 within tol -> 1.0, rec1 outside -> 0.0 -> mean 0.5
    assert abs(comp["coords"]["S"] - 0.5) < 1e-9
    # region: (1.0 + 0.5)/2 = 0.75
    assert abs(comp["region"]["S"] - 0.75) < 1e-9


def test_components_skips_absent_submetrics():
    m = CompetenceMetric(coord_tol=2.0)
    # a record with no allele keys (e.g. D-only-unsupervised edge) still yields coords+region
    recs = [{"region_acc": 1.0, "coord_errs": [1.0, 1.0]}]
    comp = m.components(recs)
    assert comp["coords"]["S"] == 1.0 and comp["region"]["S"] == 1.0
    assert comp["allele"]["n"] == 0          # no allele sub-metric present
