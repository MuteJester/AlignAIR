# Convergence study -- per-checkpoint metrics

Budget: +300000 steps. Requested (relative) checkpoint offsets: [25000, 75000, 150000, 300000].
Cohort fingerprint: `b962135fc0ef62cf59dceb50eb202eb79a476817ae079511999595392d7463e9` (dc=HUMAN_IGH_OGRDB, seed=20250722, n_per_stratum=1000).

Selection (§5) is a SEPARATE, later step -- this table is descriptive only.

| name | step | clean_v_acc | clean_j_acc | v_macro | j_macro | d_allele_macro | d_gene_acc | trunc_v_acc | shortread_acc | macro_minus_micro | conf_set_mean | conf_set_p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent | 18000 | 0.9841 | 0.9957 | 0.9819 | 0.9957 | 0.7972 | 0.8242 | 0.8890 | 0.2589 | -0.0023 | 1.1225 | 2.0000 |
| cont_step43000 | 43000 | 0.9870 | 0.9971 | 0.9853 | 0.9972 | 0.8071 | 0.8329 | 0.8847 | 0.2772 | -0.0017 | 1.1643 | 2.0000 |
| cont_step93000 | 93000 | 0.9870 | 0.9971 | 0.9865 | 0.9972 | 0.8013 | 0.8487 | 0.8804 | 0.2538 | -0.0005 | 1.1081 | 2.0000 |
| cont_step168000 | 168000 | 0.9870 | 0.9971 | 0.9846 | 0.9972 | 0.8110 | 0.8329 | 0.8977 | 0.2873 | -0.0024 | 1.1326 | 2.0000 |
| cont_step318000 | 318000 | 0.9899 | 0.9971 | 0.9911 | 0.9972 | 0.8199 | 0.8573 | 0.8963 | 0.2904 | 0.0012 | 1.0793 | 2.0000 |
