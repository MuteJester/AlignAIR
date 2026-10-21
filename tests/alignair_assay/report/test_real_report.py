"""The real-data report: markdown rendered from the plain summary dict ``workflow.real.run_real`` writes
(``real_summary.json``). Every summary here is built by hand."""
import copy
import re

from alignair_assay.report.real import (
    DETAIL_SECTIONS,
    MAX_FLAGS,
    SUMMARY_SECTIONS,
    real_report_markdown,
)

EXCLUSIONS = ("excluded:dropped_truth", "excluded:not_captured", "excluded:uncalled", "excluded:unmapped",
              "excluded:novel_gene", "excluded:nonfunctional", "excluded:beyond_318", "excluded:half_resolved",
              "excluded:low_support")
SD3_GENES = ["IGHV1-24", "IGHV1-38-4", "IGHV1-58", "IGHV1-69", "IGHV2-70", "IGHV2-70D", "IGHV3-11", "IGHV3-13",
             "IGHV3-35", "IGHV3-38", "IGHV3-43", "IGHV3-43D", "IGHV3-49", "IGHV3-53", "IGHV3-66", "IGHV3-7",
             "IGHV3-72", "IGHV3-74", "IGHV3-9", "IGHV4-4", "IGHV5-51"]


def _cell(mean, n):
    return {"mean": mean, "n": n, "n_applicable": n, "coverage": 1.0 if n else 0.0}


def _entry(a, b, n, k, verdict="tie_or_inconclusive", test="clustered_mcnemar"):
    delta = b - a
    return {"a": _cell(a, n), "b": _cell(b, n), "n_paired": n, "n_clusters": k, "mean_a_paired": a,
            "mean_b_paired": b, "delta": delta, "ci95": [delta - 0.031, delta + 0.031], "test": test,
            "p_value": 0.18, "verdict": verdict}


def _tool(classes: dict, *, invalid=0, scale=1) -> dict:
    classes = {name: count * scale for name, count in classes.items()}
    scored = sum(count for name, count in classes.items() if not name.startswith("excluded:"))

    def rate(*names):
        return sum(classes.get(name, 0) for name in names) / scored if scored else None

    summary = {"real.v.unsupported": _cell(rate("unsupported_allele", "unsupported_gene", "no_call"), scored),
               "real.v.unsupported_allele": _cell(rate("unsupported_allele"), scored),
               "real.v.unsupported_gene": _cell(rate("unsupported_gene"), scored),
               "real.v.no_call": _cell(rate("no_call"), scored),
               "real.v.any_listed_supported": _cell(0.8, scored),
               "real.v.known_top1": _cell(1.0, 30 * scale), "real.v.known_gene_top1": _cell(1.0, 30 * scale),
               "real.v.novel_known_call": _cell(0.75, 4 * scale)}
    exclusions = {name: classes.get(name, 0) for name in EXCLUSIONS}
    supported = classes.get("supported", 0)
    variant = {**{k: v for k, v in classes.items() if k != "supported"}, "excluded:low_support": supported}
    return {"v_classes": classes, "exclusions": exclusions, "summary": summary,
            "consensus_weighted": {"real.v.unsupported": 0.197, "weight": scored + 30},
            "novel_known": {"reads": 4, "to_known_gene": 0.75, "calls": {"known gene": 3, "IGHV3-23": 1}},
            "gene_support": {"v_classes": variant, "exclusions": {**exclusions, "excluded:low_support": supported},
                             "summary": {}},
            "matching": {"rows": sum(classes.values()), "duplicates": 0, "invalid_rows": invalid, "missing": 0,
                         "extra": 0}}


def _tools(scale=1) -> dict:
    alignair = {"supported": 119, "unsupported_allele": 7, "unsupported_gene": 19, "excluded:not_captured": 308,
                "excluded:novel_gene": 47}
    return {"shared": {"igblast": _tool({"supported": 117, "unsupported_allele": 14, "unsupported_gene": 20,
                                         "excluded:not_captured": 307, "excluded:novel_gene": 42}, invalid=1,
                                        scale=scale),
                       "alignair": _tool(alignair, invalid=462, scale=scale),
                       "alignair-api": _tool(alignair, invalid=462, scale=scale)},
            "genotype": {"igblast": _tool({"supported": 121, "excluded:not_captured": 327,
                                           "excluded:novel_gene": 52}, invalid=1, scale=scale),
                         "alignair-api": _tool({"supported": 129, "excluded:not_captured": 319,
                                                "excluded:novel_gene": 52}, invalid=462, scale=scale)}}


def _r1(n, k, verdict="tie_or_inconclusive") -> dict:
    return {"real.v.unsupported": _entry(0.196, 0.175, n, k, verdict),
            "real.v.unsupported_allele": _entry(0.056, 0.042, n, k),
            "real.v.unsupported_gene": _entry(0.140, 0.133, n, k),
            "real.v.no_call": _entry(0.0, 0.0, n, k),
            "real.v.known_top1": _entry(1.0, 1.0, 30, 30, test="mcnemar_exact"),
            "real.v.known_gene_top1": _entry(1.0, 1.0, 30, 30, test="mcnemar_exact")}


def _comparison(a, b, arm, n, k, *, verdict="tie_or_inconclusive", strata=("multiplex/IGHG", "multiplex/IGHM")):
    overall = _r1(n, k, verdict)
    if arm == "genotype":
        overall = {name: _entry(0.0, 0.0, n - 25, k - 25) for name in overall if "unsupported" in name
                   or "no_call" in name} | {"real.v.known_top1": _entry(1.0, 0.9667, 30, 30, test="mcnemar_exact"),
                                            "real.v.known_gene_top1": _entry(1.0, 1.0, 30, 30, test="mcnemar_exact")}
    out = {"a": a, "b": b, "arm": arm, "status": "compared", "n_reads": 500, "n_clusters": 488,
           "multi_read_clusters": 9, "excluded": {"v": 357, "d": 0, "j": 0}, "overall": overall,
           "one_sided": {"a": {"reads": 3, "reasons": {"excluded:not_captured": 3}, "documented": 1,
                               "documented_wrong": 0},
                         "b": {"reads": 40, "reasons": {"excluded:not_captured": 31, "excluded:uncalled": 9},
                               "documented": 6, "documented_wrong": 4}},
           "informational": {"real.v.any_listed_supported": {"a": _cell(0.825, n), "b": _cell(0.825, n)},
                             "real.v.unsupported_observable": {"a": _cell(0.121, n), "b": _cell(0.098, n)},
                             "real.v.known_top1_observable": {"a": _cell(1.0, 30), "b": _cell(0.9667, 30)}},
           "strata": {label: {"real.v.unsupported": _entry(0.2, 0.18, n // 2, k // 2)} for label in strata},
           "clusterings": {
               "exact": {"n_clusters": k, "comparison": {"real.v.unsupported": _entry(0.196, 0.175, n, k),
                                                         "real.v.known_top1": _entry(1.0, 1.0, 30, 30)}},
               "sl15": {"n_clusters": k - 20, "comparison": {"real.v.unsupported": _entry(0.196, 0.175, n, k - 20),
                                                             "real.v.known_top1": _entry(1.0, 1.0, 30, 28)}},
               "junction_length": {"n_clusters": 31, "comparison": {
                   "real.v.unsupported": _entry(0.196, 0.175, n, 31),
                   "real.v.known_top1": _entry(1.0, 1.0, 30, 12)}}},
           "sensitivity": {
               "sd3": {"genes": SD3_GENES, "excluded_reads": 104,
                       "comparison": {"real.v.unsupported": _entry(0.219, 0.192, 73, 70)}},
               "gene_support": {"thresholds": {"min_clonotypes": 10, "min_naive_clonotypes": 3},
                                "excluded": {"v": 481, "d": 0, "j": 0},
                                "comparison": {"real.v.unsupported": _entry(1.0, 1.0, 19, 19, test="mcnemar_exact")}}}}
    return out


def _arm_delta(tool, n=121, k=117) -> dict:
    return {"tool": tool, "a": "shared", "b": "genotype", "informational": True, "n_reads": 500, "n_clusters": 488,
            "multi_read_clusters": 9, "excluded": {"v": 379, "d": 0, "j": 0},
            "overall": {"real.v.known_top1": _entry(1.0, 1.0, 30, 30, test="mcnemar_exact"),
                        "real.v.known_gene_top1": _entry(1.0, 1.0, 30, 30, test="mcnemar_exact"),
                        "real.v.unsupported": _entry(0.033, 0.0, n, k, "b_better"),
                        "real.v.no_call": _entry(0.0, 0.0, n, k)}}


def _deciles(values) -> dict:
    keys = ("min", "p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90", "max")
    return {"n": 100, **dict(zip(keys, values)), "median": values[5]}


def _preprocess(protocol) -> dict:
    multiplex = protocol == "multiplex"
    return {
        "status": "done", "protocol": protocol, "quality": "full",
        "quality_steps": "FilterSeq quality -q 20 on each mate: reads of mean quality below 20 dropped, none shortened",
        "classifier": {"protocol": protocol, "evidence": {
            "pairs": 2000, "race_linker": 0.0 if multiplex else 0.951, "vh_primer": 0.957 if multiplex else 0.01,
            "c_primer": 0.9195 if multiplex else 0.02,
            "reason": ("R1 starts with a VH primer in 95.7% and R2 with a C primer in 92.0% of the pairs" if multiplex
                       else "R2[12:16] is the GTAC linker in 95.1% of the pairs")}},
        "accepted": True,
        "assembly_acceptance": {"pairs": 94176, "failed": 14807, "failure_rate": 0.1572, "max_failure_rate": 0.35,
                                "within_line": True, "override": False, "accepted": True},
        "counts": {"raw_pairs": 103049, "quality-r1": 102999, "both-primers": 94176, "assemble": 79369,
                   "unique": 70327, "capped": 500},
        "pass_rates": {"assembly": 0.8428, "yield": 0.6825},
        "isotypes": {"IGHG": 56, "IGHM": 444}, "isotypes_unique": {"IGHG": 7986, "IGHM": 62341},
        "wall_seconds": 203.9,
        "assembly_failures": {"mismatches": 14698, "not_significant": 109, "no_overlap": 0},
        "assembly": {
            "pass": {"n": 79369, "overlap": _deciles([24, 85, 90, 94, 97, 100, 103, 106, 109, 114, 250]),
                     "error": _deciles([0.0, 0.0, 0.0089, 0.0125, 0.0283, 0.0485, 0.0763, 0.1134, 0.1646, 0.2234,
                                        0.2992]),
                     "mates_length": _deciles([500] * 11),
                     "insert_length": _deciles([250, 386, 391, 394, 397, 400, 403, 406, 410, 415, 476])},
            "fail": {"n": 14807, "overlap": _deciles([12, 85, 91, 94, 100, 103, 106, 112, 247, 249, 250]),
                     "error": _deciles([0.25, 0.3178, 0.3373, 0.3565, 0.38, 0.4057, 0.4368, 0.4773, 0.58, 0.6074,
                                        0.684]),
                     "mates_length": _deciles([500] * 11),
                     "kinds": {"mismatches": 14698, "not_significant": 109, "no_overlap": 0}}},
        "steps": [{"name": "quality-r1", "records": 102999, "pass_rate": 0.9995, "seconds": 9.4},
                  {"name": "assemble", "records": 79369, "pass_rate": 0.8428, "seconds": 132.7}]}


def _donor(name, run, sample, protocol, *, quality="full", known=30, scale=1) -> dict:
    strata = (f"{protocol}/IGHG", f"{protocol}/IGHM")
    comparisons = [_comparison("igblast", "alignair", "shared", 143, 139, strata=strata),
                   _comparison("igblast", "alignair-api", "shared", 143, 139, strata=strata),
                   _comparison("igblast", "alignair-api", "genotype", 143, 139, strata=strata)]
    runs = {"shared": {tool: {"status": "ok", "returncode": 0, "error": None, "note": None, "wall_seconds": 6.0,
                              "n_reads": 500, "peak_rss_kb": 364532, "genotype_alleles": None,
                              "assembly": None if tool == "igblast" else {
                                  "written_records": 500, "complete_assemblies": 462, "partial_assemblies": 38,
                                  "failed_assemblies": 0}}
                       for tool in ("igblast", "alignair", "alignair-api")},
            "genotype": {tool: {"status": "ok", "returncode": 0, "error": None, "note": None, "wall_seconds": 9.4,
                                "n_reads": 500, "peak_rss_kb": 772412,
                                "genotype_alleles": {"v": 114, "d": 33, "j": 7},
                                "assembly": None if tool == "igblast" else {
                                    "written_records": 500, "complete_assemblies": 462, "partial_assemblies": 38,
                                    "failed_assemblies": 0}}
                         for tool in ("igblast", "alignair-api")}}
    return {
        "donor": name, "split": "dev", "run": run, "sample": sample, "status": "done", "stopped": None,
        "steps": {"fetch": {"status": "reused", "seconds": 0.1}, "preprocess": {"status": "done", "seconds": 204.0}},
        "fetch": {"quality": quality, "md5": {"reads": "a2b3"}, "source": "odp" if quality == "full" else "ena"},
        "protocol": protocol, "preprocess": _preprocess(protocol),
        "truth": {"reads": 500, "known": known, "novel_known": 4, "known_share": known / 500,
                  "low_known_yield": known / 500 < 0.05, "naive": 34, "anchor": {"donor": 340, "shared": 128,
                                                                                  "none": 32},
                  "clonotypes": 488, "multi_read_clonotypes": 9, "isotypes": {"IGHG": 56, "IGHM": 444},
                  "strata": {strata[0]: 56, strata[1]: 444}},
        "genotype": {"sample": sample, "v_genes": 42, "functional_v_genes": 39, "d_truth": False, "j_truth": False,
                     "captured": 45, "resolved": 44, "absent_genes": ["IGHV1-2"], "dropped_d": 0,
                     "arm_v_alleles": 114, "capture": {
                         "spanned_not_carried": ["IGHV3-11", "IGHV4-30-2"], "carried_not_spanned": [],
                         "half_resolved": ["IGHV3-62"], "absent": ["IGHV3-11"],
                         "uncalled": {"IGHV4-30-2": [{"haplotype": 0, "contig": "c=igh:463603-503524_h=0_i=0_t=1",
                                                      "aligned": 296, "deleted": 3, "state": "present",
                                                      "label": "IGHV4-28*01", "identity": 1.0, "coverage": 1.0,
                                                      "holds": "another gene's sequence"}]}},
                     "ogrdb": {"release_version": 8, "release_date": "2024-01-11"}},
        "runs": runs,
        "oracle_self_check": {"reads": known, "classes": {"excluded:not_captured": 1, "supported": known - 1},
                              "excluded_genes": {"excluded:not_captured": ["IGHV3-30-5"]}, "scored": known - 1,
                              "supported": 1.0, "unsupported": 0.0, "known_top1": 1.0, "known_gene_top1": 1.0,
                              "passed": True, "status": "passed"},
        "tools": _tools(scale), "comparisons": comparisons,
        "arm_deltas": [_arm_delta("igblast"), _arm_delta("alignair-api")]}


def _summary() -> dict:
    w97 = _donor("W-97", "SRR23654538", "P25_I151_S1", "multiplex")
    sc23 = _donor("SC-23", "SRR23446583", "P25_I58_S1", "race", known=150)
    pairs = (("igblast", "alignair", "shared"), ("igblast", "alignair-api", "shared"),
             ("igblast", "alignair-api", "genotype"))
    pooled = {"donors": ["SC-23", "W-97"], "tools": _tools(scale=2),
              "comparisons": [{**_comparison(a, b, arm, 286, 278, strata=("multiplex/IGHM", "race/IGHG")),
                               "donors": ["SC-23", "W-97"]} for a, b, arm in pairs],
              "arm_deltas": [{**_arm_delta(tool, 242, 234), "donors": ["SC-23", "W-97"]}
                             for tool in ("igblast", "alignair-api")],
              "protocols": {protocol: {"donors": [donor], "comparisons": [
                  {**_comparison(a, b, arm, n, k, strata=(f"{protocol}/IGHG", f"{protocol}/IGHM")), "donors": [donor]}
                  for a, b, arm in pairs]}
                  for protocol, donor, n, k in (("multiplex", "W-97", 143, 139), ("race", "SC-23", 150, 147))}}
    for tool in pooled["tools"]["shared"].values():  # the pooled block has no per-run extras
        for key in ("consensus_weighted", "novel_known", "matching"):
            tool.pop(key)
    images = {"sra_tools": "ncbi/sra-tools:3.4.1"
                           "@sha256:4f048ba6754246e1ed7799a2c8ed469a38427f392e403ca08b93c4255104c9ce",
              "samtools": "quay.io/biocontainers/samtools:1.23.1--ha83d96e_0"
                          "@sha256:23cda33a3a42125872766df9aaf1d2db67cdb8c85314b793465188435af31ba6",
              "immcantation": "immcantation/suite:4.8.0"
                              "@sha256:44e228a45b047bdb352363ab20a670336b003e98f3d1bac505f78af41a91ab67",
              "alignair": "alignair/alignair:3.0.0"
                          "@sha256:5df66a34ac7b2b3ece927c49ef440c89dc30a47d028415d924c827b6012ccfd3"}
    return {
        "kind": "alignair_assay.real.summary", "version": "1",
        "config": {"donors": ["W-97", "SC-23"], "split": "dev", "cap": 500, "tools": ["igblast", "alignair",
                                                                                     "alignair-api"],
                   "reference": "HUMAN_IGH_OGRDB", "device": "cpu", "stop_after": None},
        "config_hash": "b0a92a2ac952129697e6ae6462b27bdc182ac8e293633cb2a58c892ef123366d",
        "created_utc": "2026-10-21T00:38:31+00:00",
        "provenance": {
            "git": {"git_commit": "f73925d0000000000000000000000000000000000", "git_dirty": False},
            "images": {name: {"image": ref, "id": f"sha256:{name}0000", "repo_digests": [ref]}
                       for name, ref in images.items()},
            "tools": {"igblast": {"igblastn_sha256": "74250e89d049a07ad9e971b41d77fe8af0929bd811bce2c7a590d003ae0a"
                                                     "51c4"},
                      "alignair": {"image": images["alignair"], "model": "alignair-igh-human@1.0.0",
                                   "model_sha256": "c418a72cda22e81b0ff5f63eba55c4cb61e7fed7e220f6dc64bd5dfafb801b59"},
                      "alignair-api": {"checkpoint_sha256": "c418a72cda22e81b0ff5f63eba55c4cb61e7fed7e220f6dc64bd5df"
                                                            "afb801b59", "device": "cpu",
                                       "torch_version": "2.11.0+cu128"}},
            "reference": {"name": "HUMAN_IGH_OGRDB",
                          "digest": "4d70015675eb18e3af36cc94c516a83aeb7288cc8aa84aaf05d358c25489bf89"},
            "ogrdb": {"germline_set_id": "OGRDB:G00060", "release_version": 8, "release_date": "2024-01-11",
                      "sha256": "fd1f98e9acfb"},
            "igenotyper_commit": "40a00015f000adae477d25e62ec2e9e8dba9da98",
            "fetch_quality": {"SRR23654538": "full", "SRR23446583": "full"}},
        "sensitivity": {"genes": SD3_GENES, "sd3_alleles": 137, "missing_alleles": 33,
                        "cohort": {"subjects": 152, "documented_labels": 174, "undocumented_sequences": 267},
                        "files": {"vdjbase_api": {"md5": "29c3ffae"}, "ogrdb_rev9": {"md5": "1c30b9ab"},
                                  "sd3": {"md5": "8cfa14cb"}}},
        "donors": {"W-97": w97, "SC-23": sc23}, "pooled": pooled,
        "stopped_donors": [], "failed_runs": [], "oracle_failures": [], "oracle_unchecked": []}


def _section(md: str, title: str) -> str:
    """The text of section ``## title``, up to the next ``## ``."""
    start = md.index(f"\n## {title}\n")
    end = md.find("\n## ", start + 1)
    return md[start:end if end != -1 else len(md)]


def _table(text: str) -> list[list[str]]:
    """The first markdown table of ``text``, as rows of stripped cells (the header first, the rule left out)."""
    rows, started = [], False
    for line in text.splitlines():
        if line.startswith("|"):
            started = True
            if not line.startswith("|---"):
                rows.append([cell.strip() for cell in re.split(r"(?<!\\)\|", line)[1:-1]])
        elif started:
            break
    return rows


def _bullets(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("- ")]


# --- the brief's tests -------------------------------------------------------------------------------------------

def test_report_has_every_section():
    md = real_report_markdown(_summary())
    assert md.startswith("# AlignAIR real-data report")
    headings = [line[3:] for line in md.splitlines() if line.startswith("## ")]
    assert headings == [*SUMMARY_SECTIONS, *DETAIL_SECTIONS]
    for title in ("R1 headline (shared arm)", "Genotype arm", "Flags", "Exclusions", "Known answers",
                  "Novel-allele reads", "Sensitivity", "D and J (informational)",
                  "Stratification: protocol × isotype", "Preprocessing", "Truth self-check",
                  "Failed runs and refusals"):
        assert title in headings


def test_low_known_answer_yield_is_flagged():
    summary = _summary()
    summary["donors"]["SC-23"]["truth"].update(known=15, known_share=0.03, low_known_yield=True)
    md = real_report_markdown(summary)
    flags = _bullets(_section(md, "Flags"))
    low = [flag for flag in flags if "Known-answer yield below 5%" in flag]
    assert len(low) == 1 and "SC-23 3.0% (15 of 500 reads)" in low[0] and "W-97" not in low[0]
    known = _section(md, "Known answers")
    assert "15 (**3.0%**, below 5%)" in known and "30 (6.0%)" in known
    # W-97 at 6.0% is not flagged
    assert not any("Known-answer yield" in flag for flag in _bullets(_section(real_report_markdown(_summary()),
                                                                              "Flags")))


def test_lite_quality_is_shown_per_run():
    summary = _summary()
    summary["donors"]["SC-23"]["fetch"]["quality"] = "lite"
    md = real_report_markdown(summary)
    header = md.split("\n## ")[0]
    assert "W-97 SRR23654538 full" in header and "SC-23 SRR23446583 **lite**" in header
    lite = [flag for flag in _bullets(_section(md, "Flags")) if "SRA-Lite" in flag]
    assert len(lite) == 1 and "SC-23 SRR23446583" in lite[0] and "W-97" not in lite[0]
    # full quality everywhere: no flag, every run shown full
    md = real_report_markdown(_summary())
    assert "SC-23 SRR23446583 full" in md.split("\n## ")[0]
    assert not any("SRA-Lite" in flag for flag in _bullets(_section(md, "Flags")))


def test_cluster_count_beside_each_delta():
    md = real_report_markdown(_summary())
    section = _section(md, "R1 headline (shared arm)")
    rows = _table(section[section.index("R1 Δ per scope"):])
    assert rows[0] == ["Scope", "alignair (image) − igblast", "alignair-api − igblast"]
    assert [row[0] for row in rows[1:]] == ["W-97", "SC-23", "Pooled (2 donors)"]
    for row in rows[1:]:
        for cell in row[1:]:
            assert re.search(r"-?\d+\.\d pp \[[-+]\d+\.\d, [-+]\d+\.\d\], (not significant|.* better); "
                             r"\d+ paired \(\d+ clonotypes\)", cell), cell
    assert rows[1][1].endswith("143 paired (139 clonotypes)") and rows[3][2].endswith("286 paired (278 clonotypes)")
    # the components and the detail tables carry the clusters too
    for title in ("R1 per donor", "Sensitivity", "Stratification: protocol × isotype"):
        section = _section(md, title)
        assert section.count(" pp [") == section.count("clonotypes)") > 0, title


# --- the rulings -------------------------------------------------------------------------------------------------

def test_the_summary_page_leads_with_a_three_line_header():
    md = real_report_markdown(_summary())
    title, *header = [line for line in md.split("\n## ")[0].splitlines() if line.strip()]
    assert title == "# AlignAIR real-data report: W-97, SC-23 (dev)"
    assert len(header) == 3 and all(line.startswith("- ") for line in header)
    donors, data, images = header
    assert "W-97, SC-23" in donors and "dev split" in donors and "2026-10-21 00:38 UTC" in donors
    assert "cap 500" in donors and "HUMAN_IGH_OGRDB (4d70015675eb)" in donors
    assert "PRJNA555323" in data and "VDJbase genomic P25" in data and "OGRDB IGH_VDJ rev 8" in data
    assert "IGenotyper 40a00015" in data and "152 subjects" in data
    for digest in ("4f048ba67542", "23cda33a3a42", "44e228a45b04", "5df66a34ac7b"):
        assert digest in images
    assert "igblastn 74250e89d049" in images and "checkpoint c418a72cda22" in images
    order = [md.index(f"\n## {title}\n") for title in (*SUMMARY_SECTIONS, *DETAIL_SECTIONS)]
    assert order == sorted(order)
    assert md.index("\n---\n") < md.index("\n## R1 per donor\n")  # the summary page, then the detail


def test_the_bottom_line_leads_per_protocol_with_the_pre_declared_primary():
    md = real_report_markdown(_summary())
    section = _section(md, "R1 headline (shared arm)")
    assert section.index("**Bottom line.**") < section.index("pre-declared primary is the paired shared-arm R1 Δ, "
                                                             "alignair (image) − igblast, per protocol")
    rows = _table(section)
    assert rows[0] == ["Protocol (donors)", "R1 Δ, alignair (image) − igblast (primary)",
                       "known_top1 Δ, alignair (image) − igblast", "V reads out by one call alone"]
    assert [row[0] for row in rows[1:]] == ["multiplex (W-97)", "race (SC-23)"]
    multiplex = rows[1]
    assert multiplex[1] == "19.6% → 17.5%: -2.1 pp [-5.2, +1.0], not significant; 143 paired (139 clonotypes)"
    assert multiplex[2] == "100.0% → 100.0%: +0.0 pp [-3.1, +3.1], not significant; 30 paired (30 clonotypes)"
    assert multiplex[3] == ("igblast 3 (1 documented known answers, 0 of them called wrong); alignair (image) 40 "
                            "(6 documented known answers, 4 of them called wrong)")
    assert rows[2][1].endswith("150 paired (147 clonotypes)")
    # the pool over both protocols comes second, said to depend on the protocols' mix
    pooled = next(line for line in section.splitlines() if line.startswith("Pooled over both protocols"))
    assert "depends on how many reads each protocol contributes" in pooled
    assert "R1 19.6% → 17.5%: -2.1 pp [-5.2, +1.0], not significant; 286 paired (278 clonotypes)" in pooled
    assert section.index("| race (SC-23) |") < section.index("Pooled over both protocols")
    # every CI is conditional on these donors; every other verdict is exploratory
    assert "conditional on these donors" in section and "exploratory" in section
    assert "Holm runs within each comparison" in section
    # the runs' own R1, with their different denominators, are in the detail, not on the summary page
    assert "22.5% (n=151)" not in section
    assert "22.5% (n=151)" in _section(md, "R1 per donor")


def test_the_scope_table_and_the_components_of_the_primary():
    summary = _summary()
    summary["pooled"]["comparisons"][0]["overall"]["real.v.unsupported"]["verdict"] = "b_better"
    md = real_report_markdown(summary)
    section = _section(md, "R1 headline (shared arm)")
    rows = _table(section[section.index("R1 Δ per scope"):])
    # a protocol of one donor is that donor's row
    assert [row[0] for row in rows[1:]] == ["W-97", "SC-23", "Pooled (2 donors)"]
    assert rows[1][1] == "19.6% → 17.5%: -2.1 pp [-5.2, +1.0], not significant; 143 paired (139 clonotypes)"
    assert rows[3][1] == ("19.6% → 17.5%: **-2.1 pp** [-5.2, +1.0], **alignair (image) better**; 286 paired (278 "
                          "clonotypes)")
    components = _table(section[section.index("Components"):])
    assert components[0] == ["Component", "multiplex (W-97)", "race (SC-23)", "Pooled (2 donors)"]
    assert [row[0] for row in components[1:]] == ["unsupported allele", "unsupported gene", "no call"]
    assert components[1][3] == "5.6% → 4.2%: -1.4 pp [-4.5, +1.7], not significant; 286 paired (278 clonotypes)"
    assert section.index("**Bottom line") < section.index("| W-97 |") < section.index("R1 (`real.v.unsupported`")
    assert section.index("R1 (`real.v.unsupported`") < section.index("Components")
    # protocols of two donors each get their own rows
    summary["pooled"]["protocols"]["multiplex"]["donors"] = ["W-97", "W-133"]
    rows = _table(_section(real_report_markdown(summary), "R1 headline (shared arm)").split("R1 Δ per scope")[1])
    assert [row[0] for row in rows[1:]] == ["multiplex (W-97, W-133)", "W-97", "SC-23", "Pooled (2 donors)"]


def test_one_call_alone_takes_reads_out_and_the_exclusions_say_by_what():
    section = _section(real_report_markdown(_summary()), "Exclusions")
    assert "a call into an excluded gene still takes the read out of R1 for both tools" in section
    assert ("Of those, the reads one call alone takes out, by reason: alignair (image) − igblast shared: igblast 3 "
            "(not_captured 3), alignair (image) 40 (not_captured 31, uncalled 9)") in section


def test_a_summary_without_protocol_pools_leads_with_its_last_scope():
    summary = _summary()
    del summary["pooled"]["protocols"]
    section = _section(real_report_markdown(summary), "R1 headline (shared arm)")
    rows = _table(section)
    assert rows[0][0] == "Scope" and [row[0] for row in rows[1:]] == ["Pooled (2 donors)"]
    assert "Pooled over both protocols" not in section


def test_cells_say_not_significant_too_few_clonotypes_or_no_variation_never_tie():
    summary = _summary()
    overall = summary["pooled"]["comparisons"][0]["overall"]
    overall["real.v.unsupported"].update(test="too_few_clusters", n_clusters=8)
    overall["real.v.no_call"].update(delta=0.0, ci95=[0.0, 0.0])  # both tools 0% on every paired read
    md = real_report_markdown(summary)
    section = _section(md, "R1 headline (shared arm)")
    pooled = _table(section[section.index("R1 Δ per scope"):])[3]
    assert pooled[1] == "19.6% → 17.5%: -2.1 pp [-5.2, +1.0], too few clonotypes; 286 paired (8 clonotypes)"
    assert pooled[2] == "19.6% → 17.5%: -2.1 pp [-5.2, +1.0], not significant; 286 paired (278 clonotypes)"
    no_call = next(row for row in _table(section[section.index("Components"):]) if row[0] == "no call")
    assert no_call[3] == "0.0% → 0.0%: +0.0 pp [+0.0, +0.0], no variation; 286 paired (278 clonotypes)"
    # the pool's line uses the tables' words
    assert "-2.1 pp [-5.2, +1.0], too few clonotypes; 286 paired (8 clonotypes)" in section
    assert ", tie;" not in md and ", tie (" not in md and "**tie" not in md


def test_the_statistics_note_is_one_line_under_the_headline():
    section = _section(real_report_markdown(_summary()), "R1 headline (shared arm)")
    notes = [line for line in section.splitlines() if line.startswith("Statistics:")]
    assert len(notes) == 1
    assert "clonotype" in notes[0] and "Holm within each comparison" in notes[0] and "not across" in notes[0]


def test_the_genotype_arm_leads_with_known_answers_not_r1():
    md = real_report_markdown(_summary())
    section = _section(md, "Genotype arm")
    assert "0 by construction" in section and "informational" in section
    rows = _table(section)
    assert rows[0] == ["Tool", "known_top1, shared → genotype", "known_gene_top1, shared → genotype",
                       "No V call, of all reads", "Calls in genes without truth, of all reads",
                       "R1, shared → genotype"]
    igblast, api = rows[1], rows[2]
    assert igblast[0] == "igblast" and api[0] == "alignair-api"
    assert igblast[1] == "100.0% → 100.0% (Δ +0.0 pp [-3.1, +3.1], 30 paired (30 clonotypes))"
    assert igblast[3] == "0.0% → 0.0%"
    assert igblast[4] == "61.4% → 65.4% (+4.0 pp)" and api[4] == "61.6% → 63.8% (+2.2 pp)"
    assert igblast[5] == "22.5% → 0.0% (0 by construction)"
    # the tools against each other on the genotype arm
    assert "alignair-api − igblast on the genotype arm: known_top1 -3.3 pp [-6.4, -0.2], not significant; 30 " \
           "paired (30 clonotypes)" in section


def test_sensitivity_variants_side_by_side_with_the_primary():
    section = _section(real_report_markdown(_summary()), "Sensitivity")
    rows = _table(section)
    assert rows[0] == ["Scope", "Variant", "alignair (image) − igblast", "alignair-api − igblast"]
    variants = [row[1] for row in rows[1:4]]
    assert variants == ["R1 primary", "Without SD3-only genes (104 reads called to them)",
                        "With Mikelov gene-support thresholds (481 V reads out)"]
    assert rows[2][2] == "-2.7 pp [-5.8, +0.4], not significant; 73 paired (70 clonotypes)"
    assert "21 genes" in section and "33 alleles" in section and ", ".join(SD3_GENES) in section
    assert "≥10 clonotypes" in section and "≥3 naive" in section and "Mikelov et al. 2024" in section


def test_exclusions_per_tool_and_reason_with_the_invariant():
    section = _section(real_report_markdown(_summary()), "Exclusions")
    assert "never depends on whether it is correct" in section
    for reason in EXCLUSIONS:
        assert f"`{reason}`" in section  # each class explained, one line each
    assert "Peres et al. 2023" in section and "Mikelov et al. 2024" in section
    rows = _table(section)
    assert rows[0] == ["Reason", "igblast shared", "alignair (image) shared", "alignair-api shared",
                       "igblast genotype", "alignair-api genotype"]
    by_reason = {row[0]: row[1:] for row in rows[1:]}
    assert by_reason["excluded:not_captured"] == ["307 (61.4%)", "308 (61.6%)", "308 (61.6%)", "327 (65.4%)",
                                                  "319 (63.8%)"]
    assert "excluded:unmapped" not in by_reason  # a reason no run has is left out of the table


def test_uncalled_genes_count_as_genes_without_truth_and_each_donors_truth_gaps_are_named():
    summary = _summary()
    for arm in ("shared", "genotype"):
        for tool in summary["donors"]["W-97"]["tools"][arm].values():
            classes = tool["v_classes"]
            classes["excluded:uncalled"] = 40
            classes["excluded:not_captured"] -= 40
    md = real_report_markdown(summary)
    exclusions = _section(md, "Exclusions")
    assert "`excluded:uncalled`" in exclusions and "calling gap" in exclusions
    rows = {row[0]: row[1:] for row in _table(exclusions)}
    assert rows["excluded:uncalled"][0] == "40 (8.0%)"
    # where the genomic truth stops, per donor: the genes the contigs hold and the genotype does not call, with what
    # they hold, the genes deleted on every capturing contig, and the carried genes with an unknown haplotype
    gaps = exclusions[exclusions.index("Where the genomic truth stops"):]
    gap_rows = {row[0]: row[1:] for row in _table(gaps)}
    assert gap_rows["W-97"] == ["IGHV4-30-2 (h0 another gene's sequence: IGHV4-28*01 at 100.0%)", "IGHV3-11",
                                "IGHV3-62"]
    # the flag counts uncalled with not captured: the share is unchanged
    flags = "\n".join(_bullets(_section(md, "Flags")))
    assert "W-97 igblast 61.4%, alignair (image) 61.6%, alignair-api 61.6%" in flags
    # the genotype arm counts both too
    assert _table(_section(md, "Genotype arm"))[1][4] == "61.4% → 65.4% (+4.0 pp)"


def test_assembly_failures_are_described_by_kind():
    section = _section(real_report_markdown(_summary()), "Preprocessing")
    assert "14,807 of 94,176 pairs (15.7%) failed to assemble; the line is 35%: within it" in section
    assert "mismatches 14,698" in section and "not significant 109" in section and "no overlap 0" in section
    rows = {row[0]: row[1:] for row in _table(section[section.index("| Pairs"):])}
    assert rows["passed: assembled length (nt)"][1:4] == ["386", "391", "394"]
    assert rows["failed: mates' combined length (nt)"][1] == "500"
    assert rows["failed: overlap error"][5] == "0.406"
    assert "long insert" not in section
    assert "R1 starts with a VH primer in 95.7%" in section  # the classifier's evidence


def test_alignair_assembly_counts_and_the_image_status():
    summary = _summary()
    summary["donors"]["W-97"]["runs"]["shared"]["alignair-api"]["assembly"].update(
        failed_assemblies=2, fail_reasons={"no J segment": 2})
    md = real_report_markdown(summary)
    section = _section(md, "AlignAIR's AIRR output")
    rows = {(row[0], row[1], row[2]): row for row in _table(section)}
    api = rows["W-97", "alignair-api", "shared"]
    assert api[3] == "ok" and api[4:8] == ["500", "462", "38", "2"] and "no J segment 2" in api[8]
    assert "failed-assembly gate passed" in rows["W-97", "alignair", "shared"][8]
    flags = [flag for flag in _bullets(_section(md, "Flags")) if "AIRR assembly failed" in flag]
    assert len(flags) == 1 and "alignair-api on W-97 shared 2 of 500" in flags[0]
    # the image's gate trips: its run failed, with the cause
    summary["donors"]["W-97"]["runs"]["shared"]["alignair"].update(
        status="failed", returncode=3, error="4.0% of records failed to assemble, over the 1% gate")
    section = _section(real_report_markdown(summary), "AlignAIR's AIRR output")
    image = {(row[0], row[1], row[2]): row for row in _table(section)}["W-97", "alignair", "shared"]
    assert image[3] == "failed (3)" and "over the 1% gate" in image[8]


def test_airr_row_validity_is_informational():
    section = _section(real_report_markdown(_summary()), "AlignAIR's AIRR output")
    assert "R1 reads only the calls" in section and "soft clip" in section
    validity = section[section.index("AIRR rows the assay's reader flags"):]
    rows = {row[0]: row[1:] for row in _table(validity)}
    assert rows["W-97"] == ["1 of 500", "462 of 500", "462 of 500", "1 of 500", "462 of 500"]


def test_data_limit_flags():
    summary = _summary()
    summary["donors"]["SC-23"]["oracle_self_check"].update(scored=0, supported=None, passed=False,
                                                           status="unchecked")
    summary["oracle_unchecked"] = ["SC-23"]
    evidence = {"pairs": 2000, "race_linker": 0.41, "vh_primer": 0.52, "c_primer": 0.5,
                "reason": "neither protocol: linker in 41.0%, VH primer in 52.0% of the pairs"}
    summary["donors"]["W-133"] = {
        "donor": "W-133", "split": "dev", "run": "SRR23654572", "sample": "P25_I93_S1", "status": "stopped",
        "stopped": {"step": "preprocess", "reason": "W-133: the run is refused: neither protocol",
                    "evidence": evidence},
        "steps": {}, "fetch": {"quality": "full", "source": "odp"},
        "preprocess": {"status": "refused", "protocol": "unknown",
                       "classifier": {"protocol": "unknown", "evidence": evidence}}}
    summary["stopped_donors"] = [{"donor": "W-133", "step": "preprocess",
                                  "reason": "W-133: the run is refused: neither protocol"}]
    drift = ("2 predicted rows match no read (e.g. ghost_1, ghost_2): the run's ids drifted from the reads it was "
             "given, so it is not scored")
    summary["donors"]["W-97"]["runs"]["shared"]["alignair"].update(status="failed", error=drift)
    summary["failed_runs"] = [{"donor": "W-97", "tool": "alignair", "arm": "shared", "error": drift, "note": None}]
    md = real_report_markdown(summary)
    flags = _bullets(_section(md, "Flags"))
    assert len(flags) <= MAX_FLAGS
    text = "\n".join(flags)
    assert "W-133 stopped at preprocess" in text and "neither protocol: linker in 41.0%" in text
    assert "alignair on W-97 shared" in text and "match no read" in text
    assert "unchecked" in text and "SC-23" in text
    uncaptured = [flag for flag in flags if "genes without truth" in flag]
    assert len(uncaptured) == 1 and "W-97 igblast 61.4%, alignair (image) 61.6%, alignair-api 61.6%" in uncaptured[0]
    failures = _section(md, "Failed runs and refusals")
    assert "W-133" in failures and "race_linker 0.41" in failures and "ghost_1" in failures
    assert "W-133" in md.split("\n## ")[0]  # a stopped donor is still named in the header


def test_flags_are_at_most_eight():
    summary = _summary()
    for donor in ("W-97", "SC-23"):
        block = summary["donors"][donor]
        block["fetch"]["quality"] = "lite"
        block["truth"].update(known=10, known_share=0.02, low_known_yield=True)
        block["oracle_self_check"]["status"] = "failed"
        block["preprocess"]["assembly_acceptance"].update(failure_rate=0.4, within_line=False, override=True)
        block["runs"]["shared"]["alignair-api"]["assembly"]["failed_assemblies"] = 3
    summary["oracle_failures"] = ["W-97", "SC-23"]
    summary["oracle_unchecked"] = ["W-97"]
    summary["failed_runs"] = [{"donor": "W-97", "tool": "igblast", "arm": "genotype", "error": "boom", "note": None}]
    summary["stopped_donors"] = [{"donor": "W-133", "step": "fetch", "reason": "md5 differs"}]
    summary["pooled"]["comparisons"][0]["overall"]["real.v.unsupported"]["test"] = "too_few_clusters"
    flags = _bullets(_section(real_report_markdown(summary), "Flags"))
    assert len(flags) == MAX_FLAGS
    assert "self-check failed" in flags[0]  # the most severe first
    assert flags[-1].startswith("- And ") and "more" in flags[-1]


def test_stratification_comes_after_the_headline():
    md = real_report_markdown(_summary())
    assert md.index("\n## Stratification: protocol × isotype\n") > md.index("\n## R1 headline (shared arm)\n")
    section = _section(md, "Stratification: protocol × isotype")
    labels = {row[1] for row in _table(section)[1:]}
    assert {"multiplex/IGHG", "multiplex/IGHM", "race/IGHG"} <= labels


def test_the_primary_under_the_sl15_and_junction_length_clusterings():
    summary = _summary()
    md = real_report_markdown(summary)
    sensitivity = _section(md, "Sensitivity")
    part = sensitivity[sensitivity.index("Clustering"):]
    assert "candidate primary for P3b" in part and "stress test" in part
    rows = _table(part)
    assert rows[0] == ["Scope", "Metric", "exact junction (primary)", "SL15", "donor × junction length (stress)"]
    assert [row[:2] for row in rows[1:3]] == [["multiplex (W-97)", "R1"], ["multiplex (W-97)", "known_top1"]]
    assert rows[1][3] == "-2.1 pp [-5.2, +1.0], not significant; 143 paired (119 clonotypes)"
    assert rows[1][4].endswith("143 paired (31 clonotypes)")
    # the bottom line says whether SL15 changes a verdict of the primary or of known_top1
    headline = _section(md, "R1 headline (shared arm)")
    assert "Clustered by SL15 instead of exact junctions, every verdict above holds" in headline
    race = summary["pooled"]["protocols"]["race"]["comparisons"][0]
    race["overall"]["real.v.unsupported"]["verdict"] = "a_better"
    race["clusterings"]["exact"]["comparison"]["real.v.unsupported"]["verdict"] = "a_better"
    headline = _section(real_report_markdown(summary), "R1 headline (shared arm)")
    assert ("Clustered by SL15 instead of exact junctions, a verdict changes: race (SC-23) R1 igblast better → not "
            "significant") in headline


def test_the_tie_aware_metrics_are_shown_as_paired_means():
    md = real_report_markdown(_summary())
    detail = _section(md, "R1 per donor")
    w97 = _table(detail)
    row = next(row for row in w97 if row[0].startswith("unsupported, observably"))
    assert row[1:4] == ["—", "—", "—"]  # paired means only
    assert row[4] == "12.1% → 9.8% (n=143)"
    known = _section(md, "Known answers")
    paired = _table(known[known.index("Paired"):])
    assert paired[0] == ["Scope", "Pair", "known_top1", "known_gene_top1",
                         "known_top1, observably (informational: paired means)"]
    first = paired[1]
    assert first[:2] == ["W-97", "alignair (image) − igblast, shared"]
    assert first[2] == "+0.0 pp [-3.1, +3.1], not significant; 30 paired (30 clonotypes)"
    assert first[4] == "100.0% → 96.7% (n=30)"
    assert "observed V" in known


def test_known_answers_and_novel_reads_per_tool_and_arm():
    md = real_report_markdown(_summary())
    known = _section(md, "Known answers")
    assert "150 (30.0%)" in known  # SC-23's yield
    novel = _section(md, "Novel-allele reads")
    rows = {row[0]: row[1:] for row in _table(novel)}
    assert rows["igblast shared"] == ["8", "75.0%", "known gene 6, IGHV3-23 2"]


def test_truth_self_check_and_d_j():
    md = real_report_markdown(_summary())
    rows = {row[0]: row[1:] for row in _table(_section(md, "Truth self-check"))}
    assert rows["W-97"][-1] == "passed" and rows["SC-23"][0] == "150"
    assert "No donor has D or J truth" in _section(md, "D and J (informational)")
    # a donor with D truth: each run's D classes and its informational real.d.unsupported
    summary = _summary()
    donor = summary["donors"]["SC-23"]
    donor["genotype"]["d_truth"] = True
    igblast = donor["tools"]["shared"]["igblast"]
    igblast["summary"]["real.d.unsupported"] = _cell(0.25, 200)
    igblast["d_classes"] = {"supported": 150, "unsupported_allele": 50, "excluded:not_captured": 300}
    rows = {(row[0], row[1], row[2]): row[3:] for row in _table(_section(real_report_markdown(summary),
                                                                         "D and J (informational)"))}
    assert rows["SC-23", "igblast", "shared"][:2] == ["25.0% (n=200)", "supported 150, unsupported_allele 50, "
                                                                        "excluded:not_captured 300"]
    assert rows["SC-23", "igblast", "shared"][2:] == ["—", "—"]  # no J truth
    assert ("W-97", "igblast", "shared") not in rows


def test_a_pair_not_compared_or_without_paired_reads():
    summary = _summary()
    donor = summary["donors"]["W-97"]
    donor["comparisons"][0] = {"a": "igblast", "b": "alignair", "arm": "shared", "status": "not compared",
                               "reason": "alignair on the shared arm did not run successfully"}
    del donor["tools"]["shared"]["alignair"]
    for name in ("real.v.unsupported", "real.v.unsupported_allele", "real.v.unsupported_gene", "real.v.no_call"):
        del donor["comparisons"][1]["overall"][name]  # every read excluded by either call
    original = copy.deepcopy(summary)
    md = real_report_markdown(summary)
    assert summary == original  # the input is not changed
    headline = _section(md, "R1 headline (shared arm)")
    w97 = _table(headline[headline.index("R1 Δ per scope"):])[1]
    assert w97[1] == "not compared: alignair on the shared arm did not run successfully"
    assert w97[2] == "no paired read (357 of 500 V-excluded by either call)"
    detail = _section(md, "R1 per donor")
    assert "no paired read" in detail
    assert _table(detail)[1][2] == "—"  # the run's own R1, which did not run


def test_a_run_stopped_after_preprocessing_renders():
    summary = _summary()
    summary["config"]["stop_after"] = "preprocess"
    for block in summary["donors"].values():
        for key in ("truth", "runs", "oracle_self_check", "tools", "comparisons", "arm_deltas"):
            block.pop(key)
        block["stopped_after"] = "preprocess"
    summary["pooled"] = {"donors": [], "tools": {"shared": {}, "genotype": {}}, "comparisons": [], "arm_deltas": []}
    summary["sensitivity"] = None
    md = real_report_markdown(summary)
    assert [line[3:] for line in md.splitlines() if line.startswith("## ")] == [*SUMMARY_SECTIONS,
                                                                                *DETAIL_SECTIONS]
    assert "Nothing was scored: the run stopped after preprocessing" in _section(md, "R1 headline (shared arm)")
    assert "R1 starts with a VH primer in 95.7%" in _section(md, "Preprocessing")


def test_missing_values_render_as_a_dash():
    summary = _summary()
    summary["donors"]["W-97"]["tools"]["shared"]["igblast"]["summary"]["real.v.unsupported"] = {
        "mean": float("nan"), "n": 0, "n_applicable": 0, "coverage": 0.0}
    rows = _table(_section(real_report_markdown(summary), "R1 per donor"))
    assert rows[1][1] == "— (n=0)"
