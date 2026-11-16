"""The novel-allele screen: does training with synthetic novel alleles in the banks (infra Phase 2) call held-out novel
alleles better, at no cost elsewhere?

The candidate is chip-v1 trained by the novel-allele recipe (``recipe.NOVEL_ALLELES``) from a novel record cache,
screened (``screen.run_screen``) under the protocol of the default cache, against the registered baseline screen. It
is then scored on the held-out S7 donors (``HELDOUT``: their reads come from V alleles no reference holds) in two
banks (``heldout_scores``): ``extended``, the model's reference plus every donor's novel alleles (``heldout_reference``:
one row per novel sequence), and ``genotype``, each donor's carried alleles. The baseline screens of init seeds 0, 1
and 2 of the protocol are scored the same way. Held-out sequences a training epoch also drew (``training_overlap``)
are scored apart.

``novel_verdict`` is the owner's pass rule (2026-11-07): ``pass`` when the extended bank's novel-read V top-1 exact is
at least ``MIN_NOVEL_GAIN`` above the baseline seeds' mean and above every seed, its known reads and both of the
genotype bank's read classes lose at most ``MAX_LOSS``, and no gated dev metric (``GATED``) is worse, mixed or
regressing; ``uncalibrated`` without 3 baseline seeds or with an uncalibrated dev decision; ``fail`` otherwise."""
from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from alignair_lab.baselines import baseline_screens
from alignair_lab.headline import HEADLINE, SECONDARY
from alignair_lab.noise import MIXED, UNCALIBRATED, WORSE, ZERO_SPREAD
from alignair_lab.recipe import BASELINE_ARCHITECTURE, NOVEL_ALLELES
from alignair_lab.screen import DEFAULT_STEPS, run_screen

HELDOUT = Path("/media/thomas/NVMe_Data/alignair_arch/s7_full/suites/S7")
MIN_NOVEL_GAIN = 0.02
MAX_LOSS = 0.005
MIN_BASELINE_SEEDS = 3
BASELINE_SEEDS = (0, 1, 2)
GATED = tuple(metric for metric, _ in (*HEADLINE, *SECONDARY))
CONDITIONS = ("extended", "genotype")
# The candidate trains on the novel cache's records, not the baseline seeds' (the novel cache draws its own reads), and
# the band and the seeds measure init-seed noise on fixed batches: a dev clause, or part of a gain, may be the draw's.
CAVEAT = ("the null band and the baseline seeds measure init-seed noise on fixed batches; the candidate trains on "
          "another data draw (the novel cache's own records), whose effect they leave out")


def heldout_reference(reference, donors) -> tuple:
    """``reference`` extended with the donors' novel V alleles, one row per novel sequence, and the row of each
    ``(subject, name)``: a name every donor gives one sequence keeps it; a name given several sequences gets a row
    per sequence named ``<name>_<subject>`` after the first donor, in file order, that carries it."""
    from alignair.reference.reference_set import ReferenceSet
    sequences: dict[str, list[tuple[str, str]]] = {}
    for donor in donors:
        for name, allele in donor.novel.items():
            held = sequences.setdefault(name, [])
            if allele.sequence not in [sequence for sequence, _ in held]:
                held.append((allele.sequence, donor.subject_id))
    rows = {}
    for name, held in sequences.items():
        for sequence, subject in held:
            rows[name if len(held) == 1 else f"{name}_{subject}"] = sequence
    mapping = {}
    for donor in donors:
        for name, allele in donor.novel.items():
            held = sequences[name]
            first = next(subject for sequence, subject in held if sequence == allele.sequence)
            mapping[(donor.subject_id, name)] = name if len(held) == 1 else f"{name}_{first}"
    return reference.extend(ReferenceSet.from_genotype({"V": rows, "J": {}})), mapping


def _empty() -> dict:
    return {"novel_reads": 0, "novel_exact": 0, "novel_called_base": 0, "known_reads": 0, "known_exact": 0,
            "by_snps": {}, "excluded": {"reads": 0, "exact": 0}}


def heldout_scores(checkpoint, suite, *, device: str, batch_size: int, exclude=frozenset()) -> dict:
    """The held-out reads' V top-1 exact in the ``extended`` and ``genotype`` banks (see the module docstring): a read
    is exact when the first allele of its ``v_call`` is its truth's row. Novel reads of a sequence in ``exclude``
    (one training also drew) go under ``excluded``."""
    from alignair import Aligner
    from alignair_assay.core.cases import read_cases
    from alignair_assay.suites.donors import donors_from_json
    suite = Path(suite)
    cases = read_cases(suite / "cases.jsonl.gz")
    donors = donors_from_json(json.loads((suite / "donors.json").read_text(encoding="utf-8")))
    own = Aligner.from_pretrained(str(checkpoint), device=device).reference
    extended, mapping = heldout_reference(own, donors)
    aligner = Aligner.from_pretrained(str(checkpoint), reference=extended, device=device)
    reference = aligner.reference
    sequences = [case["sequence"] for case in cases]
    calls = {"extended": [record["v_call"] for record in aligner.predict(sequences, batch_size=batch_size).records],
             "genotype": [None] * len(cases)}
    by_donor: dict[str, list[int]] = {}
    for i, case in enumerate(cases):
        by_donor.setdefault(case["groups"]["subject"], []).append(i)
    for donor in donors:
        reads = by_donor.get(donor.subject_id, [])
        if not reads:
            continue
        genotype: dict[str, set[str]] = {}
        for names in donor.alleles.values():
            for name in names:
                row = mapping.get((donor.subject_id, name), name)
                letter = next((g for g in reference.genes if row in reference.gene(g).index), None)
                if letter is not None:
                    genotype.setdefault(letter, set()).add(row)
        records = aligner.with_genotype(genotype).predict([sequences[i] for i in reads], batch_size=batch_size).records
        for i, record in zip(reads, records):
            calls["genotype"][i] = record["v_call"]
    novel_of = {(donor.subject_id, name): allele for donor in donors for name, allele in donor.novel.items()}
    out = {"rows": len(extended.gene("V")) - len(own.gene("V"))}
    for condition in CONDITIONS:
        found = _empty()
        for case, call in zip(cases, calls[condition]):
            subject, truth = case["groups"]["subject"], case["record"]["truth_v_call"]
            first = (call or "").split(",")[0]
            exact = int(first == mapping.get((subject, truth), truth))
            snps = case["labels"]["novel_snps"]
            if snps == "none":
                found["known_reads"] += 1
                found["known_exact"] += exact
                continue
            allele = novel_of[(subject, truth)]
            if allele.sequence in exclude:
                found["excluded"]["reads"] += 1
                found["excluded"]["exact"] += exact
                continue
            found["novel_reads"] += 1
            found["novel_exact"] += exact
            found["novel_called_base"] += int(first == allele.base)
            counts = found["by_snps"].setdefault(snps, [0, 0])
            counts[0] += 1
            counts[1] += exact
        out[condition] = found
    return out


def training_overlap(cache, suite) -> frozenset:
    """The held-out novel sequences any epoch of any component of the novel cache ``cache`` drew (``epoch_alleles``
    rerun for the epochs its records span; no simulation)."""
    from alignair.genairr import resolve_dataconfig
    from alignair.reference.reference_set import ReferenceSet
    from alignair.train.records.novel import NovelAlleles, epoch_alleles
    from alignair_assay.suites.donors import donors_from_json
    manifest = json.loads((Path(cache) / "manifest.json").read_text(encoding="utf-8"))
    donors = donors_from_json(json.loads((Path(suite) / "donors.json").read_text(encoding="utf-8")))
    heldout = {allele.sequence for donor in donors for allele in donor.novel.values()}
    references, drawn = {}, set()
    for component in manifest["components"]:
        definition = (component.get("parameters") or {}).get("novel_alleles")
        if definition is None:
            continue
        spec = NovelAlleles.from_dict(definition)
        name = component["reference"]
        if name not in references:
            references[name] = ReferenceSet.from_dataconfigs(resolve_dataconfig(name))
        seed, records = int(component["seed"]), int(component["records"])
        for epoch in range(seed // spec.epoch_records, (seed + max(records, 1) - 1) // spec.epoch_records + 1):
            drawn.update(allele.sequence for allele in epoch_alleles(references[name], spec, epoch))
    return frozenset(drawn & heldout)


def _rate(scores: Mapping, condition: str, kind: str) -> float | None:
    block = scores[condition]
    return block[f"{kind}_exact"] / block[f"{kind}_reads"] if block[f"{kind}_reads"] else None


def _change(candidate: Mapping, baselines: Sequence[Mapping], condition: str, kind: str) -> float | None:
    values = [value for value in (_rate(b, condition, kind) for b in baselines) if value is not None]
    mine = _rate(candidate, condition, kind)
    return None if mine is None or not values else mine - statistics.fmean(values)


def novel_verdict(candidate: Mapping, baselines: Sequence[Mapping], decisions: Mapping) -> dict:
    """The pass rule (see the module docstring) on the candidate's and the baseline seeds' held-out scores and the
    screen's dev ``decisions``."""
    seeds = [value for value in (_rate(b, "extended", "novel") for b in baselines) if value is not None]
    novel = _rate(candidate, "extended", "novel")
    gain = None if novel is None or not seeds else novel - statistics.fmean(seeds)
    margin = None if novel is None or not seeds else novel - max(seeds)
    changes = {"known_change": _change(candidate, baselines, "extended", "known"),
               "genotype_novel_change": _change(candidate, baselines, "genotype", "novel"),
               "genotype_known_change": _change(candidate, baselines, "genotype", "known")}
    found = {"novel_gain": gain, "novel_margin": margin, **changes, "caveat": CAVEAT}
    unset = [metric for metric in GATED
             if (decisions.get(metric) or {}).get("decision") in (None, UNCALIBRATED, ZERO_SPREAD)]
    if len(seeds) < MIN_BASELINE_SEEDS or unset or novel is None:
        reasons = []
        if len(seeds) < MIN_BASELINE_SEEDS:
            reasons.append(f"{len(seeds)} baseline seeds scored ({MIN_BASELINE_SEEDS} needed)")
        if unset:
            reasons.append(f"dev decisions uncalibrated: {', '.join(unset)}")
        if novel is None:
            reasons.append("no novel reads scored")
        return {"verdict": "uncalibrated", "reasons": reasons, **found}
    reasons = []
    if gain < MIN_NOVEL_GAIN:
        reasons.append(f"novel gain {100 * gain:+.1f} pp is below +{100 * MIN_NOVEL_GAIN:.1f} pp")
    if margin <= 0:
        reasons.append(f"novel exact {100 * novel:.1f}% is within the baseline seeds (best {100 * max(seeds):.1f}%)")
    for label, key in (("known reads", "known_change"), ("genotype bank novel reads", "genotype_novel_change"),
                       ("genotype bank known reads", "genotype_known_change")):
        change = changes[key]
        if change is not None and change < -MAX_LOSS:
            reasons.append(f"{label} {100 * change:+.1f} pp (at most -{100 * MAX_LOSS:.1f} pp)")
    worse = [metric for metric in GATED if decisions[metric]["decision"] in (WORSE, MIXED)]
    if worse:
        reasons.append(f"dev metrics worse or mixed: {', '.join(worse)}")
    regressions = [f"{metric} ({', '.join(decisions[metric]['regressions'])})" for metric in GATED
                   if decisions[metric].get("regressions")]
    if regressions:
        reasons.append(f"dev regressions: {', '.join(regressions)}")
    return {"verdict": "fail" if reasons else "pass", "reasons": reasons, **found}


def _block(novel: Mapping) -> list[str]:
    """The held-out lines appended to a novel screen's headline, ending in its verdict."""
    lines = ["", f"Held-out novel alleles ({novel['heldout']}; {len(novel['overlap'])} also drawn in training, scored "
                 "apart):"]
    for who, scores in (("candidate", novel["scores"]["candidate"]),
                        *((f"baseline init {seed}", s) for seed, s in novel["scores"]["baselines"].items())):
        parts = []
        for condition in CONDITIONS:
            n, k = _rate(scores, condition, "novel"), _rate(scores, condition, "known")
            parts.append(f"{condition}: novel {'n/a' if n is None else f'{100 * n:.1f}%'}, "
                         f"known {'n/a' if k is None else f'{100 * k:.1f}%'}")
        lines.append(f"  {who}: " + "; ".join(parts))
    tail = f": {'; '.join(novel['reasons'])}" if novel["reasons"] else ""
    lines.append(f"Note: {novel.get('caveat', CAVEAT)}.")
    lines.append(f"Verdict: {novel['verdict']}{tail}")
    return lines


def run_novel_screen(name: str, *, steps: int = DEFAULT_STEPS, cache, novel_cache, heldout=HELDOUT, schedule: str,
                     init_seed: int | None = 0, suites: Sequence[str] | None = None, max_cases: int | None = None,
                     device: str = "cuda", data_workers: int, eval_batch_size: int, dev_build, out_root,
                     model_options: Mapping | None = None, baseline_options: Mapping | None = None,
                     echo: Callable[[str], None] | None = None) -> dict:
    """Screen chip-v1 trained from ``novel_cache`` by the novel-allele recipe, score it and the protocol's baseline
    seeds on ``heldout``, and write ``<screen>/novel.json`` and the verdict on its headline (see the module
    docstring). ValueError without a novel cache, before anything runs."""
    if novel_cache is None:
        raise ValueError("a novel-allele screen trains from a novel record cache: give --novel-cache "
                         "(python -m alignair_lab cache --novel)")
    if not (Path(heldout) / "donors.json").is_file():
        raise FileNotFoundError(f"{heldout} holds no held-out suite (donors.json)")
    record = run_screen(name, steps=steps, architecture=BASELINE_ARCHITECTURE, model_options=model_options,
                        suites=suites, max_cases=max_cases, device=device, data_workers=data_workers,
                        eval_batch_size=eval_batch_size, cache=cache, dev_build=dev_build, out_root=out_root,
                        schedule=schedule, init_seed=init_seed, baseline_options=baseline_options,
                        novel_alleles=NOVEL_ALLELES, training_cache=novel_cache, echo=echo)
    screen = Path(out_root) / name
    overlap = training_overlap(novel_cache, heldout)
    options = dict(device=device, batch_size=eval_batch_size, exclude=overlap)
    candidate = heldout_scores(record["candidate"]["model_path"], heldout, **options)
    baselines = {}
    for seed in BASELINE_SEEDS:
        done = [base for _, base in baseline_screens(Path(out_root), record["protocol"], baseline_options or {}, seed)
                if base.get("status") == "done" and base.get("candidate")]
        if done:
            baselines[seed] = heldout_scores(done[0]["candidate"]["model_path"], heldout, **options)
    verdict = novel_verdict(candidate, list(baselines.values()), record.get("decisions") or {})
    novel = {"heldout": str(heldout), "novel_cache": str(novel_cache), "overlap": sorted(overlap),
             "scores": {"candidate": candidate, "baselines": {str(seed): s for seed, s in baselines.items()}},
             **verdict}
    (screen / "novel.json").write_text(json.dumps(novel, indent=1) + "\n", encoding="utf-8")
    lines = _block(novel)
    with open(screen / "headline.txt", "a", encoding="utf-8") as headline:
        headline.write("\n".join(lines) + "\n")
    for line in lines:
        (echo or print)(line)
    return {**record, "novel": novel}
