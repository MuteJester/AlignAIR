# AlignAIR — Unified, Data-Driven Model Refactor

**Date:** 2026-07-07
**Goal:** Replace the SingleChain/MultiChain split and the scattered `has_d` gene-branching with one
`AlignAIR` model assembled from data (`GeneSpec` + `GeneBranch` + `MetaHead`). Pure implementation
refactor — identical layers, ops, and math; **proven** behavior-preserving.

## Motivation

`has_d: bool` was a weak stand-in for "which genes exist," so the gene list was re-derived in ~15
places with two different orderings (`v,j,d` in the model/loss vs `v,d,j` in predict) — a latent
trap. Each gene's config (kernels, counts, latent) was split across four inline dicts in `__init__`,
and a single gene's pipeline was smeared across six `ModuleDict`s and three `forward` loops. Multi vs
single chain was a subclass + an `_extra_meta_heads` seam.

## Design

**One source of truth — `GeneSpec`** (`config/alignair_config.py`): `name, allele_count, seg_kernels,
cls_kernels, latent_size, short_d_penalty`. `AlignAIRConfig.gene_specs` returns the canonical ordered
`(V, [D], J)`. A gene exists iff it has a spec — `has_d` never gets re-expanded into a gene list.

**One block per gene — `GeneBranch`** (`models/gene_branch.py`): a gene's whole pipeline as a
cohesive, testable unit — `segment(emb) -> (start_logits, end_logits, start_exp, end_exp)` and
`classify(emb, start_exp, end_exp) -> allele_probs`.

**One block per meta prediction — `MetaHead`**: optional `Linear+act+dropout` mid, a `Linear` head,
optional output activation, optional post-step weight clamp. Covers mutation / indel / productivity /
chain_type uniformly (replaces four bespoke head pairs).

**One model — `AlignAIR`** (`models/alignair.py`): `embedding -> orientation detect/correct/re-embed
-> meta_tower -> {GeneBranch per gene} -> {MetaHead per meta task} -> per-gene classify`. Single vs
multi is data: `cfg.gene_specs` (which genes) and `cfg.num_chain_types` (chain_type head present iff
> 1). `SingleChainAlignAIR` / `MultiChainAlignAIR` remain as thin aliases.

**Dataconfig-driven construction:** `AlignAIRConfig.from_dataconfigs(*dataconfigs)` derives union
allele counts, `has_d`, and `num_chain_types` (distinct `dc.metadata.chain_type`). One dataconfig ->
single-chain; several -> multi-chain. No hand-typed allele counts anywhere.

The short-D span penalty moved from `if cfg.has_d` in the shared loss to `spec.short_d_penalty`, so
the loss is gene-agnostic and iterates `cfg.gene_specs`.

## Equivalence guarantee

A golden snapshot of the pre-refactor models (single / multi / light: outputs + weights on fixed
inputs) was captured, its weights remapped to the new parameter names, loaded into the new `AlignAIR`
(strict), and re-run. Result: **max|Δ| = 0.00e+00 on every output for all three cases** — the refactor
changes names and structure only, not the function computed. The legacy→unified state_dict key map
(`seg_towers.g.* -> branches.g.seg_tower.*`, `mutation_rate_mid/head -> meta_heads.mutation_rate.
mid/head`, etc.) exists as a migration utility but is not needed for production (retraining fresh).

## Blast radius

Model-internal only. Two couplings updated: `test_single_chain.py` clamp test (now reaches
`meta_heads["mutation_rate"].head`) and the `xray` hook-target heuristic (`.head` suffix). External
consumers read `cfg.has_d` / allele counts / `cfg.latent(gene)`, all preserved. Full suite green.

## Non-goals

Multi-chain *training* (gym stream mixing dataconfigs + emitting `chain_type` targets) remains a
follow-on; the model + `from_dataconfigs` support it, but the trainer still streams one dataconfig.
