# Stage 2 — Result

**DECISION: `GO_PRODUCTION_REDISTRIBUTABLE`**  (2026-07-27)

An exact production profile path is resolved for every required profile, the bank builds deterministically,
and C0 produces valid decisions on the burned Gate-3 development FIT split.

## Profile source resolution (Task 2a)
All 10 required profiles resolve to a redistributable production source:
- **CC0 (no attribution):** human_igh / human_igk / human_igl (OGRDB/AIRR-C), igsf_vset (Pfam PF07686).
- **CC BY-4.0 (attribution required):** tra/trb/trg/trd, murine_vh, camelid (vhh) — IMGT/GENE-DB, cleared
  by the owner (terms page verbatim "CC BY-4.0"; relicensing 2026-07-01). Provenance archived in
  `license_evidence.json` (raw HTML private + SHA-256 + Wayback).

**Redistribution statement:** raw IMGT files are **NOT** shipped. Only DERIVED HMM profiles are
redistributed — CC BY-4.0 with attribution + modification notice for IMGT-derived profiles, CC0 for
OGRDB/Pfam. AlignAIR downloads germline DATA and builds profiles locally with pyhmmer (vendored HMMER, not
an IMGT tool); it does not invoke IMGT web tools. Attribution/modification notice go in the bundle NOTICE,
model card, and profile manifest, without implying IMGT endorsement.

## Production bank (Task 2.2/2.3 machinery; Step-6 real build)
Deterministic pyhmmer build (Builder seed=42; DATE/COM neutralized). Functional (F) complete alleles only;
translated from the annotated coding frame; internal stops / partials / invalid residues / unresolved
frames rejected; identical PROTEIN sequences deduplicated before MSA. V uses IMGT unique-numbering gapped
alignment; OGRDB J (no frame field) uses reading-frame resolution + [WF]G.G motif-anchor alignment. Camelid
profiles are labelled `camelid_igh_v/j` (non-human/camelid evidence, not biologically definitive "VHH").
V and J are paired only within the same species/locus group.

Bank checksum: `e5c8e7352e2e17920352475b40b1c38abe72ab2dafa7785e5cba05d14eb26f55`  ·  HMMER: pyhmmer 0.12.1 / HMMER 3.4

| group | region | source | license | input | accepted | protein-distinct | hmm_sha256 |
|---|---|---|---|---|---|---|---|
| human_igh | j | ogrdb_human_igh | CC0-1.0 | 7 | 7 | 7 | `0daf1525451e` |
| human_igh | v | ogrdb_human_igh | CC0-1.0 | 209 | 203 | 148 | `266065d8f3ae` |
| human_igk | j | ogrdb_human_igk | CC0-1.0 | 7 | 7 | 7 | `de9bffd1e3cf` |
| human_igk | v | ogrdb_human_igk | CC0-1.0 | 82 | 79 | 64 | `e4e84fd3bbd2` |
| human_igl | j | ogrdb_human_igl | CC0-1.0 | 10 | 10 | 8 | `272ceea96f4f` |
| human_igl | v | ogrdb_human_igl | CC0-1.0 | 88 | 83 | 76 | `c467b9b9e21f` |
| igsf_vset | domain | pfam_pf07686 | CC0 | - | - | - | `50f73e528488` |
| murine_vh | j | imgt_gene_db | CC-BY-4.0 | 8 | 8 | 6 | `80d7e5e47ed5` |
| murine_vh | v | imgt_gene_db | CC-BY-4.0 | 415 | 405 | 377 | `c5d3ef2a63bf` |
| tra | j | imgt_gene_db | CC-BY-4.0 | 62 | 62 | 58 | `ea4ee7afbb6d` |
| tra | v | imgt_gene_db | CC-BY-4.0 | 94 | 94 | 72 | `a600b6155163` |
| trb | j | imgt_gene_db | CC-BY-4.0 | 14 | 14 | 13 | `0fb5ba90e1d2` |
| trb | v | imgt_gene_db | CC-BY-4.0 | 103 | 103 | 85 | `abfc9ad67712` |
| trd | j | imgt_gene_db | CC-BY-4.0 | 5 | 5 | 4 | `5ccfdbc3d741` |
| trd | v | imgt_gene_db | CC-BY-4.0 | 7 | 7 | 7 | `ff0e111c176b` |
| trg | j | imgt_gene_db | CC-BY-4.0 | 7 | 7 | 5 | `a6c1670c7c63` |
| trg | v | imgt_gene_db | CC-BY-4.0 | 14 | 14 | 10 | `0432b318af94` |
| vhh | j | imgt_gene_db | CC-BY-4.0 | 6 | 6 | 6 | `d1f33031d01a` |
| vhh | v | imgt_gene_db | CC-BY-4.0 | 73 | 73 | 73 | `42ef416da7dd` |
| igsf_vset | domain | pfam_pf07686 | CC0 | (Pfam HMM used directly) | - | - | `50f73e528488` |

Raw germline references, MSAs, and HMM binaries are **private** (git-ignored); only the manifest + checksums
are committed. Full per-profile input/accepted/rejected-by-reason/protein-distinct/MSA/HMM counts+hashes are
in `profile_bank_manifest.json`.

## C0 development diagnostic (Task 2.4) — NOT a release PASS
Run on the burned Gate-3 **FIT split only** (240 records, all expected `supported`). The sealed
test (`locked_test.jsonl`) is never opened; fresh selection / blind-LOCO are never read.

- C0 state counts: `{"insufficient_evidence": 11, "supported": 133, "unknown_ood": 84, "unsupported_locus": 12}`
- supported-recall on fit: **0.5542**
- Main confusions: weak human-IGH margin and camelid/mouse germline competition (humanized / human-framework
  VH are close to human-homologous camelid VHH). This is expected for an **HMM-only baseline** — C0 is the
  floor; the contextual encoder (C1–C4) and the fresh-selected threshold grid-point (Stage 6) lift it.

C0's rule structure, score directions, and finite threshold grid are **frozen on burned-dev**; the single
production grid point is chosen on FRESH selection in Stage 6 — never here. This diagnostic is a development
signal, not fresh evidence and not a release PASS.

## Stage-2 GO criteria
- [x] Exact production profile path resolved before bank construction (2a GO, all CC0/CC BY-4.0).
- [x] Bank builds deterministically (bank_checksum reproducible).
- [x] C0 produces valid decisions on burned-dev fixtures (all four states produced).
