# AIRR field coverage

AlignAIR emits an AIRR-C rearrangement TSV. This page is the honest contract for
*which* fields it populates, so downstream tools (Change-O / Alakazam, Scirpy,
nf-core/airrflow) know what to expect. The principle throughout is **honest
absence**: a field is left empty when AlignAIR cannot determine it reliably, rather
than guessed.

## Emitted by AlignAIR

| Field(s) | Notes |
|----------|-------|
| `sequence_id`, `sequence`, `rev_comp` | `sequence` is the canonical (forward) orientation the coordinates refer to; `rev_comp=T` when the input read was reoriented. |
| `locus` | inferred from the reference / model. |
| `v_call`, `d_call`, `j_call` | top call per gene. |
| `v_call_set`, `d_call_set`, `j_call_set` | equivalence set (all alleles the evidence cannot distinguish). |
| `v_resolved_call`, `…_call_level`, `…_set_confidence` | most-specific safe call (allele→gene→family→abstain) + confidence. *AlignAIR extension columns.* |
| `productive` | model head. |
| `vj_in_frame` | junction length is a multiple of 3. |
| `stop_codon` | a stop appears in the V→J coding frame (anchored at the conserved Cys-104 codon). |
| `junction`, `junction_aa`, `junction_length` | CDR3 incl. the conserved Cys/Trp(Phe) codons; empty when V/J anchors are unrecoverable (e.g. short fragments). |
| `np1`, `np1_length`, `np2`, `np2_length` | non-templated nucleotides V→D and D→J (V→J when there is no D). |
| `*_sequence_start/end`, `*_germline_start/end` | 1-based AIRR coordinates. |
| `v_cigar`, `d_cigar`, `j_cigar` | exact CIGAR from gapped alignment (`--no-full-alignment` falls back to a coordinate-derived CIGAR). |
| `sequence_alignment`, `germline_alignment` | gapped alignments (require `parasail`; in `[cli]` extra). |
| `v_identity`, `d_identity`, `j_identity` | percent identity from the gapped alignment. |
| `is_contaminant` | *AlignAIR extension*: advisory out-of-scope flag; calls are retained regardless. |

## Provided by you, preserved into the output

These are carried straight through from `--metadata` (e.g. a 10x
`filtered_contig_annotations.csv` or an AIRR sample sheet), joined by read id — see
the [10x](https://github.com/MuteJester/AlignAIR/tree/main/examples/10x) and
[bulk-AIRR](https://github.com/MuteJester/AlignAIR/tree/main/examples/airr) examples.

| Field(s) | Source |
|----------|--------|
| `cell_id` / `barcode` | 10x barcode; group rows by this to reconstruct a cell (Scirpy/Change-O). |
| `umi_count` / `umis`, `duplicate_count`, `consensus_count`, `reads` | quantitation columns. |
| `sample_id`, `subject_id`, and any `--keep-columns` you name | sample sheet. |
| `c_call` / `c_gene` | constant region from the upstream assembler (10x `c_gene`); see below. |

## Deferred (not emitted yet)

| Field(s) | Why |
|----------|-----|
| `fwr1..4`, `cdr1..3` (+ `_aa`, `_start`, `_end`) | require IMGT germline region numbering per allele, which AlignAIR does not model. Planned. |
| `c_call` (from the read) | AlignAIR aligns V(D)J only; it does not detect the constant region. Use the assembler's `c_gene` via `--metadata` for now. |
| `germline_alignment_d_mask`, `v_germline_alignment` … | per-segment germline alignments beyond the stitched `germline_alignment`. Planned. |
| `complete_vdj`, `locus`-level QC beyond `is_contaminant` | under consideration. |

If your pipeline needs one of the deferred fields, open an issue — this matrix is the
place we track and prioritise them.
