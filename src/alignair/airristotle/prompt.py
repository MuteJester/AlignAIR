"""Turn a GenAIRR record into an AIRRistotle training example: a genotype-in-prompt + read, followed
by the annotation, where CALLS and COORDINATES are copy-pointer targets INTO the prompt (exact,
single-source). MVP: small genotype = the true V/D/J alleles + a few distractors (no novel-allele /
dynamic sampling yet). Target arrays are aligned so label[t] describes token t; the training loss
applies the causal next-token shift (hidden[t] predicts label[t+1]).

`build_prompt_core` (prompt + metadata) is shared by training (`build_example`) and inference/decode."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Block:
    marker_pos: int          # prompt index of this allele's <V>/<D>/<J> marker
    seq_start: int           # prompt index where this allele's germline sequence starts
    seq_len: int
    gene: str                # "v"/"d"/"j"
    name: str


@dataclass
class PromptMeta:
    read_block_start: int
    read_len: int
    blocks: list                        # all allele blocks (for decode resolution)
    call_marker_pos: dict               # gene -> prompt index of the TRUE allele's marker (training)
    germ_block_start: dict              # gene -> prompt index where the TRUE allele's seq starts
    prompt_len: int


@dataclass
class Example:
    input_ids: list
    gen_target: list
    copy_target: list
    is_copy: list
    loss_mask: list
    prompt_len: int


def _allele_block(tokens, tok, marker, name, seq):
    marker_pos = len(tokens)
    tokens.append(tok.id(marker))
    tokens.extend(tok.encode(list(name)))
    tokens.append(tok.id(tok.S))
    seq_start = len(tokens)
    tokens.extend(tok.encode(list(seq.upper())))
    return marker_pos, seq_start, len(seq)


def build_prompt_core(record, reference_set, tokenizer, n_distractors: int = 8, rng=None):
    """Build the PROMPT (genotype allele blocks + read) and its metadata. Genotype = the record's true
    V/D/J alleles + `n_distractors` random distractors per gene (shuffled)."""
    import random as _random
    rng = rng or _random.Random(0)
    tok = tokenizer
    seq = str(record["sequence"]).upper()
    genes = {"v": ("V", tok.V), "d": ("D", tok.D), "j": ("J", tok.J)}
    tokens = [tok.id(tok.GENO)]
    blocks, call_marker_pos, germ_block_start = [], {}, {}
    for g, (G, marker) in genes.items():
        true_call = str(record.get(f"{g}_call") or "").split(",")[0]
        gene = reference_set.gene(G)
        if not true_call or true_call not in gene.names:
            continue
        names = [true_call]
        pool = [n for n in gene.names if n != true_call]
        rng.shuffle(pool)
        names += pool[:n_distractors]
        rng.shuffle(names)
        for nm in names:
            s = gene.sequences[gene.names.index(nm)]
            mpos, sstart, slen = _allele_block(tokens, tok, marker, nm, s)
            blocks.append(Block(mpos, sstart, slen, g, nm))
            if nm == true_call:
                call_marker_pos[g] = mpos
                germ_block_start[g] = sstart
    tokens.append(tok.id(tok.READ))
    read_block_start = len(tokens)
    tokens.extend(tok.encode(list(seq)))
    prompt_len = len(tokens)
    meta = PromptMeta(read_block_start=read_block_start, read_len=len(seq), blocks=blocks,
                      call_marker_pos=call_marker_pos, germ_block_start=germ_block_start,
                      prompt_len=prompt_len)
    return tokens, meta


def build_example(record, reference_set, tokenizer, n_distractors: int = 8, rng=None):
    tok = tokenizer
    tokens, meta = build_prompt_core(record, reference_set, tokenizer, n_distractors, rng)
    prompt_len = meta.prompt_len
    read_block_start = meta.read_block_start
    genes = {"v": ("V", tok.V), "d": ("D", tok.D), "j": ("J", tok.J)}

    gen_t, copy_t, is_copy, loss = [], [], [], []

    def emit_gen(marker_or_char):
        tid = tok.id(marker_or_char)
        tokens.append(tid); gen_t.append(tid); copy_t.append(0); is_copy.append(0); loss.append(1)

    def emit_copy(prompt_pos, placeholder):
        tokens.append(tok.id(placeholder)); gen_t.append(0); copy_t.append(int(prompt_pos))
        is_copy.append(1); loss.append(1)

    emit_gen(tok.ANNOT)
    emit_gen(tok.ORI); emit_gen("+")
    for g, (G, marker) in genes.items():
        if g not in meta.call_marker_pos:
            continue
        emit_gen(marker)
        emit_copy(meta.call_marker_pos[g], marker)
        fields = [(f"{G}S", read_block_start + int(record[f"{g}_sequence_start"])),
                  (f"{G}E", read_block_start + int(record[f"{g}_sequence_end"])),
                  (f"{G}GS", meta.germ_block_start[g] + int(record[f"{g}_germline_start"])),
                  (f"{G}GE", meta.germ_block_start[g] + int(record[f"{g}_germline_end"]))]
        for fmark, ppos in fields:
            if 0 <= ppos < prompt_len:
                emit_gen(getattr(tok, fmark)); emit_copy(ppos, getattr(tok, fmark))
    if record.get("junction_start") is not None:
        emit_gen(tok.JNS); emit_copy(read_block_start + int(record["junction_start"]), tok.JNS)
        emit_gen(tok.JNE); emit_copy(read_block_start + int(record["junction_end"]), tok.JNE)
    emit_gen(tok.PROD); emit_gen("1" if record.get("productive") else "0")
    emit_gen(tok.EOS)

    n_prompt = prompt_len
    return Example(
        input_ids=tokens,
        gen_target=[0] * n_prompt + gen_t,
        copy_target=[0] * n_prompt + copy_t,
        is_copy=[0] * n_prompt + is_copy,
        loss_mask=[0] * n_prompt + loss,
        prompt_len=prompt_len,
    )
