"""Turn a GenAIRR record into an AIRRistotle training example: a genotype-in-prompt + read, followed
by the annotation, where CALLS and COORDINATES are copy-pointer targets INTO the prompt (exact,
single-source). MVP: small genotype = the true V/D/J alleles + a few distractors (no novel-allele /
dynamic sampling yet). Target arrays are aligned so label[t] describes token t; the training loss
applies the causal next-token shift (hidden[t] predicts label[t+1])."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Example:
    input_ids: list          # prompt tokens then annotation tokens (ids)
    gen_target: list         # vocab id (used when is_copy==0 and loss_mask==1)
    copy_target: list        # index into input_ids (< prompt_len) (used when is_copy==1)
    is_copy: list            # 1 if this step's label is a copy-pointer, else 0
    loss_mask: list          # 1 if this step contributes to the loss (annotation region), else 0
    prompt_len: int


def _allele_block(tokens, tok, marker, name, seq):
    """Append '<marker> <name-as-chars> <S> <seq-as-chars>' to tokens (ids); return (marker_pos, seq_start)."""
    marker_pos = len(tokens)
    tokens.append(tok.id(marker))
    tokens.extend(tok.encode(list(name)))
    tokens.append(tok.id(tok.S))
    seq_start = len(tokens)
    tokens.extend(tok.encode(list(seq.upper())))
    return marker_pos, seq_start


def build_example(record, reference_set, tokenizer, n_distractors: int = 8, rng=None):
    import random as _random
    rng = rng or _random.Random(0)
    tok = tokenizer
    seq = str(record["sequence"]).upper()
    genes = {"v": ("V", tok.V), "d": ("D", tok.D), "j": ("J", tok.J)}

    # ---- PROMPT: genotype allele blocks + read ----
    tokens = [tok.id(tok.GENO)]
    call_marker_pos, germ_block_start = {}, {}
    for g, (G, marker) in genes.items():
        true_call = str(record.get(f"{g}_call") or "").split(",")[0]
        gene = reference_set.gene(G)
        if not true_call or true_call not in gene.names:
            continue
        names = [true_call]
        pool = [n for n in gene.names if n != true_call]
        rng.shuffle(pool)
        names += pool[:n_distractors]
        rng.shuffle(names)                         # true allele not always first
        for nm in names:
            s = gene.sequences[gene.names.index(nm)]
            mpos, sstart = _allele_block(tokens, tok, marker, nm, s)
            if nm == true_call:
                call_marker_pos[g] = mpos
                germ_block_start[g] = sstart
    tokens.append(tok.id(tok.READ))
    read_block_start = len(tokens)
    tokens.extend(tok.encode(list(seq)))
    prompt_len = len(tokens)

    # ---- ANNOTATION (targets aligned to the token emitted at each step) ----
    gen_t, copy_t, is_copy, loss = [], [], [], []

    def emit_gen(marker_or_char):                  # generate a vocab token
        tid = tok.id(marker_or_char)
        tokens.append(tid); gen_t.append(tid); copy_t.append(0); is_copy.append(0); loss.append(1)

    def emit_copy(prompt_pos, placeholder):        # copy step: label = prompt_pos; input token = placeholder
        tokens.append(tok.id(placeholder)); gen_t.append(0); copy_t.append(int(prompt_pos))
        is_copy.append(1); loss.append(1)

    emit_gen(tok.ANNOT)
    emit_gen(tok.ORI); emit_gen("+")               # MVP: forward reads only
    for g, (G, marker) in genes.items():
        if g not in call_marker_pos:
            continue
        emit_gen(marker)
        emit_copy(call_marker_pos[g], marker)      # CALL = copy the true allele's <marker> position
        fields = [(f"{G}S", read_block_start + int(record[f"{g}_sequence_start"])),
                  (f"{G}E", read_block_start + int(record[f"{g}_sequence_end"])),
                  (f"{G}GS", germ_block_start[g] + int(record[f"{g}_germline_start"])),
                  (f"{G}GE", germ_block_start[g] + int(record[f"{g}_germline_end"]))]
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
