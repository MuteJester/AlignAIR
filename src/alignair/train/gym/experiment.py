"""GenAIRR Experiment Factory."""


def build_experiment(dataconfig, params, allow_curatable: bool = False):
    """Compile a GenAIRR experiment at the given curriculum params (forward orientation).
    allow_curatable: permit simulation from references with curatable issues (e.g. alleles with
    no detected anchor) — needed for some custom FASTA references built via the cartridge builder."""
    from GenAIRR import Experiment
    exp = Experiment.on(dataconfig)
    if allow_curatable:
        exp = exp.allow_curatable_refdata()

    genotype = params.get("genotype")
    if genotype is None and params.get("genotype_seed") is not None:
        from GenAIRR.genotype import Genotype

        genotype = Genotype.sample(
            dataconfig,
            seed=int(params["genotype_seed"]),
            subject_id=params.get("genotype_subject_id"),
        )
    if genotype is not None:
        exp = exp.with_genotype(genotype)

    allele_restrictions = params.get("restrict_alleles")
    if allele_restrictions:
        exp = exp.restrict_alleles(
            **{str(key).lower(): value for key, value in allele_restrictions.items()}
        )

    recombine_kwargs = {
        key: params[key]
        for key in (
            "np1_lengths",
            "np2_lengths",
            "v_allele_weights",
            "d_allele_weights",
            "j_allele_weights",
        )
        if params.get(key) is not None
    }
    exp = exp.recombine(**recombine_kwargs)
    if params.get("productive_only", False):
        exp = exp.productive_only()

    mutation_kwargs = {
        "model": params.get("mutation_model", "s5f"),
        "s5f_model": params.get("s5f_model", "hh_s5f"),
    }
    if params.get("segment_rates") is not None:
        mutation_kwargs["segment_rates"] = params["segment_rates"]
    if params.get("v_subregion_rates") is not None:
        mutation_kwargs["v_subregion_rates"] = params["v_subregion_rates"]
    if params.get("mutation_count") is not None:   # per-read SHM distribution (stratified)
        exp = exp.mutate(count=params["mutation_count"], **mutation_kwargs)
    elif params.get("mutation_rate"):              # 0/None => skip SHM (TCR loci: GenAIRR forbids
        exp = exp.mutate(rate=params["mutation_rate"], **mutation_kwargs)   # mutate(); use seq-errors/indels
    if dataconfig.metadata.has_d:
        exp = exp.invert_d(prob=float(params.get("invert_d_prob", 0.05)))
    revision_prob = float(params.get("receptor_revision_prob", 0.0))
    if revision_prob > 0.0:
        exp = exp.receptor_revision(
            prob=revision_prob,
            same_haplotype=bool(params.get("receptor_revision_same_haplotype", True)),
        )
    exp = exp.end_loss_5prime(length=params["end_loss_5"]).end_loss_3prime(length=params["end_loss_3"])

    pcr_count = params.get("pcr_error_count")
    pcr_rate = params.get("pcr_error_rate")
    if pcr_count is not None or pcr_rate is not None:
        exp = exp.pcr_amplify(count=pcr_count, rate=pcr_rate)

    exp = exp.polymerase_indels(
        count=params["indel_count"],
        insertion_prob=float(params.get("indel_insertion_prob", 0.5)),
    )
    if params.get("seq_error_count") is not None:
        exp = exp.sequencing_errors(count=params["seq_error_count"])
    else:
        exp = exp.sequencing_errors(rate=params["seq_error_rate"])
    exp = exp.ambiguous_base_calls(count=params["ambiguous_count"])
    contaminate_prob = float(params.get("contaminate_prob", 0.0))
    if contaminate_prob > 0.0:
        exp = exp.contaminate(prob=contaminate_prob)
    paired_end = params.get("paired_end")
    if paired_end:
        exp = exp.paired_end(**paired_end)
    return exp.compile()
