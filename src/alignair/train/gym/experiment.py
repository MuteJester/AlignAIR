"""The GenAIRR experiment the curriculum, the record streams and the benchmark simulate reads with."""
from ...errors import ConfigurationError
from ...reference.reference_set import is_tcr

# The keys build_experiment reads: the curriculum's (``Curriculum.params``), productive_only (selected records), the
# share of inverted Ds, and a GenAIRR genotype (novel alleles).
EXPERIMENT_KEYS = frozenset({"mutation_rate", "end_loss_5", "end_loss_3", "indel_count", "seq_error_rate",
                             "ambiguous_count", "productive_only", "invert_d_prob", "genotype"})
# GenAIRR's own share of inverted Ds, for the reads that are not training records (the demo, a bundle's validation).
GENAIRR_INVERT_D_PROB = 0.05
MUTATION_MODEL, S5F_MODEL = "s5f", "hh_s5f"
INDEL_INSERTION_PROB = 0.5     # the share of polymerase indels that are insertions


def build_experiment(dataconfig, params, allow_curatable: bool = False):
    """Compile a GenAIRR experiment at the given curriculum params (forward orientation). ``params`` holds keys of
    ``EXPERIMENT_KEYS`` only (ConfigurationError otherwise), ``invert_d_prob`` among them. allow_curatable: permit
    simulation from references with curatable issues (e.g. alleles with no detected anchor), needed for some custom
    FASTA references built via the cartridge builder."""
    from GenAIRR import Experiment
    unknown = sorted(set(params) - EXPERIMENT_KEYS)
    if unknown:
        raise ConfigurationError(f"build_experiment does not read {', '.join(unknown)}", context={"field": unknown[0]},
                                 hint=f"its keys are {', '.join(sorted(EXPERIMENT_KEYS))}")
    if "invert_d_prob" not in params:
        raise ConfigurationError("build_experiment needs invert_d_prob, the share of records with an inverted D",
                                 context={"field": "invert_d_prob"})
    exp = Experiment.on(dataconfig)
    if allow_curatable:
        exp = exp.allow_curatable_refdata()
    if params.get("genotype") is not None:
        exp = exp.with_genotype(params["genotype"])
    exp = exp.recombine()
    if params.get("productive_only", False):
        exp = exp.productive_only()
    # Somatic hypermutation is BCR-only biology; GenAIRR rejects mutate() on TCR sequences.
    if not is_tcr(dataconfig) and params.get("mutation_rate"):
        exp = exp.mutate(rate=params["mutation_rate"], model=MUTATION_MODEL, s5f_model=S5F_MODEL)
    if dataconfig.metadata.has_d:
        exp = exp.invert_d(prob=float(params["invert_d_prob"]))
    exp = exp.end_loss_5prime(length=params["end_loss_5"]).end_loss_3prime(length=params["end_loss_3"])
    exp = exp.polymerase_indels(count=params["indel_count"], insertion_prob=INDEL_INSERTION_PROB)
    exp = exp.sequencing_errors(rate=params["seq_error_rate"])
    exp = exp.ambiguous_base_calls(count=params["ambiguous_count"])
    return exp.compile()
