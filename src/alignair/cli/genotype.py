"""``alignair genotype`` — a donor's V genotype from an AIRR rearrangement table, by the frequency rule
``predict --infer-genotype`` uses (``genotype.infer``, TIgGER's rule): per V gene, the most frequent alleles among the
rows whose ``v_call`` is a single allele, until they explain ``--explain`` of those rows; a gene with fewer than
``--min-reads`` such rows has no evidence and is left open. Any tool's table works (a ``v_call`` listing several
alleles is a tie and counts for none). Writes the genotype as an AIRR GenotypeSet and prints it, gene by gene.
"""
from __future__ import annotations

import csv

from ..errors import ConfigurationError, InputError, report_error
from ..genotype.infer import EXPLAIN, MIN_READS
from ..reference.loci import FALLBACK_LOCUS


def register(sub) -> None:
    p = sub.add_parser("genotype", help="a donor's V genotype from an AIRR table (the frequency rule of predict "
                                        "--infer-genotype), written as an AIRR GenotypeSet")
    p.add_argument("--airr", required=True, help="an AIRR rearrangement TSV with a v_call column (any tool's)")
    p.add_argument("--out", required=True, help="the AIRR GenotypeSet JSON to write")
    p.add_argument("--locus", default=FALLBACK_LOCUS, help="the genotype's locus (default %(default)s)")
    p.add_argument("--explain", type=float, default=EXPLAIN,
                   help="the share of a gene's single-allele rows its alleles explain (default %(default)s)")
    p.add_argument("--min-reads", type=int, default=MIN_READS,
                   help="the single-allele rows a gene needs to be genotyped (default %(default)s)")
    p.set_defaults(func=run)


def run(args) -> int:
    from ..genotype.airr import write_genotype_set
    from ..genotype.infer import GenotypeEvidence
    if not (0.0 < args.explain <= 1.0) or args.min_reads < 1:
        return report_error(ConfigurationError("--explain must be in (0, 1] and --min-reads at least 1",
                                       context={"explain": args.explain, "min_reads": args.min_reads}))
    evidence = GenotypeEvidence()
    with open(args.airr, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "v_call" not in (reader.fieldnames or ()):
            return report_error(InputError(f"{args.airr} has no v_call column", context={"path": args.airr}))
        for index, row in enumerate(reader):
            evidence.observe(index, row["v_call"])
    genotype = evidence.genotype(explain=args.explain, min_reads=args.min_reads)
    write_genotype_set(args.out, genotype, args.locus)
    for gene, counts in sorted(evidence.counts.items()):
        if gene in genotype:
            print(f"{gene}: {', '.join(genotype[gene])}")
        else:
            print(f"{gene}: open ({sum(counts.values())} single-allele rows, fewer than {args.min_reads})")
    print(f"genotyped {len(genotype)} of {len(evidence.counts)} V genes -> {args.out}")
    return 0
