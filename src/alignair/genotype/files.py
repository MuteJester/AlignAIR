"""Genotype files: ``{segment: [allele names]}`` (segments V, D, J, any case) as JSON or YAML, read and their keys
checked; ``genotype.bank`` checks the names against a reference. A file that is not such a mapping (a FASTA, broken YAML
or JSON, a list, a segment given one string) or one keyed by gene names is refused with a ConfigurationError whose hint
gives the format."""
from __future__ import annotations

import re

from ..errors import ConfigurationError

SEGMENTS = ("V", "D", "J")
FORMAT_HINT = ("A genotype is YAML or JSON keyed by segment (V, D, J), each listing allele names of the bound "
               "reference, e.g. 'V: [IGHV1-2*02, IGHV3-23*01]'. Germline sequences (a FASTA) go to --reference.")
_GENE_NAME = re.compile(r"^(IG[HKL]|TR[ABGD])[VDJ]", re.IGNORECASE)


def check_segment_keys(keys, *, source: str = "the genotype", context=None) -> None:
    """ConfigurationError unless every key is a segment (V, D, J, any case): one naming the gene names given, when
    keys look like gene names (IGHV3-23), else one naming the keys."""
    bad = [str(k) for k in keys if str(k).upper() not in SEGMENTS]
    if not bad:
        return
    genes = [k for k in bad if _GENE_NAME.match(k)]
    context = {**(context or {}), "keys": bad}
    if genes:
        raise ConfigurationError(f"{source} is keyed by gene names ({', '.join(genes)}); its keys are segments "
                                 "(V, D, J), each listing allele names", context=context, hint=FORMAT_HINT)
    raise ConfigurationError(f"{source}'s keys are segments (V, D, J); got {', '.join(bad)}", context=context,
                             hint=FORMAT_HINT)


def check_segment_values(genotype, *, source: str = "the genotype is not {V|D|J: allele names}",
                         context=None) -> None:
    """ConfigurationError unless each segment's entry is a collection of allele names: not one string (which would be
    read as its characters), not None or a number. ``source`` begins the message."""
    for segment, names in genotype.items():
        if isinstance(names, (str, bytes)) or not hasattr(names, "__iter__"):
            raise ConfigurationError(f"{source}: its {segment} entry is {type(names).__name__} {names!r}, not a list "
                                     "of allele names", context={**(context or {}), "segment": str(segment)},
                                     hint=FORMAT_HINT)


def load_genotype(path: str) -> dict[str, set[str]]:
    """Load a genotype from JSON or YAML: ``{segment: [allele names]}`` (a ``{name: sequence}`` mapping per segment
    works too; the sequences are ignored), or an AIRR GenotypeSet such as ``alignair genotype`` writes. Returns
    ``{segment: set(names)}``, segments lower-case. ConfigurationError for a file that is not such a mapping, or one
    keyed by gene names. ``genotype_bank`` checks its alleles against a
    reference."""
    with open(path) as f:
        text = f.read()
    data = _parse(text, path)
    if "genotype_class_list" in data:                   # an AIRR GenotypeSet (``alignair genotype`` writes one)
        data = _segments_of_genotype_set(data, path)
    check_segment_keys(data, source=f"{path}", context={"path": path})
    check_segment_values(data, source=f"{path} is not a YAML or JSON genotype", context={"path": path})
    return {str(g).lower(): {str(n) for n in names} for g, names in data.items()}


def _segments_of_genotype_set(genotype_set: dict, path: str) -> dict[str, list[str]]:
    """The documented alleles of every genotype in an AIRR GenotypeSet, by segment (the letter after the locus in each
    name: IGHV1-2*02 is V). ConfigurationError for a label whose segment its name does not tell."""
    segments: dict[str, list[str]] = {}
    for genotype in genotype_set.get("genotype_class_list") or []:
        for allele in genotype.get("documented_alleles") or []:
            label = str(allele.get("label", ""))
            match = _GENE_NAME.match(label)
            if match is None:
                raise ConfigurationError(f"{path}: the GenotypeSet allele {label!r} does not name its segment "
                                         "(V, D or J after the locus, e.g. IGHV1-2*02)",
                                         context={"path": path, "allele": label}, hint=FORMAT_HINT)
            segments.setdefault(label[match.end() - 1].upper(), []).append(label)
    return segments


def _parse(text: str, path: str) -> dict:
    """The file's mapping: YAML for a .yaml/.yml file, else JSON, else YAML. ConfigurationError when neither parses
    to a mapping (a FASTA, broken YAML or JSON, a list or a plain string)."""
    import json

    import yaml

    def refused(reason):
        return ConfigurationError(f"{path} is not a YAML or JSON genotype: {reason}", context={"path": path},
                                  hint=FORMAT_HINT)
    try:
        if path.endswith((".yaml", ".yml")):
            data = yaml.safe_load(text)
        else:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        problem = str(getattr(exc, "problem", None) or exc).strip().splitlines()[0]
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1})" if mark is not None else ""
        raise refused(f"it does not parse{where}: {problem}") from exc
    if not isinstance(data, dict):
        raise refused(f"it holds {type(data).__name__ if data is not None else 'nothing'}, not a mapping of "
                      "segments to allele names")
    return data
