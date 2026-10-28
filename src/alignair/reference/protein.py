"""Protein views and label maps derived from the ordered nucleotide reference."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from operator import index

from ..errors import ConfigurationError, DataContractError
from .labels import EquivalenceClassMap, build_equivalence_classes, db_hash
from .reference_set import ReferenceSet

DEFAULT_ANCHOR_RESIDUES = {"V": "C", "J": "W"}


@dataclass(frozen=True)
class ProteinGermline:
    name: str
    aa_seq: str
    frame: int
    anchor_aa: int
    gene: str


@dataclass
class ProteinGermlineSet:
    genes: dict[str, dict[str, ProteinGermline]] = field(default_factory=dict)
    quarantined: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class ProteinReference:
    """One source reference plus its translated germlines, class maps, and explicit hash scope."""
    source: ReferenceSet
    germlines: ProteinGermlineSet
    class_maps: dict[str, EquivalenceClassMap]
    db_hash: str
    hash_genes: tuple[str, ...]

    def prototype_sequences(self, gene: str) -> dict[int, str]:
        """One representative translation per equivalence class in this reference."""
        gene = gene.upper()
        if gene not in self.class_maps:
            raise ConfigurationError("protein class vocabulary is absent", context={"gene": gene})
        return {label: self.germlines.genes[gene][members[0]].aa_seq
                for label, members in self.class_maps[gene].class_members.items()}

    def metadata(self) -> dict:
        return {**{f"n_{gene.lower()}_classes": labels.n_classes() for gene, labels in self.class_maps.items()},
                "db_hash": self.db_hash,
                **{f"quarantined_{gene.lower()}": len(rows) for gene, rows in self.germlines.quarantined.items()}}


def _genes(values, *, protein=False):
    try:
        genes = tuple(str(value).upper() for value in values)
    except TypeError as exc:
        raise ConfigurationError("reference genes must be an iterable of gene names") from exc
    if not genes or len(set(genes)) != len(genes) or not set(genes) <= {"V", "D", "J"}:
        raise ConfigurationError("reference genes must be a nonempty, unique selection of V, D, J")
    if protein and "D" in genes:
        raise ConfigurationError("D has no fixed translated germline frame; retain D allele-level labels")
    return genes


def _source_reference(source):
    if isinstance(source, ReferenceSet):
        return source
    if isinstance(source, str):
        import GenAIRR as ga
        try:
            source = getattr(ga, source)
        except AttributeError as exc:
            raise ConfigurationError("unknown reference dataconfig name") from exc
    if not callable(getattr(source, "allele_list", None)) or not hasattr(source, "metadata"):
        raise ConfigurationError("supply a ReferenceSet, a GenAIRR dataconfig, or its name")
    return ReferenceSet.from_dataconfigs(source)


def _gene(reference, gene):
    reference.validate(genes=(gene,))
    return reference.gene(gene)


def _anchor_policy(overrides):
    policy = dict(DEFAULT_ANCHOR_RESIDUES)
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ConfigurationError("anchor residues must be a per-gene mapping")
        for gene, residue in overrides.items():
            gene = str(gene).upper()
            if gene not in policy or not isinstance(residue, str) or residue not in "ACDEFGHIKLMNPQRSTVWY" or len(residue) != 1:
                raise ConfigurationError("anchor residues must select one canonical residue for V or J")
            policy[gene] = residue
    return policy


def translate_germline(name: str, ungapped_seq: str, anchor: int, gene: str, *, expected_residue=None):
    """Translate in the conserved-anchor frame; return a germline or a quarantine reason.

    Defaults preserve the existing IGH anchor policy. Other reference policies must be explicit.
    """
    from ..data.projection import translate_nt

    gene = _genes((gene,), protein=True)[0]
    expected = _anchor_policy({gene: expected_residue} if expected_residue is not None else None)[gene]
    try:
        if isinstance(anchor, bool) or index(anchor) < 0:
            raise ValueError
        anchor = index(anchor)
    except (ValueError, TypeError):
        return None, "invalid conserved anchor"
    frame = anchor % 3
    sequence = translate_nt(ungapped_seq, frame)
    position = (anchor - frame) // 3
    if position >= len(sequence):
        return None, f"anchor residue index {position} out of range (len {len(sequence)})"
    if sequence[position] != expected:
        return None, f"anchor residue {sequence[position]!r} != expected {expected!r}"
    return ProteinGermline(name, sequence, frame, position, gene), None


def build_protein_germlines(source, genes=("V", "J"), *, anchor_residues=None) -> ProteinGermlineSet:
    """Translate an ordered reference view; retain source indices and report failed anchor checks."""
    genes = _genes(genes, protein=True)
    policy = _anchor_policy(anchor_residues)
    reference = _source_reference(source)
    translated, quarantined = {}, {}
    for gene in genes:
        rows = _gene(reference, gene)
        translated[gene], quarantined[gene] = {}, {}
        for name, sequence in zip(rows.names, rows.sequences):
            anchor = (rows.anchors or {}).get(name)
            if anchor is None:
                quarantined[gene][name] = "no conserved anchor"
                continue
            germline, reason = translate_germline(name, sequence, anchor, gene, expected_residue=policy[gene])
            if germline is None:
                quarantined[gene][name] = reason
            else:
                translated[gene][name] = germline
    return ProteinGermlineSet(translated, quarantined)


def build_protein_reference(source, genes=("V", "J"), *, hash_genes=None, anchor_residues=None) -> ProteinReference:
    """Build protein labels without replacing or reordering the source nucleotide allele vocabulary.

    Hash scope defaults to projected genes. Explicit scopes preserve historical audit identities.
    An absent D gene contributes no rows when included in the hash scope.
    """
    genes = _genes(genes, protein=True)
    scope = _genes(genes if hash_genes is None else hash_genes)
    if not set(genes) <= set(scope):
        raise ConfigurationError("reference hash scope must include every projected gene")
    reference = _source_reference(source)
    germlines = build_protein_germlines(reference, genes, anchor_residues=anchor_residues)
    fingerprint = db_hash((name, sequence) for gene in scope if gene in reference.genes
                          for name, sequence in zip(_gene(reference, gene).names, reference.gene(gene).sequences))
    maps = {gene: build_equivalence_classes(gene, {name: row.aa_seq for name, row in germlines.genes[gene].items()},
                                           fingerprint) for gene in genes}
    for gene, labels in maps.items():
        if not labels.n_classes():
            raise DataContractError("reference has no usable protein labels for a required gene",
                                    context={"gene": gene, "quarantined": len(germlines.quarantined[gene])})
    return ProteinReference(reference, germlines, maps, fingerprint, scope)


def build_class_maps(dc_name="HUMAN_IGH_OGRDB"):
    """Historical V/J maps-and-metadata interface, backed by the shared reference view."""
    view = build_protein_reference(dc_name)
    return view.class_maps, view.metadata()
