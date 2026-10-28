import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const referenceLabels: DocPage = {
  slug: "reference-labels",
  title: "Reference and label API",
  section: "Design",
  lead: "Derive protein germlines and class labels from the same ordered reference used for nucleotide allele prediction.",
  body: () => (
    <>
      <p>
        <code>alignair.reference.ReferenceSet</code> owns nucleotide allele names, sequences, indices,
        anchors, and locus membership. <code>build_protein_reference</code> creates a protein view from
        that source. It returns translated V/J germlines, anchor-check quarantine reasons,
        protein-equivalence label maps, and their reference identity. The source allele order is preserved.
      </p>
      <H2 id="construction">Build a shared reference view</H2>
      <CodeBlock lang="python" code={`import GenAIRR as ga
from alignair.reference import ReferenceSet, build_protein_reference

source = ReferenceSet.from_dataconfigs(ga.HUMAN_IGH_OGRDB)
protein = build_protein_reference(source)

allele_name = source.gene("V").names[0]
nucleotide_index = source.gene("V").index[allele_name]
protein_class = protein.class_maps["V"].class_of(allele_name)
compatible_alleles = protein.class_maps["V"].ambiguity_set(protein_class)

print(protein.metadata())
print(protein.germlines.quarantined)`} />
      <p>
        The builder accepts an existing <code>ReferenceSet</code>, a GenAIRR dataconfig, or a dataconfig
        name. Supplying an existing reference retains that object as <code>protein.source</code>.
        A custom reference can use the same path when its <code>GeneReference</code> entries provide
        conserved anchors. Allele names, sequence rows, and allele-index mappings must agree;
        inconsistent source tables raise <code>DataContractError</code>.
      </p>
      <H2 id="labels">Allele indices and protein labels</H2>
      <DocTable head={["Representation", "Index semantics"]} rows={[
        ["ReferenceSet / GeneReference", "Allele indices follow the source reference order. They remain the indices used by nucleotide allele heads and full-allele protein experiments."],
        ["EquivalenceClassMap", "V/J alleles with identical accepted translations share a class. Class IDs follow first occurrence while visiting sorted allele names; class members are stored as sorted allele names."],
        ["D alleles", "D keeps the source allele vocabulary. The protein-reference builder excludes fixed-frame D equivalence classes and rejects a request to construct them."],
      ]} />
      <p>
        <code>class_of(name)</code> returns a protein class ID, <code>ambiguity_set(id)</code> returns
        its member alleles, and <code>n_classes()</code> sizes the corresponding categorical head.
        Quarantined alleles retain their source nucleotide indices and are omitted from protein class
        maps. An unknown or quarantined allele lookup raises <code>KeyError</code>, allowing a sample
        policy to record that rejection explicitly. A required gene with no accepted protein labels
        raises <code>DataContractError</code> during reference construction.
      </p>
      <p>
        Match these label counts to the <DocLink to="model-composition">model configuration</DocLink>{" "}
        and carry the same maps into sample preparation and inference. The{" "}
        <DocLink to="training-batches">batch adapter API</DocLink> describes categorical class targets
        and full-allele multi-hot targets.
      </p>
      <H2 id="quality">Anchor checks and explicit policies</H2>
      <p>
        <code>build_protein_germlines(source, genes=("V", "J"))</code> provides the translated
        germlines and quarantine results without assembling class maps. Translation starts at
        <code> anchor % 3</code> and checks the residue at the conserved anchor. Missing, invalid,
        out-of-range, or mismatched anchors produce a quarantine reason. Negative, boolean, and
        fractional anchors are invalid.
      </p>
      <p>
        Defaults preserve the existing IGH policy: V expects C and J expects W. A different source policy
        must be supplied explicitly through <code>anchor_residues</code>, for example a validated
        per-gene mapping selected by the caller. The builder does not infer a replacement policy from
        whichever residue happens to appear. Anchor checking retains the existing scope; sample policies
        decide how to handle stop-containing or nonproductive reads.
      </p>
      <H2 id="analysis">Analyze protein-class ambiguity</H2>
      <CodeBlock lang="python" code={`# Inspect ambiguous V classes in the reference constructed above.
ambiguous_v = {
    class_id: members
    for class_id, members in protein.class_maps["V"].class_members.items()
    if len(members) > 1
}
print(ambiguous_v)`} />
      <CodeBlock lang="python" code={`# One representative protein per class, keyed by the model's class IDs.
v_prototypes = protein.prototype_sequences("V")
class_ids = sorted(v_prototypes)
prototype_sequences = [v_prototypes[class_id] for class_id in class_ids]`} />
      <p>
        Equivalence classes record equality of complete reference protein translations. Keep all
        member alleles when interpreting a class prediction. Choosing the first member introduces
        an allele-order tie break; it does not establish which nucleotide allele generated the read.
        This reference relationship does not bound predictive accuracy on mutated, partial, or
        indel-affected observations.
      </p>
      <p>
        <code>ProteinReference.prototype_sequences(gene)</code> keeps class IDs attached to their
        representative translations. The representative is the first stored member of each class;
        all members have the same complete translated sequence. Quarantined alleles remain outside
        these classes. An absent class vocabulary raises <code>ConfigurationError</code>.
      </p>
      <p>
        Use the <DocLink to="evaluation-reports">observation collector</DocLink> to compare supervised
        truth with model predictions and the <DocLink to="model-inference">feature-inspection API</DocLink>
        to examine named component inputs. Preserve the reference label mapping and observation support
        when joining these analyses. The unused <code>aa.forensics</code> module has been removed;
        its anchor-aligned identity heuristic is not provided as an information-ceiling estimator
        or a replacement allele caller.
      </p>
      <p>
        The unused <code>aa.local_aligner</code> protein-class baseline and the duplicate
        <code> reference.protein.prototypes_by_class</code> helper have also been removed. Use
        <code> protein.prototype_sequences(gene)</code> for prototype lookup. The existing
        <code> alignair.align</code> backends retain their nucleotide alignment scoring and
        query-global, germline-ends-free coordinate contract; they do not provide the former
        BLOSUM62 local protein-class caller. Parasail remains an optional dependency for active
        nucleotide alignment.
      </p>
      <H2 id="nomenclature">Expand allele names without losing provenance</H2>
      <CodeBlock lang="python" code={`from alignair.reference import (
    asc_to_imgt_multi, class_alleles, class_genes, class_families,
)

# The original dataconfig owns ASC provenance; ReferenceSet owns model allele order.
allele_mapping = asc_to_imgt_multi(ga.HUMAN_IGH_OGRDB)
v_labels = protein.class_maps["V"]
v_class = v_labels.class_of(allele_name)

alleles = class_alleles(v_class, v_labels, allele_mapping)
genes = class_genes(v_class, v_labels, allele_mapping)
families = class_families(v_class, v_labels, allele_mapping)
print(alleles, genes, families)`} />
      <p>
        <code>alignair.reference.nomenclature</code> owns allele-name parsing and class expansion
        for either model input alphabet. Class helpers return sorted, unique lists. Omitting
        <code> allele_mapping</code> preserves the class’s original names; an unmapped member also
        retains its original name. Each mapped member expands to every recorded origin before
        gene or family names are deduplicated. Multiple origins are retained even when they span
        different genes or families. No representative allele is selected.
      </p>
      <p>
        <code>asc_to_imgt_multi(dataconfig)</code> reads the authoritative
        <code> asc_tables["V"]</code> rows and supports V-only ASC provenance. Additional ASC gene
        tables require an explicit implementation rather than silently applying the V policy.
        <code> gene_of(name)</code> removes the allele suffix while preserving the naming scheme.
        <code> family_of(name)</code> currently recognizes numeric IMGT IGHV, IGHD and IGHJ family
        prefixes; it does not interpret an ASC cluster ID as an IMGT family. Use an explicit
        provenance mapping before family scoring. <code>catalogue_families(class_map, allele_mapping)</code>
        applies the same policy across every class.
      </p>
      <p>
        Invalid names, malformed mappings and invalid class-ID types raise
        <code> ConfigurationError</code> with reference-nomenclature context. A valid integer ID
        absent from the class map raises <code>KeyError</code>. New expanded lists can be modified
        without changing the source class map or provenance table.
      </p>
      <p>
        Product predictions use this shared expansion for top calls and calibrated class sets.
        Phase-3 <code>score_gold</code> and <code>score_concordance</code> accept the same explicit
        <code> allele_mapping</code> keyword; their default uses the supplied class names directly.
        Bind the mapping to the study’s declared reference before comparing IMGT labels.
        Lineage selection, confidence intervals, bootstrap sampling and endpoint decisions retain
        their existing procedures.
      </p>
      <H2 id="identity">Reference identity and hash scope</H2>
      <CodeBlock lang="python" code={`# Training class maps retain their historical V/J scope.
training_reference = build_protein_reference(source)
assert training_reference.hash_genes == ("V", "J")

# Include D when the reference identity must also bind the D catalogue.
vdj_reference = build_protein_reference(source, hash_genes=("V", "D", "J"))
assert vdj_reference.hash_genes == ("V", "D", "J")`} />
      <p>
        The identity includes the source nucleotide allele names and sequences for the selected scope,
        including quarantined rows. The scope must include every projected gene. An absent D gene
        contributes no rows. The legacy order-independent, 16-hex SHA-1 identity is preserved so existing
        class-map and calibration bindings remain stable. Artifact-integrity checks continue to use their
        existing checksums.
      </p>
      <H2 id="serialization">Serialization and compatibility</H2>
      <p>
        <code>EquivalenceClassMap.to_json()</code> and <code>from_json()</code> retain the existing
        <code> alignair.aa.equivalence.v1</code> wire schema. This stored format identifier remains
        valid after removal of the Python compatibility module. Relocating the implementation does not rename
        that stored schema or assign new class IDs. Loading validates contiguous integer IDs, nonempty
        classes, and exact agreement between allele-to-class and class-to-member mappings. Unsupported
        schemas and malformed maps raise <code>ModelError</code> before prediction or calibration uses them.
        Historical documents without a schema field remain accepted when their mappings are valid.
      </p>
      <p>
        The implementations live in <code>alignair.reference.protein</code> and
        <code> alignair.reference.labels</code>. Import <code>build_class_maps</code>
        directly from <code>alignair.reference.protein</code>.
        Training setup, convergence evaluation, and product loading use the shared owners.
        Study-specific read filtering remains in the sample-preparation layer.
      </p>
      <p>
        The exploratory <code>alignair.aa.phase0</code>, <code>aa.audit</code> and
        <code> aa.report</code> APIs have been retired. Their simulation report and fixed-window
        summaries have no replacement entry point. Reference inspection uses the shared
        <code> ProteinReference.metadata()</code> and <code>EquivalenceClassMap</code> APIs above;
        codon projection remains in <code>alignair.data.projection</code>.
      </p>
      <H2 id="migration">Migrate former protein reference imports</H2>
      <p>
        The <code>aa.reference</code> and <code>aa.equivalence</code> facades have been removed.
        Use the shared imports below. Class IDs, member ordering, reference hashes and serialized
        class-map schemas keep their existing contracts.
      </p>
      <DocTable head={["Former import", "Replacement"]} rows={[
        ["aa.taxonomy.gene_of / family_of / class_genes / class_families / catalogue_families", "alignair.reference.nomenclature (same function names; class/catalogue helpers return sorted lists)"],
        ["aa.product.asc functions", "alignair.reference.nomenclature (same function names)"],
        ["aa.reference.ANCHOR_AA", "alignair.reference.protein.DEFAULT_ANCHOR_RESIDUES"],
        ["aa.reference.AAGermline", "alignair.reference.ProteinGermline"],
        ["aa.reference.AAGermlineSet", "alignair.reference.ProteinGermlineSet"],
        ["aa.reference.build_aa_germlines", "alignair.reference.build_protein_germlines"],
        ["aa.reference.translate_germline", "alignair.reference.protein.translate_germline"],
        ["aa.equivalence.EquivalenceClassMap", "alignair.reference.EquivalenceClassMap"],
        ["aa.equivalence.build_equivalence_classes", "alignair.reference.build_equivalence_classes"],
        ["aa.equivalence.db_hash", "alignair.reference.labels.db_hash"],
      ]} />
      <p>
        The former <code>asc_to_imgt_map</code> and <code>ImgtView</code> representative-only helpers
        have been removed. Use the complete mapping from <code>asc_to_imgt_multi</code> with shared
        class expansion, or pass it explicitly to study scoring. Class IDs, stored class maps,
        calibration bindings and product result fields retain their existing contracts.
      </p>
    </>
  ),
};

export const labelPages = [referenceLabels];
