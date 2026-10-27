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
      <H2 id="identity">Reference identity and hash scope</H2>
      <CodeBlock lang="python" code={`# Training class maps retain their historical V/J scope.
training_reference = build_protein_reference(source)
assert training_reference.hash_genes == ("V", "J")

# The Phase-0 audit retains its historical V/D/J scope.
audit_reference = build_protein_reference(source, hash_genes=("V", "D", "J"))`} />
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
        <code> alignair.aa.equivalence.v1</code> wire schema. Relocating the implementation does not rename
        that stored schema or assign new class IDs. Loading validates contiguous integer IDs, nonempty
        classes, and exact agreement between allele-to-class and class-to-member mappings. Unsupported
        schemas and malformed maps raise <code>ModelError</code> before prediction or calibration uses them.
        Historical documents without a schema field remain accepted when their mappings are valid.
      </p>
      <p>
        The implementations live in <code>alignair.reference.protein</code> and
        <code> alignair.reference.labels</code>. Historical <code>aa.reference</code>,
        <code> aa.equivalence</code>, and <code>aa.data.build_class_maps</code> imports delegate to them.
        Phase-0, training setup, convergence evaluation, and product loading use the shared owners.
        Study-specific read filtering remains in the sample-preparation layer.
      </p>
    </>
  ),
};

export const labelPages = [referenceLabels];
