import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const augmentation: DocPage = {
  slug: "protein-augmentation",
  title: "Protein augmentation API",
  section: "Design",
  lead: "Crop protein training reads with aligned reference evidence, explicit compatible classes, and independent boundary and classification masks.",
  body: () => (
    <>
      <p>
        <code>alignair.data.augmentation</code> owns crop geometry, protein corruption operators, and
        partial-class target generation. <code>ProteinCropAugmenter</code> composes a shared
        <DocLink to="reference-labels"> protein reference</DocLink> with a
        <DocLink to="sample-preparation"> sample builder</DocLink> and a NumPy random generator.
        It returns the same accepted-sample or rejection-reason contract as the builder.
      </p>
      <H2 id="usage">Prepare augmented samples</H2>
      <CodeBlock lang="python" code={`import numpy as np
from alignair.reference import build_protein_reference
from alignair.data.tokenizer import AAProteinTokenizer
from alignair.data.sampling import ProteinSampleBuilder
from alignair.data.augmentation import ProteinCropAugmenter, ProteinCropPolicy

reference = build_protein_reference("HUMAN_IGH_OGRDB")
builder = ProteinSampleBuilder(
    reference.class_maps, AAProteinTokenizer(), max_length=192,
    # Omit d_index for a V/J-only model.
    d_index=reference.source.gene("D").index,
)
augment = ProteinCropAugmenter(
    reference, builder, np.random.default_rng(17),
    ProteinCropPolicy(crop_probability=0.4,
                      substitution_probability=0.4, il_probability=0.15),
)
result = augment(record)`} />
      <p>
        Base acceptance remains the builder’s policy. The crop draw selects a prefix, suffix, or
        centered internal window, with a retained fraction drawn uniformly from 0.4 to 0.9 and at
        least one residue retained. Substitutions and I/L collapse have independent probability
        draws after cropping. These two corruptions preserve sequence length. The supplied record
        and its base token vector are not modified. Reusing the same input order and initial generator
        state reproduces augmentation draws.
      </p>
      <H2 id="coordinates">Map read codons to reference residues</H2>
      <p>
        All query, germline, and crop intervals are zero-based and half-open. Simulator segment CIGARs
        start at the declared segment coordinates. <code>alignair.data.alignment</code> validates the
        complete operation string and checks that its consumed query and reference lengths agree
        with those intervals. Operations follow the
        {" "}<a href="https://samtools.github.io/hts-specs/SAMv1.pdf">SAM CIGAR specification</a>:
        M, =, and X consume both axes; I/S consume query bases; D/N consume reference bases;
        H/P consume neither.
      </p>
      <p>
        A read codon supplies class evidence only when its three bases map consecutively to one
        complete codon in the germline’s conserved-anchor frame. Insertions, deletions, frame
        disagreement, and unknown reference residues can interrupt this evidence. Reliable adjacent
        residues form fragments; interruptions split them. Evidence comes from the source germline,
        so an observed somatic substitution does not eliminate its source class from the target set.
      </p>
      <H2 id="labels">Conservative compatible classes</H2>
      <p>
        A V/J class is compatible when its representative protein contains every reliable germline
        fragment in order. Fragment locations and gaps are unconstrained. For example, observing W G Q
        allows both WGQAA and AWGQAAV, while excluding WGREE. Read residue 10 can correspond to
        J-reference residue 0; it is never used directly as a prototype index.
      </p>
      <DocTable head={["Target", "Meaning"]} rows={[
        ["{gene}_valid", "Gene-level supervision. False when the crop contains no owned gene residues."],
        ["{gene}_class_supervise", "Classification-only supervision. False when no reliable class evidence is available; known boundaries still contribute."],
        ["{gene}_compatible", "Compatible class IDs. Empty on excluded classification rows."],
        ["{gene}_class", "Source class ID when classification is supervised; otherwise -1."],
        ["{gene}_class_reason", "Source label, reference fragments, absent gene, partial D, or the reason reference evidence is unavailable."],
        ["{gene}_reference_evidence", "Count of mapped reference residues used for a cropped V/J target."],
      ]} />
      <p>
        Missing CIGARs or reference coordinates, inconsistent alignment spans, and unavailable canonical
        codons produce an empty compatible set with classification disabled. They do not produce an
        all-class target. Malformed CIGARs and invalid coordinates raise <code>TrainingDataError</code>.
        The source class must remain in every generated nonempty compatible set.
      </p>
      <p>
        Uncropped rehearsal retains source singleton labels. D remains an allele target: a crop that
        retains its entire original visible interval keeps the source allele label; partial or absent
        D is excluded from all D losses. This does not establish full biological D observability.
        The <DocLink to="training-batches">categorical batch adapter</DocLink> preserves the masks;
        shared <DocLink to="training-tasks">losses and metrics</DocLink> use their own supervised counts.
        Diagnostic reason strings remain sample metadata rather than tensors sent to the model.
      </p>
      <H2 id="checkpoints">Training policy and existing checkpoints</H2>
      <p>
        Crop-training continuation checkpoints record the sample policy
        <code> protein-crop-reference-fragments-v1</code>. A full-state checkpoint with a missing or
        different policy cannot resume optimizer state. Use <code>TrainingConfig.init_from</code> for an explicit
        weights-only warm start with a fresh optimizer and schedule. This changes crop targets and
        removes the previous restriction on cropping trimmed V reads; existing scientific training
        results need to be re-evaluated under this policy. Inference weight loading is unchanged.
      </p>
      <p>
        The shared training lifecycle reseeds the source stream and augmentation generator from the
        seed and checkpoint step; it provides statistical continuation rather than exact replay of
        the next uninterrupted batch. See the <DocLink to="training-engine">training engine guide</DocLink>
        for checkpoint and process RNG contracts.
      </p>
      <H2 id="migration">Migrate former augmentation imports</H2>
      <p>
        The <code>alignair.aa.augment</code> and <code>alignair.aa.truncation</code> wrappers have
        been removed. Import their operators from the shared data modules.
      </p>
      <DocTable head={["Former import", "Replacement"]} rows={[
        ["aa.augment.aa_substitute / aa_indel / il_ambiguity / aa_truncate", "alignair.data.augmentation.protein (same function names)"],
        ["aa.truncation.crop_window / shift_clip / compatible_classes", "alignair.data.augmentation.cropping (same function names)"],
      ]} />
      <p>
        <code>compatible_classes</code> compares known reference-aligned windows. Use
        <code> fragment_compatible_classes</code> for fragments with unknown reference positions;
        <code> ProteinCropAugmenter</code> already applies that conservative policy to partial reads.
      </p>
    </>
  ),
};

export const augmentationPages = [augmentation];
