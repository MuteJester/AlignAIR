import { CodeBlock, Callout } from "../../components/ui";
import { DocTable, type DocPage } from "./doc-kit";

const airrFields: DocPage = {
  slug: "airr-fields",
  title: "AIRR output fields",
  section: "Reference",
  lead: "The field-by-field contract: which columns AlignAIR populates, how they are derived, and how to read them.",
  body: () => (
    <>
      <p>
        <code>alignair predict</code> writes an AIRR-C rearrangement TSV. The guiding principle is honest absence: a
        field is left blank when AlignAIR cannot derive it reliably, rather than filled with a guess. A schema-valid
        record is not the same as a fully-derived one - the assembly-status fields below tell the two apart.
      </p>

      <h2>Calls and coordinates</h2>
      <DocTable
        head={["Field(s)", "How it is derived"]}
        rows={[
          [<><code>sequence_id</code>, <code>sequence</code>, <code>rev_comp</code></>, "See orientation, below."],
          [<code>locus</code>, "Inferred from the model / reference; a multi-locus model attributes each read to one locus."],
          [<>V/D/J <code>_call</code></>, "Top call per gene from the classifier heads (over the embedded reference)."],
          [<>V/D/J <code>_call_set</code></>, "Equivalence set: every allele the read cannot distinguish. AlignAIR extension."],
          [<><code>*_sequence_start/end</code>, <code>*_germline_start/end</code></>, "1-based AIRR coordinates in the coordinate frame described under orientation."],
          [<>V/D/J <code>_cigar</code></>, "CIGAR from the germline reader; a lighter --columns preset falls back to a coordinate-derived CIGAR."],
          [<code>mutation_rate</code>, "Neural estimate of the read's SHM load (AlignAIR extension)."],
        ]}
      />

      <h2>Junction and regions</h2>
      <p>Derived from the assembled alignment; blank on partial records.</p>
      <DocTable
        head={["Field(s)", "How it is derived"]}
        rows={[
          [<><code>junction</code>, <code>junction_aa</code>, <code>junction_length</code>, <code>junction_aa_length</code></>, "CDR3 including the conserved codons; re-derived in read coordinates through the CIGAR when the read has a V/J indel; blank when anchors cannot be placed."],
          [<><code>np1</code>, <code>np2</code>, <code>np1_length</code>, <code>np2_length</code></>, "Non-templated nucleotides V to D and D to J (V to J when the locus has no D)."],
          [<><code>fwr1</code>, <code>cdr1</code>, <code>fwr2</code>, <code>cdr2</code>, <code>fwr3</code>, <code>cdr3</code>, <code>fwr4</code> (and suffixes <code>*_aa</code>, <code>*_start</code>, <code>*_end</code>)</>, "Framework/CDR sequences sliced from the IMGT-gapped alignment, their amino acid translations, and their 1-based start and end alignment coordinates."],
          [<><code>sequence_alignment</code>, <code>germline_alignment</code>, <code>sequence_alignment_aa</code>, <code>germline_alignment_aa</code></>, "Full IMGT-gapped query and germline alignments (both nucleotide and translated amino acids), reconstructed from coordinates and the embedded reference."],
          [<>Per-segment <code>*_sequence_alignment</code>, <code>*_germline_alignment</code>, <code>*_sequence_alignment_aa</code>, <code>*_germline_alignment_aa</code>, <code>*_alignment_start</code>, <code>*_alignment_end</code> (for <code>v/d/j</code>)</>, "Per-segment slices of the stitched alignments (nucleotide and translated amino acids), with their 0-based start and end indices in the canonical sequence alignment."],
          [<>V/D/J <code>_identity</code></>, "Percent identity per segment."],
        ]}
      />
      <Callout kind="note">
        The gapped alignment fields are produced by AlignAIR's own IMGT-gap reconstruction and do not require an external
        aligner. Installing <code>parasail</code> (bundled in <code>[cli]</code>) or selecting the WFA reader changes how
        coordinates and CIGARs are found, not whether these fields are emitted.
      </Callout>

      <h2>Quality fields</h2>
      <DocTable
        head={["Field", "How it is derived"]}
        rows={[
          [<code>productive</code>, <>AIRR productivity, <strong>derived</strong> from <code>vj_in_frame</code> and <code>stop_codon</code>. Blank means unknown, not <code>F</code>.</>],
          [<code>productive_prediction</code>, "The neural productivity head's advisory output (AlignAIR extension). A hint, not a determination."],
          [<code>vj_in_frame</code>, "Junction length is a multiple of 3."],
          [<code>stop_codon</code>, "A stop codon appears in the V-to-J coding frame."],
        ]}
      />

      <h2>Orientation, sequence, and rev_comp</h2>
      <p>
        The emitted <code>sequence</code> and <code>rev_comp</code> follow the AIRR convention: <code>rev_comp=T</code>{" "}
        means the alignment used the reverse complement of the emitted <code>sequence</code>. Cropping happens before
        orientation, so <code>input_sequence</code> is the post-crop, pre-orientation read.
      </p>
      <DocTable
        head={["Input transform", "Emitted sequence", "rev_comp", "Coordinates apply to"]}
        rows={[
          ["Forward", "the (forward) query", "F", <code>sequence</code>],
          ["Reverse complement", "the original query", "T", <code>RC(sequence)</code>],
          ["Complement only", "the forward-frame read", "F", <><code>sequence</code> (original in <code>input_sequence</code>)</>],
          ["Reverse only", "the forward-frame read", "F", <><code>sequence</code> (original in <code>input_sequence</code>)</>],
        ]}
      />
      <p>
        For a reverse-complement read the emitted <code>sequence</code> is the original query and the coordinates apply
        to <code>RC(sequence)</code> - the AIRR / IgBLAST convention. Coordinates never require a double reverse.
      </p>

      <h2>Assembly status and record quality</h2>
      <DocTable
        head={["Field", "Meaning"]}
        rows={[
          [<code>airr_assembly_status</code>, <><code>complete</code> / <code>partial</code> (valid calls, a product such as the junction could not be assembled) / <code>failed</code> (an exception; light fields still emitted).</>],
          [<code>airr_assembly_reason</code>, <>Reason on a partial record: <code>nonproductive_indel</code>, <code>missing_calls_or_coordinates</code>, <code>collapsed_segment</code>, <code>incomplete_alignment</code>.</>],
          [<code>airr_assembly_error</code>, "Stores the exception class or traceback when airr_assembly_status is failed."],
          [<code>segmentation_low_quality</code>, "The V/J segmentation collapsed; coordinates unreliable, assembly partial."],
          [<code>length_cropped</code>, "The input exceeded the model's sequence length window (default: 576 nt) and was cropped before orientation."],
          [<><code>orientation</code>, <code>input_sequence</code></>, "The detected transform and the original post-crop read."],
        ]}
      />
      <h3>A recommended minimum filter</h3>
      <CodeBlock
        lang="python"
        code={`import pandas as pd\n\ntable = pd.read_csv("out.tsv", sep="\\t")\nusable = table[\n    (table["airr_assembly_status"] == "complete")\n    & ~table["segmentation_low_quality"].fillna(False).astype(bool)\n]`}
      />
      <p>Partial and failed records are still emitted (with their calls); select them explicitly for family-level results.</p>

      <h2>Output presets</h2>
      <p>
        AlignAIR provides four column output presets via the <code>--columns</code> parameter:
      </p>
      <DocTable
        head={["Preset", "Description", "Fields Included"]}
        rows={[
          [<code>full</code>, "The default schema. Emits all 78+ fields including full alignments, per-segment slices, and region boundaries.", "All fields in the schema."],
          [<code>core</code>, "Compact output that skips gapped alignment reconstruction for maximum speed, but retains calls and coordinates.", "sequence_id, sequence, rev_comp, locus, v_call, d_call, j_call, c_call, productive, junction, junction_aa, junction_length, and all V/D/J sequence and germline coordinates & CIGARs."],
          [<code>minimal</code>, "Bare minimum columns needed to identify V/D/J gene calls.", "sequence_id, sequence, locus, v_call, d_call, j_call, productive."],
          [<code>airr</code>, "The MiAIRR-minimal required rearrangement columns.", "sequence_id, sequence, rev_comp, productive, v_call, d_call, j_call, sequence_alignment, germline_alignment, junction, junction_aa, v_cigar, d_cigar, j_cigar."],
        ]}
      />

      <h2>Metadata integration and collision safety</h2>
      <p>
        When using the <code>--metadata</code> option to carry study/experiment descriptors into the AIRR TSV:
      </p>
      <ul className="list-disc pl-6 space-y-1.5 my-4">
        <li><strong>Collision Namespacing:</strong> To prevent metadata from clobbering model-derived columns, any metadata column that shares a name with a produced AIRR field is automatically namespaced to <code>meta_&lt;column_name&gt;</code> in the output.</li>
        <li><strong>The <code>c_call</code> Exception:</strong> The constant-region gene call <code>c_call</code> is treated as a protected, fill-only column. If metadata supplies <code>c_call</code>, it will populate the output <code>c_call</code> column only if it was originally blank.</li>
      </ul>

      <h2>Reserved and not-yet-populated</h2>
      <DocTable
        head={["Field(s)", "Status"]}
        rows={[
          [<code>is_contaminant</code>, "Reserved. No contaminant classifier runs by default, so this column is present but always blank. Do not filter on it."],
          [<><code>*_resolved_call</code>, <code>*_call_level</code>, <code>*_set_confidence</code></>, "Reserved for calibrated confidence; populated only by the optional calibration step. Use *_call_set for ambiguity today."],
          [<><code>c_call</code> (from the read)</>, "AlignAIR aligns V(D)J only. Supply the assembler's c_gene via --metadata."],
        ]}
      />
    </>
  ),
};

export const airrFieldsPages: DocPage[] = [airrFields];
