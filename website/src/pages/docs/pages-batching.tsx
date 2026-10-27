import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const trainingBatches: DocPage = {
  slug: "training-batches",
  title: "Training batch API",
  section: "Design",
  lead: "Adapt source records to one model-input and target contract, with explicit padding, coordinates, and supervision.",
  body: () => (
    <>
      <p>
        The adapters in <code>alignair.data</code> return a <code>TrainingBatch</code> containing
        <code> inputs</code> and <code>targets</code>. Inputs use <code>tokenized_sequence</code>;
        target names follow the <DocLink to="training-tasks">training task API</DocLink>.
        Choose an adapter by the source annotations and objective. Alphabet alone does not determine
        whether gene labels are allele sets or categorical classes.
        The <DocLink to="sample-preparation">sample preparation API</DocLink> describes source filtering
        and bounded collection before categorical batching.
      </p>
      <H2 id="adapters">Choose a source adapter</H2>
      <DocTable head={["Adapter", "Source and target semantics"]} rows={[
        ["NucleotideBatchAdapter(reference, config)", <>Simulator nucleotide records with query and germline
          alignment annotations. Produces allele multi-hot targets, nucleotide boundaries, edit-state labels,
          auxiliary targets, and optional orientation/locus labels. Requires multilabel classification.</>],
        ["ProteinAlleleBatchAdapter(reference, config)", <>Simulator records containing nucleotide coordinates,
          a junction-derived frame, and an exported protein sequence. Projects boundaries to complete codons
          while retaining the full nucleotide allele vocabulary. Requires multilabel classification and
          disabled nucleotide orientation correction.</>],
        ["CategoricalBatchAdapter.from_config(config)", <>Prepared samples containing encoded tokens,
          integer gene classes, boundaries, and optional validity masks or compatible class sets.
          The same contract can represent either alphabet.</>],
      ]} />
      <CodeBlock lang="python" code={`from alignair.data import NucleotideBatchAdapter

# model.cfg and reference must use the same allele order and counts.
adapter = NucleotideBatchAdapter(
    reference, model.cfg, augment_orientation=False, overflow="error",
)
batch = adapter(records)
batch = batch.to(device, non_blocking=True)
outputs = model(batch.inputs)
loss, parts = tasks.loss(outputs, batch.targets, task_weights)

# Existing training steps can also unpack the batch.
inputs, targets = batch`} />
      <p>
        Adapters prepare CPU tensors by default; pass <code>device=</code> or move the completed batch.
        <code>TrainingBatch.pin_memory()</code> supports DataLoader pinning before a GPU transfer.
        <code>token_mask</code> is true for nonzero tokens and <code>lengths</code> counts them.
        Token zero remains padding. Model construction and boundary modes are defined in the
        <DocLink to="model-composition">composition API</DocLink>.
      </p>
      <p>
        The nucleotide record adapter honors <code>config.reject_empty_sequences</code>. Historical NT
        configurations permit simulator records with an empty observed read; their gene supervision is
        absent. Enable the flag to reject those records. Prepared sample collation rejects empty reads
        by default, and protein record adaptation requires a nonempty exported translation.
      </p>
      <H2 id="windows">Window and coordinate rules</H2>
      <p>
        Record adapters default to <code>overflow="prefix"</code>, retaining the beginning of a read
        when it exceeds the configured window. They clip boundary targets and positional labels to that
        prefix. A gene entirely outside the prefix receives zero placeholder coordinates and allele labels,
        with its supervision mask false. The source record is not modified. Use
        <code> overflow="error"</code> when a study must reject overlength records, or crop the source
        with matching annotations before adapting it.
      </p>
      <p>
        Half-open boundary models reserve a terminal slot, so a read must use fewer than
        <code> max_seq_length</code> token slots. Record adapters account for this in their prefix policy;
        prepared categorical samples must already fit. Without a fixed window,
        <code> collate_samples(..., reserve_terminal_slot=True)</code> adds a trailing padding slot.
        The generic collator never truncates. Existing
        center-padded CSV samples retain their padding and shifted coordinates; collation does not shift
        them again.
      </p>
      <p>
        Query spans are validated in <code>data.coordinates</code>. Nucleotide edit-state construction
        also requires the actual germline spans for visible segments; missing truth is an error.
        Protein projection needs only query spans. A residue belongs to a gene only when all three
        nucleotides in its codon belong to that gene; boundary-straddling codons are excluded from that
        gene’s projected interval.
      </p>
      <p>
        Protein allele adaptation requires an exported translation whose length agrees with the declared
        junction frame. <code>data.adapters.translated_record_usable</code> applies the same eligibility rule
        in simulator runners. Some source records fail this check and are excluded before batching.
        This is a length/frame check, not a sequence-identity check; it neither guesses an alternative
        frame nor replaces the exported translation.
      </p>
      <H2 id="prepared">Prepare categorical and custom samples</H2>
      <CodeBlock lang="python" code={`from alignair.data import CategoricalBatchAdapter

sample = {
    "tokens": encoded_tokens,             # one-dimensional integer CPU vector
    "v_class": 0, "v_start": 0, "v_end": 4,
    "j_class": 1, "j_start": 5, "j_end": 8,
    "v_compatible": {0, 2},              # optional set of class IDs
    "j_valid": 0.0,                      # exclude J supervision for this sample
}
adapter = CategoricalBatchAdapter.from_config(model.cfg)
batch = adapter([sample])`} />
      <p>
        Class indices become <code>torch.long</code>; boundaries, validity fields, and compatible-set
        membership become <code>torch.float32</code>. Each gene with compatible sets needs that annotation
        on every sample in the batch, plus explicit vocabulary counts. <code>from_config</code> supplies
        those counts. Validity fields are preserved for V, D, and J; an omitted validity field defaults
        to one when other samples supply it. Prepared D labels retain the historical default
        <code> d_valid=1</code>.
      </p>
      <CodeBlock lang="python" code={`import torch
from alignair.data import PositionLabels, TrainingSample, collate_samples

sample = TrainingSample(
    tokens=encoded_tokens,
    targets={
        "v_class": torch.tensor(0, dtype=torch.long),
        "v_start": torch.tensor([0.0], dtype=torch.float32),
        "v_end": torch.tensor([4.0], dtype=torch.float32),
        "state_labels": PositionLabels(per_token_states, padding=-100),
    },
    record_id="read-001",
)
batch = collate_samples([sample], max_length=128)`} />
      <p>
        <code>TrainingSample</code> supplies already encoded tokens and explicitly typed targets.
        <code>collate_samples</code> stacks dense targets without changing their dtype, right-pads tokens
        with zero, and pads <code>PositionLabels</code> with their declared ignore value. Positional
        targets must match the source token-vector length. Target fields, dtypes, and dense shapes must
        agree across samples. Empty reads, incompatible fields, and overlength samples raise
        <code> TrainingDataError</code> with available record or field context. The collator does not
        construct missing labels or check a custom task’s scientific assumptions.
      </p>
      <H2 id="ownership">Shared implementation and compatibility</H2>
      <p>
        Tokenizers, translation and codon projection, coordinate validation, allele-set encoding, and
        nucleotide target construction live in <code>alignair.data</code>. Historical AA tokenizer,
        translation, and batch functions, plus the trainer’s <code>build_batch</code>, delegate to these
        implementations. The separate stage-one collator has been removed; its caller uses the categorical
        adapter with model vocabulary counts. Older <code>collate_aa</code> calls without compatible sets
        remain supported; compatible sets require an explicitly configured adapter.
      </p>
      <p>
        <code>train.gym.gym_collate</code> still builds the simulator’s intermediate ground-truth view,
        which includes germline and teacher fields beyond model targets. It shares vector padding and
        allele encoding with the adapters. Training schedules, experiment selection, and study-specific
        sample construction remain with their runners; reusable batch assembly belongs here.
      </p>
    </>
  ),
};

export const batchingPages = [trainingBatches];
