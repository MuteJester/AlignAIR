import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const trainingBatches: DocPage = {
  slug: "training-batches",
  title: "Training batch API",
  section: "Design",
  lead: "Adapt GenAIRR records to the chip's model inputs and training targets, with explicit padding, windows and supervision.",
  body: () => (
    <>
      <p>
        <code>alignair.data.NucleotideBatchAdapter</code> turns a list of simulator records into a
        <code> TrainingBatch</code> of <code>inputs</code> and <code>targets</code>. Target names follow the
        <DocLink to="training-tasks"> training task API</DocLink>; records come from the
        <DocLink to="curriculum-sources"> curriculum source API</DocLink>. The runner builds two adapters per run with
        <code> data.training.prepare_training_data(setup)</code>: one for training batches and one, without orientation
        augmentation, for validation batches.
      </p>
      <H2 id="adapters">Adapt records</H2>
      <CodeBlock lang="python" code={`from alignair.data import NucleotideBatchAdapter

# model.cfg and reference must use the same allele order and counts.
adapter = NucleotideBatchAdapter(reference, model.cfg, augment_orientation=False, overflow="error")
batch = adapter(records)
batch = batch.to(device, non_blocking=True)
outputs = model(batch.inputs)
loss, parts = tasks.loss(outputs, batch.targets)

# Training steps can also unpack the batch.
inputs, targets = batch`} />
      <p>
        <code>NucleotideBatchAdapter(reference, config, device=&quot;cpu&quot;, augment_orientation=None,
        overflow=&quot;prefix&quot;)</code> needs GenAIRR records with query and germline alignment annotations; a
        record that does not satisfy the schema raises <code>TrainingDataError</code> with the record index and id.
        The batch is moved to <code>device</code>. <code>TrainingBatch.pin_memory()</code> supports DataLoader
        pinning before a GPU transfer; <code>token_mask</code> is true for nonzero tokens and <code>lengths</code>
        counts them. Token 0 is padding. The adapter honors <code>config.reject_empty_sequences</code>, which is off by
        default.
      </p>
      <H2 id="inputs">Model inputs</H2>
      <DocTable head={["Input", "Contents"]} rows={[
        [<code>tokenized_sequence</code>, "(B, max_seq_length) token ids, right-padded with 0, after the orientation transform."],
        [<code>orientation</code>, "(B,) the transform applied to each read; the model uses it in place of its own call while training."],
        [<code>{"bank_mask_{gene}"}</code>, "(B, alleles) the genotype the batch is aligned against, one mask for the whole batch: the batch's true alleles plus random others, its size drawn log-uniformly from 1 to every allele."],
        [<><code>{"teacher_coord_{gene}"}</code>, <code>teacher_region</code></>, "The true frame columns and regions, which the chip reads in place of its own while it trains (teacher forcing)."],
        [<><code>{"bank_order_{gene}"}</code>, <code>{"bank_extension_*"}</code></>, "Only when a read comes from a novel allele: the batch's novel rows, compiled on the reference's frames and added to the bound bank for that batch."],
      ]} />
      <p>
        With <code>use_orientation</code>, each training read is put in one of the 4 orientations (forward, reverse
        complement, complement, reverse) drawn uniformly, so three quarters of training reads are not forward;
        <code> augment_orientation=False</code> keeps every read forward, as the validation adapter does. Requesting
        augmentation without the model&apos;s orientation head raises <code>ConfigurationError</code>. Targets stay in
        the forward frame. Validation batches get bank masks drawn the same way, so their scan metrics are measured
        against a random genotype per batch; <code>{"bank_{gene}_size"}</code> reports its mean size.
      </p>
      <H2 id="targets">Training targets</H2>
      <DocTable head={["Target", "Contents"]} rows={[
        [<><code>{"{gene}_start"}</code>, <code>{"{gene}_end"}</code></>, "(B, 1) float32 read positions of each segment, end exclusive."],
        [<code>{"{gene}_supervise"}</code>, "(B,) bool: the segment is in the window and can be scored. An inverted D is not supervised."],
        [<code>{"{gene}_allele"}</code>, "(B, alleles) multi-hot truth set over the reference's alleles."],
        [<><code>region</code>, <code>region_allowed</code></>, "Each base's recombination region, and a bitmask of the regions the read cannot tell apart at a boundary."],
        [<code>{"coord_{gene}"}</code>, "Each base's column in its gene's frame where its alignment maps it; -100 elsewhere."],
        [<code>state_labels</code>, "Per-base edit states (germline, substitution, insertion, deletion), padded with -100."],
        [<><code>v_anchor</code>, <code>j_anchor</code></>, "The read position of the V Cys and J Trp/Phe codon, or max_seq_length (the \"not in read\" slot); with _supervise masks. Only with anchor_heads."],
        [<code>d_presence</code>, "Absent / too short / callable for the D in the window, with a _supervise mask. Only with a D presence head."],
        [<><code>mutation_rate</code>, <code>indel_count</code>, <code>orientation</code></>, "Per-read scalars and the orientation label."],
      ]} />
      <p>
        Anchors and D presence are unscorable, and so unsupervised, when the segment is in the read but its CIGAR is
        missing or disagrees with its read and germline spans. Query spans are validated in <code>data.coordinates</code>; a
        visible segment without germline coordinates raises <code>TrainingDataError</code> (no germline coordinates
        are inferred). When a read comes from a novel allele, the family alleles the read cannot tell from it join
        its truth set.
      </p>
      <H2 id="windows">Window rules</H2>
      <p>
        A read may use <code>max_seq_length - 1</code> positions; the last slot is reserved for a segment end at the
        read&apos;s length. With the default <code>overflow=&quot;prefix&quot;</code> an over-long record keeps its
        beginning: boundary targets and positional labels are clipped to the prefix, a gene entirely outside it gets
        zero coordinates, an empty allele set and a false supervision mask, and an anchor codon the prefix cuts becomes
        &ldquo;not in read&rdquo;. The source record is not modified. <code>overflow=&quot;error&quot;</code> raises
        <code> TrainingDataError</code> (&ldquo;training record exceeds the model window&rdquo;) instead.
      </p>
      <H2 id="prepared">Prepared samples</H2>
      <CodeBlock lang="python" code={`import torch
from alignair.data import PositionLabels, TrainingSample, collate_samples

sample = TrainingSample(
    tokens=encoded_tokens,                                # one-dimensional integer CPU vector
    targets={
        "v_allele": torch.tensor([0.0, 1.0, 1.0]),         # multi-hot over the V alleles
        "v_start": torch.tensor([0.0]),
        "v_end": torch.tensor([6.0]),
        "state_labels": PositionLabels(per_base_states, padding=-100),
    },
    record_id="read-001",
)
batch = collate_samples([sample], max_length=576, reserve_terminal_slot=True)`} />
      <p>
        <code>TrainingSample</code> holds encoded tokens and explicitly typed targets. <code>collate_samples</code>
        stacks dense targets without changing their dtype, right-pads tokens with 0 and pads
        <code> PositionLabels</code> with their declared value. Positional targets must match the token vector&apos;s
        length; target fields, dtypes and dense shapes must agree across samples. Empty reads (rejected by default),
        incompatible fields and reads longer than the available slots raise <code>TrainingDataError</code>
        (&ldquo;encoded read exceeds the available model slots&rdquo;). The collator never truncates and does not
        construct missing labels. <code>NucleotideBatchAdapter</code> builds its samples and collates them this way.
      </p>
    </>
  ),
};

export const batchingPages = [trainingBatches];
