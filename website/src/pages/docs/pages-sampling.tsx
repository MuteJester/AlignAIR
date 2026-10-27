import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const samplePreparation: DocPage = {
  slug: "sample-preparation",
  title: "Sample preparation API",
  section: "Design",
  lead: "Prepare protein-class samples through explicit acceptance policies, and collect source records with bounded consumption and complete outcome counts.",
  body: () => (
    <>
      <p>
        <code>alignair.data.sampling</code> separates source parsing, scientific acceptance, and collection.
        A builder returns <code>SampleResult(sample, reason)</code>. An accepted sample has a mapping
        and no rejection reason; a rejected sample has a declared reason and no mapping. The selected{" "}
        <DocLink to="training-batches">batch adapter</DocLink> turns those mappings into model batches.
      </p>
      <H2 id="protein">Prepare protein-class samples</H2>
      <CodeBlock lang="python" code={`from alignair.reference import build_protein_reference
from alignair.data.tokenizer import AAProteinTokenizer
from alignair.data.sampling import ProteinSampleBuilder, ProteinSamplePolicy

reference = build_protein_reference("HUMAN_IGH_OGRDB")
builder = ProteinSampleBuilder(
    reference.class_maps, AAProteinTokenizer(), max_length=192,
    policy=ProteinSamplePolicy(require_productive=True, allow_stops=False),
)
result = builder(record)
if result.sample is None:
    print(result.reason)
else:
    sample = result.sample`} />
      <p>
        The builder uses the exported <code>sequence_aa</code>, checks its length against the declared
        nucleotide reading frame, maps V/J calls through the supplied protein label maps, and projects
        boundaries onto fully owned codons. Both V and J need observable bounds. Reads must use fewer
        than <code>max_length</code> tokens so the exclusive terminal endpoint has a model slot.
        Tokenization must preserve one residue per position; a nonempty encoding with a different length
        raises <code>TrainingDataError</code>.
      </p>
      <p>
        The default policy requires productivity and rejects stop-containing reads. Curriculum experiments
        that retain those reads explicitly choose <code>require_productive=False, allow_stops=True</code>.
        Both policies use the same label and coordinate assembly. The builder preserves the source record.{" "}
        <DocLink to="reference-labels">Reference and label construction</DocLink> defines the class IDs.
      </p>
      <p>
        Supplying <code>d_index=reference.source.gene("D").index</code> adds D allele targets. Observable,
        known, forward D calls receive their allele ID and <code>d_valid=1</code>. Absent, inverted,
        or unknown D calls receive <code>d_class=-1</code>, zero placeholder coordinates, and
        <code> d_valid=0</code>. Omitting the D vocabulary produces V/J targets only.
      </p>
      <H2 id="rejections">Rejection and failure contracts</H2>
      <DocTable head={["Outcome", "Meaning"]} rows={[
        ["not_productive / stop_codon", "A read fails the selected scientific acceptance policy."],
        ["no_aa / no_frame / translation_length_mismatch", "The exported translation cannot be placed on the declared residue grid."],
        ["unknown_allele / no_bounds", "Required V/J labels or projected boundaries are unavailable."],
        ["empty / too_long", "The encoded read is empty or cannot fit with its terminal endpoint."],
        ["TrainingDataError", "Malformed annotation types, invalid coordinates, or incompatible tokenizer output violate the preparation contract."],
        ["ConfigurationError", "Invalid policies, windows, tokenizers, class maps, or D vocabularies fail before record processing."],
      ]} />
      <p>
        Rejection priority is stable: productivity, translation presence, frame presence, stop policy,
        translation length, class lookup, required bounds, and encoding/window checks. Malformed fields
        raise with available record and field context. Unexpected callback exceptions propagate with
        their original type.
      </p>
      <H2 id="collection">Collect within a source-record budget</H2>
      <CodeBlock lang="python" code={`from collections import Counter
from alignair.data.sampling import (
    collect_samples, check_collection_counts, REJECT_REASONS,
)

counts = Counter()
collection = collect_samples(
    source_records, builder, want=64, budget=20_000,
    counter=counts, rejection_reasons=REJECT_REASONS,
    select=diversity_selector,  # optional: (sample, selected_samples) -> bool
)
check_collection_counts(counts, REJECT_REASONS)
samples = collection.samples
consumed_records = collection.generated`} />
      <p>
        <code>collect_samples</code> consumes at most <code>budget</code> records and stops as soon as
        <code> want</code> samples are selected. Its <code>SampleBuilder</code> protocol supports
        source-specific record types. Custom builders declare their rejection names through
        <code> rejection_reasons</code>. A selector inspects usable samples and the selected sequence;
        declining a sample records a selection outcome. It remains usable for the preparation census.
        Source exhaustion or an insufficient budget raises <code>SampleBudgetExhausted</code> with the
        actual consumed and collected counts.
      </p>
      <H2 id="diversity">Select a cohort with class-diversity floors</H2>
      <CodeBlock lang="python" code={`from alignair.data.sampling import ClassDiversity, collect_samples

diversity = ClassDiversity({"v_class": 3, "j_class": 2})
collection = collect_samples(
    source_records, builder, want=64, budget=20_000,
    rejection_reasons=REJECT_REASONS, select=diversity,
)
diversity.validate(collection.samples)`} />
      <p>
        <code>ClassDiversity</code> accepts any declared sample-field names with positive integer
        minimums. Samples must supply observed, nonnegative integer class IDs for those fields;
        missing labels and unknown IDs such as <code>-1</code> raise <code>TrainingDataError</code>.
        The policy copies its minimums and retains no collection state, so it can be reused with
        independent selected sequences.
      </p>
      <p>
        Until every minimum is met, it accepts a sample when any declared field contributes an unseen
        ID. Afterward it accepts freely to fill the batch. An unseen ID can still be accepted for
        a field whose own minimum is already met; this preserves the protein proof study’s sampling
        rule. A small batch or limited source can therefore fill before every minimum is reached.
        Call <code>validate</code> on the completed cohort. Insufficient diversity raises
        <code> SampleDiversityError</code> with observed and required counts; it is both a
        <code> TrainingDataError</code> and a <code>ValueError</code> for historical assertion callers.
      </p>
      <p>
        The historical <code>diverse_selector</code> and <code>assert_batch_diverse</code> functions
        delegate to this policy with their established V/J minimums. Selection declines remain
        <code> selection_skipped</code> in the collector census. Use this policy before a
        <DocLink to="training-engine">one-batch learning diagnostic</DocLink>; training mechanics
        do not choose the scientific cohort policy.
      </p>
      <H2 id="accounting">Validate preparation and selection counts</H2>
      <CodeBlock lang="text" code={`generated = usable + sum(rejection counts) + preparation_errors
usable = selected + selection_skipped + selection_errors + selection_untracked`} />
      <p>
        Missing zero-valued fields are read as zero. Preparation and selector exceptions increment
        their respective error counters before propagating. Appending to a valid historical census
        whose selection outcomes were never recorded preserves those prior usable rows as
        <code> selection_untracked</code>. The collector does not invent their selection decisions.
        <code>check_collection_counts</code> rejects negative, fractional, or inconsistent counts.
      </p>
      <H2 id="genairr">Use the GenAIRR provider adapter</H2>
      <CodeBlock lang="python" code={`from alignair.data.sampling import collect_usable, check_counter

provider_counts = Counter()
samples, source_info = collect_usable(
    start_seed=5_000_000, budget=20_000, want=64,
    class_maps=reference.class_maps, tokenizer=AAProteinTokenizer(), max_len=192,
    counter=provider_counts,
)
check_counter(provider_counts)`} />
      <p>
        This adapter preserves the existing recombination/S5F source and productive, stop-free policy.
        <code>source_info</code> records the requested start/budget, actual generated count, and consumed
        first/last seed. Every generated record advances that interval, including preparation rejections
        and selection skips. Use separate, bounded ranges for held-out and training sources.
        The historical <code>SeedBudgetExhausted</code> name remains an alias of the shared budget error,
        retaining its existing error code.
      </p>
      <H2 id="ownership">Implementation ownership</H2>
      <p>
        Sampling contracts, protein preparation, generic collection, and the GenAIRR adapter live in
        separate modules under <code>data.sampling</code>. Translation eligibility is shared with the
        record batch adapters through <code>data.projection</code>. Historical <code>aa.data</code>
        imports delegate to these owners. The <DocLink to="curriculum-sources">curriculum source API</DocLink>
        supplies training records. Curriculum and partial-coverage runners use the same base
        builder and retain their explicit scientific policies. Crop augmentation and compatible-label
        transformations live in the <DocLink to="protein-augmentation">protein augmentation API</DocLink>,
        separate from base sample preparation.
      </p>
    </>
  ),
};

export const samplingPages = [samplePreparation];
