import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const curriculumSources: DocPage = {
  slug: "curriculum-sources",
  title: "Curriculum source API",
  section: "Design",
  lead: "Construct reproducible training-record streams from a curriculum mixture, per-reference mutation caps, library flanks and a replaceable provider.",
  body: () => (
    <>
      <p>
        <code>alignair.train.gym.training_records</code> builds the lazy record stream a training run reads.
        <code> CurriculumStreamPolicy</code> chooses the mixture of curriculum components; a record provider (GenAIRR
        by default) simulates each component&apos;s records. Turning records into model inputs and targets belongs to
        the <DocLink to="training-batches">batch adapter</DocLink>.
      </p>
      <H2 id="usage">Choose a source policy</H2>
      <CodeBlock lang="python" code={`from alignair.data.flanks import FlankPolicy
from alignair.train.gym import CurriculumStreamPolicy, training_records

policy = CurriculumStreamPolicy(flanks=FlankPolicy())   # the mix TrainingConfig trains on
records = training_records(("HUMAN_IGH_OGRDB",), policy=policy, seed=17)
first = next(records)                                    # a GenAIRR record mapping`} />
      <p>
        Supply an iterable of GenAIRR dataconfigs or their built-in names. These differ from the ordered
        <code> ReferenceSet</code> used by the <DocLink to="model-setup">model setup API</DocLink>. Policy fields and
        request arguments are validated when constructed; names are resolved, experiments compiled and records drawn
        only when the iterator advances. <code>CurriculumStreamPolicy</code> leaves reads bare by default
        (<code>flanks=None</code>); <code>TrainingConfig</code> trains with <code>FlankPolicy()</code> unless told
        otherwise.
      </p>
      <DocTable head={["Policy field", "Default", "Behavior"]} rows={[
        ["progresses", "(0.3, 0.6, 0.9)", "Finite values in [0, 1]. Each contributes a full-length curriculum component."],
        ["heavy_shm", "0.25", "When positive, adds a full-length component at progress 0.3 with this SHM rate; the effective mutation cap still applies."],
        ["short_boost", "1", "Positive integer: how many times the three amplicon profiles repeat."],
        ["mutation_cap", "None", "Optional cap in [0, 1] on every component's SHM rate, applied per reference; TCR references always use 0."],
        ["mode", "\"auto\"", "auto: one reference, one progress value, heavy_shm=0 and short_boost=1 give a single curriculum component; anything else gives the full mixture. amplicon: always the mixture (progresses may then be empty). Several references always use the mixture."],
        ["indel_form / indel_fraction", "\"weighted\" / (0.05, 0.25)", "The polymerase-indel draw (see the curriculum table). \"uniform\" is the historical form of the 1.0.0 recipe."],
        ["selected_fraction", "0.75", "The share of each full-length component's records drawn under GenAIRR's productive_only(); a fraction k/n with n at most 10. Amplicon records are never selected."],
        ["flanks", "None", "A FlankPolicy puts library flanks around every stream's records."],
        ["novel_alleles", "None", "A NovelAlleles draws part of every stream's reads from synthetic novel V alleles."],
      ]} />
      <H2 id="curriculum">The curriculum</H2>
      <p>
        <code>Curriculum().params(p)</code> maps a progress <code>p</code> in [0, 1] to GenAIRR parameters by linear
        ramps. The training policy uses the weighted indel form; <code>Curriculum</code> itself defaults to
        <code> indel_form=&quot;uniform&quot;</code>.
      </p>
      <DocTable head={["Parameter", "At p = 0", "At p = 1"]} rows={[
        ["S5F somatic hypermutation rate (per base)", "0.005", "0.15"],
        ["End loss, each end (uniform range)", "0 nt", "0 to 40 nt"],
        ["Reads with polymerase indels (weighted form)", "5%", "25%; of those, 1, 2 or 3 indels with weights 0.75, 0.20, 0.05"],
        ["Sequencing error rate (per base)", "0", "0.02 x 0.757 (about 1.5%)"],
        ["Ambiguous (N) bases", "0", "0 to 5"],
      ]} />
      <p>
        The 0.757 factor (<code>SEQUENCING_ERROR_SCALE</code>) keeps the sequencing-error level the chip trained at
        after a GenAIRR change made every replacement base differ from the original. TCR references get no SHM:
        <code> mutation_cap_for</code> caps them at 0 and the experiment builder never adds a mutation step for them.
        The default provider disables D inversion unless the component&apos;s parameters request it. Orientation is
        not a curriculum knob: the batch adapter puts each read in one of 4 orientations, drawn uniformly.
      </p>
      <H2 id="mixture">Components and ordering</H2>
      <p>
        The mixture lists the progress components in order, then the three amplicon profiles per
        <code> short_boost</code> repetition, then the heavy-SHM component. Amplicons use the progress-0.6 parameters
        with their own end loss; they are simulated end loss, not crops of completed records. The default mix has 7
        components.
      </p>
      <DocTable head={["Amplicon component", "5′ end loss", "3′ end loss"]} rows={[
        ["V-anchored", "0–15 nucleotides", "150–350 nucleotides"],
        ["J-anchored", "150–350 nucleotides", "0–15 nucleotides"],
        ["Both-end fragment", "70–175 nucleotides", "70–175 nucleotides"],
      ]} />
      <p>
        Each component has <code>n</code> streams, where <code>selected_fraction</code> is <code>k/n</code>: at 0.75, a
        full-length component interleaves three selected streams and one unselected stream, and an amplicon component
        four unselected streams. The default mix therefore reads 28 streams (7 components of 4). When GenAIRR cannot
        draw productive records from a reference (an incompletely annotated custom reference), the trainer runs with
        no selected records, says so, and lists the reference in <code>train_args[&quot;selection_unavailable&quot;]</code>.
      </p>
      <p>
        Streams, components and reference panels are interleaved in complete round-robin rounds. A round pulls one
        item from each member before emission, so the first request can initialize several streams. If any member
        ends, the mixture ends, and an incomplete final round is discarded.
      </p>
      <H2 id="seeds">Seed blocks</H2>
      <p>
        GenAIRR draws record <code>k</code> of a stream of seed <code>s</code> from seed <code>s + k</code>, so streams
        with nearby seeds would repeat each other&apos;s draws. Each stream therefore draws from its own block of
        2<sup>32</sup> consecutive seeds, placed by a SHA-256 of the run seed, the purpose
        (<code>&quot;training&quot;</code>, <code>&quot;validation&quot;</code> or <code>&quot;probe&quot;</code>), the
        start step, the panel, the component and the stream name (<code>train.gym.streams.seed_block</code>). Each
        purpose has its own quarter of the seed range, so validation and training streams never share a block. Within
        one call the blocks are checked to be distinct. A resumed run passes its <code>start_step</code> and draws
        blocks of its own.
      </p>
      <H2 id="loci">Mutation caps per reference</H2>
      <CodeBlock lang="python" code={`import GenAIRR as ga
from alignair.train.gym.streams import effective_mutation_caps

caps = effective_mutation_caps((ga.HUMAN_IGH_OGRDB, ga.HUMAN_TCRB_IMGT), mutation_cap=None)
# {'HUMAN_IGH_OGRDB': None, 'HUMAN_TCRB_IMGT': 0.0}`} />
      <p>
        The trainer records these caps in <code>train_args[&quot;effective_mutation_caps&quot;]</code>, and the
        exported model card lists them. Names come from the built-in dataconfig name where available, then a custom
        name or the receptor descriptor; repeated and colliding names get distinct suffixes so every panel remains
        represented.
      </p>
      <H2 id="flanks">Library flanks</H2>
      <p>
        GenAIRR simulates the rearrangement alone, so its reads start in V and end in J. A <code>FlankPolicy</code>
        adds what real reads carry around V(D)J and moves every read coordinate with it. The defaults: a random-base
        5′ flank on 40% of reads (1–140 nt), masked to N on 40% of those together with the next 0–12 read bases (a
        primer over the first V bases), a random-base 3′ flank on 30% of reads (1–100 nt), and on 30% of unmasked
        5′ flanks each flank base turned to N at a rate drawn from 0–15%. A read grows to at most 576 bases. Flanks
        are drawn after the provider, from the stream&apos;s own seed block.
      </p>
      <H2 id="providers">Replace the provider</H2>
      <CodeBlock lang="python" code={`def record_source(dataconfig, parameters, seed):
    # Implement RecordSource; honor the component parameters and seed.
    return custom_provider.stream_records(dataconfig, parameters=parameters, seed=seed)

records = training_records(
    provider_references, policy=policy, seed=17, record_source=record_source,
)`} />
      <p>
        A <code>RecordSource</code> returns an iterable of fresh record mappings for one reference, parameter
        dictionary and seed. References must expose <code>metadata.chain_type</code>; otherwise
        <code> ConfigurationError</code> (&ldquo;source references must supply receptor metadata&rdquo;). An injected
        provider with explicit references does not require GenAIRR. <code>run_training(..., record_source=...)</code>
        replaces the provider of every training and validation stream; with <code>data_workers</code> above 0 it runs in
        worker processes and must be picklable.
      </p>
      <p>
        Record reading runs under the <code>curriculum_source</code> and <code>curriculum_component</code> error
        stages with the stream&apos;s seed: AlignAIR errors created there carry that context, and other exceptions keep
        their type. The stream does not save or restore its position: a resumed run starts new streams from its own
        seed blocks.
      </p>
    </>
  ),
};

export const streamPages = [curriculumSources];
