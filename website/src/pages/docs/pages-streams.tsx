import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const curriculumSources: DocPage = {
  slug: "curriculum-sources",
  title: "Curriculum source API",
  section: "Design",
  lead: "Construct reproducible training-record streams with explicit mixtures, per-reference mutation policies, and replaceable providers.",
  body: () => (
    <>
      <p>
        <code>alignair.train.gym.training_records</code> assembles native curriculum records for both
        nucleotide and protein runners. <code>CurriculumStreamPolicy</code> chooses the parameter
        mixture; a record provider supplies the simulator or another implementation. Tokenization,
        label mapping, and acceptance remain with the
        <DocLink to="sample-preparation"> sample builder</DocLink> and
        <DocLink to="training-batches"> batch adapter</DocLink>.
      </p>
      <H2 id="usage">Choose a source policy</H2>
      <CodeBlock lang="python" code={`from alignair.train.gym import CurriculumStreamPolicy, training_records

policy = CurriculumStreamPolicy(
    progresses=(0.3, 0.6, 0.9), heavy_shm=0.25,
    short_boost=1, mutation_cap=None, mode="auto",
)
records = training_records(("HUMAN_IGH_OGRDB",), policy=policy, seed=17)

# Protein runners retain their established always-amplicon source policy.
protein_records = training_records(
    ("HUMAN_IGH_OGRDB",), policy=CurriculumStreamPolicy(mode="amplicon"),
    seed=17, translated_only=True,
)`} />
      <p>
        Supply an iterable of provider references, such as GenAIRR dataconfigs, or their built-in names.
        These differ from the ordered <code>ReferenceSet</code> used by the
        <DocLink to="model-setup"> model setup API</DocLink>. Policy fields and request arguments are
        validated when constructed. Names are resolved, experiments compiled, and records read only
        when the iterator advances. The default provider permits curatable reference data and
        disables D inversion unless component parameters explicitly request it.
      </p>
      <DocTable head={["Policy", "Behavior"]} rows={[
        ["mode=auto", "For one reference, one progress value, heavy_shm=0, and short_boost=1, use one curriculum stream. Otherwise use the amplicon mixture."],
        ["mode=amplicon", "Always use the amplicon mixture. Empty progresses can deliberately omit full-length curriculum components."],
        ["Multiple reference panels", "Always use each panel’s amplicon mixture, including in auto mode."],
        ["progresses", "Finite values in [0,1]. Each contributes a full-length curriculum component."],
        ["heavy_shm", "Adds a full-length high-mutation component when positive; the effective mutation cap still applies."],
        ["short_boost", "Positive integer repeating the three amplicon profiles with distinct component offsets."],
        ["mutation_cap", "Optional finite value in [0,1]. Applied per reference; TCR references always use zero."],
      ]} />
      <H2 id="mixture">Components and ordering</H2>
      <p>
        Components appear in progress order, followed by three moderate-background amplicon profiles
        per short-read repetition, then the optional heavy-SHM component. Profiles use native simulator
        end loss. They do not crop completed records or infer replacement coordinates, and are separate
        from <DocLink to="protein-augmentation">protein crop augmentation</DocLink>.
      </p>
      <DocTable head={["Amplicon component", "5′ end loss", "3′ end loss"]} rows={[
        ["V / forward anchored", "0–15 nucleotides", "150–350 nucleotides"],
        ["J anchored", "150–350 nucleotides", "0–15 nucleotides"],
        ["Both-end fragment", "70–175 nucleotides", "70–175 nucleotides"],
      ]} />
      <p>
        Component <code>i</code> uses <code>seed + i</code>; reference panel <code>j</code> starts at
        <code> seed + 1013*j</code>. Components and panels are interleaved in complete round-robin
        rounds. A round pulls one item from each member before emission, so the first request can
        initialize several streams. If any member ends, the mixture ends. An incomplete final round
        is discarded, including any items already read from earlier members in that round.
      </p>
      <H2 id="loci">Locus labels and mutation provenance</H2>
      <p>
        Mixed-panel records receive a numeric <code>chain_type</code> class for their distinct normalized
        locus, in reference order. Two IGH panels share a class, matching the model’s ordered locus
        schema. Tagging copies the mapping and leaves the provider’s original record unchanged.
        Repeated amplicon parameter dictionaries are independent as well.
      </p>
      <CodeBlock lang="python" code={`import GenAIRR as ga
from alignair.train.gym.streams import effective_mutation_caps

caps = effective_mutation_caps(
    (ga.HUMAN_IGH_OGRDB, ga.HUMAN_TCRB_IMGT), mutation_cap=None,
)
# IGH retains its curriculum rates; TCRB is capped at 0.0.`} />
      <p>
        Receptor detection is shared with experiment construction, so simulation and cap provenance
        use the same decision. Names come from materialized provider names where available, then
        custom names or receptor descriptors. Repeated and colliding names receive distinct suffixes
        so every panel remains represented.
      </p>
      <H2 id="translation">Protein eligibility</H2>
      <p>
        <code>translated_only=True</code> checks the exported protein length against the junction-derived
        frame. It does not require productivity or reject stops; those are sample-builder policies.
        Malformed annotations raise <code>TrainingDataError</code>. The filter imposes no raw-record
        rejection budget. Use a bounded provider or the
        <DocLink to="sample-preparation"> bounded collection API</DocLink> when consumption must be
        limited. Optimizer-step limits do not bound attempts hidden inside a filtering source.
      </p>
      <H2 id="providers">Replace the provider</H2>
      <CodeBlock lang="python" code={`def record_source(dataconfig, parameters, seed):
    # Implement RecordSource; honor the component policy and seed.
    return custom_provider.stream_records(dataconfig, parameters=parameters, seed=seed)

records = training_records(
    provider_references, policy=policy, seed=17, record_source=record_source,
)`} />
      <p>
        A <code>RecordSource</code> returns an iterable of fresh record mappings for one reference,
        parameter dictionary, and seed. Explicit references must expose <code>metadata.chain_type</code>.
        An injected provider with explicit references does not require GenAIRR. Already prepared samples
        can go directly to <DocLink to="training-engine">training_iterations</DocLink>.
      </p>
      <p>
        Operations supply scoped stage and seed diagnostics. AlignAIR errors created there retain
        that context; exceptions keep their original type. The scope ends before records reach the
        runner. The API does not restore checkpoints or change resume seeds. Nucleotide and protein
        runners now use this owner, and their old private stream implementations have been removed.
      </p>
    </>
  ),
};

export const streamPages = [curriculumSources];
