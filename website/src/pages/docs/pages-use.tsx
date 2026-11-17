import type { ReactNode } from "react";
import { CodeBlock, Callout } from "../../components/ui";
import CodeTabs from "../../components/CodeTabs";
import { DocTable, DocLink, H2, H3, type DocPage } from "./doc-kit";
import ModelTable from "../../components/data/ModelTable";
import ColumnTable from "../../components/data/ColumnTable";

const SECTION = "Use on your data";

const OGRDB_SETS = "https://ogrdb.airr-community.org/germline_sets/Homo%20sapiens";
const OGRDB_GENES = "https://ogrdb.airr-community.org/gene_table";
const AIRRFLOW_USAGE = "https://nf-co.re/airrflow/latest/docs/usage/";
const AIRRFLOW_PARAMETERS = "https://nf-co.re/airrflow/latest/parameters/";

function Ext({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noreferrer" className="text-brand-600 underline">
      {children}
    </a>
  );
}

/* ------------------------------------------------------------------------------------------------------------------ */
/* Choose a model                                                                                                     */
/* ------------------------------------------------------------------------------------------------------------------ */
const chooseModel: DocPage = {
  slug: "choose-a-model",
  title: "Choose a model",
  section: SECTION,
  lead: "Pick the pretrained model for your locus, know which allele names it writes, and get it ready to run offline.",
  body: () => (
    <>
      <p>
        AlignAIR has one pretrained human model per locus. Each calls alleles from its own germline set, listed here
        with its allele counts. Copy an id with the button beside it.
      </p>
      <ModelTable />

      <H2 id="pick-by-locus">Pick by locus</H2>
      <p>
        A model calls one locus, so use the model of the locus your reads come from: heavy chains with{" "}
        <code>alignair4-igh-human</code>, beta chains with <code>alignair4-trb-human</code>, and so on. The light chains
        are two loci with a model each: kappa (<code>alignair4-igk-human</code>) and lambda (
        <code>alignair4-igl-human</code>).
      </p>
      <p>
        A library that holds several loci, such as the heavy and light contigs of one 10x run, is aligned once per
        locus, each part with its own model. Split the reads by locus first when you can (a 10x{" "}
        <code>filtered_contig_annotations.csv</code> names each contig's locus in its <code>chain</code> column). A
        model calls only alleles of its own locus, so the calls it gives a read of another locus mean nothing.
      </p>
      <CodeTabs
        cli={`for locus in igh igk igl; do
  alignair predict --model alignair4-\${locus}-human --input contigs_\${locus}.fasta --out \${locus}.tsv
done`}
        python={`from alignair import Aligner, read_sequences

for locus in ("igh", "igk", "igl"):
    aligner = Aligner.from_pretrained(f"alignair4-{locus}-human")
    ids, reads, info = read_sequences(f"contigs_{locus}.fasta")
    aligner.predict(reads, ids=ids).write_airr(f"{locus}.tsv")`}
      />

      <H2 id="what-a-model-contains">What a model contains</H2>
      <p>A model is one file. It holds:</p>
      <ul>
        <li>
          what it learned from simulated reads of its locus: full-length reads and fragments, in any orientation,
          with somatic hypermutation on the IG loci. It reads at most 575 nt of a read;
        </li>
        <li>
          its germline set: every V, D and J allele with its IMGT-gapped form and the position of its conserved
          anchor (the V Cys and J Trp/Phe codons). The IG models carry OGRDB's human sets (for IGH, OGRDB V8), the
          TCR models IMGT's. Every call is an allele of this set, unless you give another set with{" "}
          <code>--reference</code>;
        </li>
        <li>a card: its id, version, germline set, allele counts and how long it trained.</li>
      </ul>
      <p>
        It aligns single reads. Paired R1/R2 reads are merged first, and BAM or CRAM files are not read. The models
        are released under GPL-3.0-or-later, as the code is. Each model's full card is on{" "}
        <DocLink to="model-cards">Model cards</DocLink>.
      </p>

      <H2 id="allele-names">Allele names: OGRDB and IMGT</H2>
      <p>The IG models call alleles by their OGRDB labels, the TCR models by their IMGT names:</p>
      <ul>
        <li>
          <strong>IGH:</strong> V alleles are labelled <code>IGHVF…-G…</code>. In <code>IGHVF6-G21*07</code>,{" "}
          <code>F6</code> is the family and <code>G21</code> the gene group. IGH D and J alleles keep IMGT-style names
          (<code>IGHD3-10*01</code>, <code>IGHJ6*02</code>).
        </li>
        <li>
          <strong>IGK and IGL:</strong> OGRDB labels of their own, for V and J alike, such as{" "}
          <code>IGKV1-5TCS*03</code>, <code>IGKJ2-27FL*01</code>, <code>IGLV6-A5V2*05</code> and{" "}
          <code>IGLJ2/3-HDWH*01</code>.
        </li>
        <li>
          <strong>TRA, TRB, TRD, TRG:</strong> IMGT names (<code>TRBV20-1*01</code>).
        </li>
      </ul>
      <p>
        So a table from an IG model will not join on <code>v_call</code> with one IgBLAST wrote against IMGT's set, a
        gene-usage plot shows OGRDB labels, and a genotype written in IMGT names is refused (
        <code>ALIGNAIR_GENOTYPE_ALLELE_NOT_FOUND</code>). You have two ways to work with IMGT-named data:
      </p>
      <ul>
        <li>
          <strong>Look your alleles up in OGRDB.</strong> OGRDB publishes the human germline sets at{" "}
          <Ext href={OGRDB_SETS}>ogrdb.airr-community.org/germline_sets</Ext> (AIRR JSON, gapped and ungapped FASTA),
          with gene tables at <Ext href={OGRDB_GENES}>ogrdb.airr-community.org/gene_table</Ext>. Find the OGRDB name of
          each allele you need there, and write your genotype or join table with those names.
        </li>
        <li>
          <strong>Call with IMGT's set instead.</strong> Bind the IMGT germline set itself with{" "}
          <code>--reference</code>; the calls then carry its names. See{" "}
          <DocLink to="own-reference">Use your own germline reference</DocLink>.
        </li>
      </ul>

      <H2 id="download-and-cache">First use, the cache and offline use</H2>
      <p>
        The first run with a model id downloads the model from Hugging Face, checks its SHA-256 and keeps it in the
        cache: <code>~/.cache/alignair</code> on Linux, <code>~/Library/Caches/alignair</code> on macOS. Set{" "}
        <code>ALIGNAIR_CACHE_DIR</code> to put it elsewhere, a shared directory on a cluster for example. No login is
        needed. Later runs read the cached file. (A model from a Hugging Face repo, <code>hf://org/repo</code>, public
        or private, needs <code>pip install "alignair[hub]"</code>; the ids above do not.)
      </p>
      <p>
        To run where there is no internet (a compute node, a plane), download the model ahead on a machine that has
        it, then run with <code>--offline</code>, which never touches the network:
      </p>
      <CodeTabs
        cli={`alignair models get alignair4-igh-human
alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv --offline`}
        python={`from alignair import Aligner

aligner = Aligner.from_pretrained("alignair4-igh-human", offline=True)   # the cache only`}
      />
      <p>
        <code>alignair4-igh-human@&lt;version&gt;</code> pins a version (<code>alignair models list</code> shows the
        latest). Every run records the model id, version and file checksum it used in <code>out.tsv.run.json</code>,
        beside the table. More on{" "}
        <DocLink to="cache-and-environment">Cache, offline and environment variables</DocLink>.
      </p>

      <H2 id="list-and-inspect">List the models and read a model's card</H2>
      <p>
        <code>alignair models list</code> lists every model id with its latest version and whether it is installed,
        available, or has an update. <code>alignair info</code> prints a model's card without loading it; given an id,
        it downloads the model first if it is not cached:
      </p>
      <CodeBlock
        title="shell (output abridged)"
        code={`$ alignair info alignair4-igh-human
  alleles:   V=198 D=33 J=7   params=314812
  reference: HUMAN_IGH_OGRDB (BCR_HEAVY, Human)`}
      />
      <p>
        The card also shows when the model was made, how many steps it trained and a fingerprint of its allele order.{" "}
        <code>alignair info alignair4-igh-human --json</code> prints all of it as JSON.
      </p>

      <H2 id="nothing-fits">Nothing fits?</H2>
      <DocTable
        head={["Your case", "What to do"]}
        rows={[
          [
            "Your locus has a model, but your germline set differs: novel alleles, a curated set, IMGT's names",
            <>
              Bind your germline FASTA with <code>--reference</code>; no training. See{" "}
              <DocLink to="own-reference">Use your own germline reference</DocLink>.
            </>,
          ],
          [
            "You want calls only from one donor's alleles",
            <>
              Give the donor's genotype with <code>--genotype</code>. See{" "}
              <DocLink to="donor-genotype">Align with a donor genotype</DocLink>.
            </>,
          ],
          [
            "Another species, or a locus no model covers",
            <>
              Train a model from your germline set. Start at <DocLink to="do-i-need-to-train">Do I need to train?</DocLink>
            </>,
          ],
        ]}
      />
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Prepare your input                                                                                                 */
/* ------------------------------------------------------------------------------------------------------------------ */
const prepareInput: DocPage = {
  slug: "prepare-input",
  title: "Prepare your input",
  section: SECTION,
  lead: "What AlignAIR reads, what it does with read ids and unusable reads, and how to give it a table or a stream.",
  body: () => (
    <>
      <H2 id="formats">Formats</H2>
      <p>
        <code>--input</code> takes FASTA (<code>.fasta</code>, <code>.fa</code>, <code>.fna</code>), FASTQ (
        <code>.fastq</code>, <code>.fq</code>), a CSV or TSV table with a sequence column, or plain text with one read
        per line; any of them gzipped (<code>.gz</code>). The format comes from the file name, else from the first
        line. FASTQ qualities are read but not used.
      </p>
      <CodeTabs
        cli={`alignair predict --model alignair4-igh-human --input reads.fastq.gz --out out.tsv`}
        python={`from alignair import Aligner, read_sequences

aligner = Aligner.from_pretrained("alignair4-igh-human")
ids, reads, info = read_sequences("reads.fastq.gz")
aligner.predict(reads, ids=ids).write_airr("out.tsv")`}
      />
      <p>
        AlignAIR aligns single reads, full-length or fragments, in any orientation: it finds the orientation itself
        and reports it in <code>rev_comp</code> and <code>orientation</code>. Merge paired R1/R2 reads before aligning
        (pRESTO's AssemblePairs, fastp or vsearch, for example). BAM and CRAM files are not read.
      </p>

      <H2 id="read-ids">Read ids</H2>
      <p>
        A read's id is the first word of its FASTA or FASTQ header, or a table's id column. A read without one gets{" "}
        <code>seq</code> and its position in the file, counted from 0 (<code>seq7</code>). Every read is kept, even
        when its id repeats: the second <code>read01</code> becomes <code>read01_dup1</code>, the third{" "}
        <code>read01_dup2</code>. For a FASTA whose reads are named read01, read02 and read01 again, the first column
        of the table is:
      </p>
      <CodeBlock
        lang="text"
        title="output"
        code={`sequence_id
read01
read02
read01_dup1`}
      />
      <p>
        A <DocLink to="metadata-join">metadata join</DocLink> still matches both reads to the table's{" "}
        <code>read01</code> row, because it uses the id as the input file gave it.
      </p>

      <H2 id="dropped-reads">Reads that are dropped</H2>
      <p>
        Each read is upper-cased, and the IUPAC ambiguity codes (R, Y, S, W, K, M, B, D, H, V) are read as N. Any other
        character (a gap, a digit, punctuation) is unusable: it is read as N too, but a read in which more than 20% of
        the characters are unusable is dropped, and so is an empty read. The console line counts them, and{" "}
        <code>--rejects-out</code> writes them to a table with the reason:
      </p>
      <CodeTabs
        cli={`$ alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv \\
    --rejects-out rejects.tsv
aligned 3 reads (2 dropped) -> out.tsv
$ cat rejects.tsv
sequence_id	position	reason	sequence
junk1	3	ambiguous	NNNN----????!!!!ACGTACGTXXXXXXXXXX
empty1	4	empty`}
        python={`from alignair import read_sequences

ids, reads, info = read_sequences("reads.fasta", collect_rejects=True)
print(info["n_dropped"])     # 2
print(info["rejects"][0])    # {'sequence_id': 'junk1', 'position': 3, 'reason': 'ambiguous', 'sequence': '...'}`}
      />
      <p>
        <code>position</code> counts from 0. If every read is dropped, the run stops with{" "}
        <code>[ALIGNAIR_INPUT_INVALID] no valid reads in reads.fasta</code>: check the file's format first.
      </p>

      <H2 id="long-reads">Reads longer than 575 nt</H2>
      <p>
        A model reads at most 575 nt. A longer read is aligned on its first 575 nt, as the read is written in the file,
        and its row says <code>length_cropped</code> = <code>T</code>. The cut happens before AlignAIR orients the read,
        so which end is lost depends on the read's orientation:
      </p>
      <ul>
        <li>
          a <strong>forward</strong> read loses its 3' end. When much sequence comes before the V (a long leader or
          primer), the J and the junction can be lost;
        </li>
        <li>
          a <strong>reverse-complemented</strong> read loses the 5' end of the molecule: the start of the V. Its J and
          junction are kept, and its V alignment starts further into the germline.
        </li>
      </ul>
      <p>
        So trim long reads to the V(D)J region before aligning: remove the constant region, leader and primers in
        your preprocessing. After a run, rows with <code>length_cropped</code> = <code>T</code> are the reads to check.
      </p>

      <H2 id="stdin">Read from standard input</H2>
      <p>
        <code>--input -</code> reads standard input, so AlignAIR can sit at the end of a pipe. The format comes from
        the first line (<code>&gt;</code> for FASTA, <code>@</code> for FASTQ).
      </p>
      <CodeBlock code={`zcat run1/*.fastq.gz | alignair predict --model alignair4-igh-human --input - --out run1.tsv`} />
      <p>
        <code>--infer-genotype</code> reads its input twice, so it needs a file, not <code>-</code>. In Python, pass
        the reads as a list of strings to <code>predict</code>.
      </p>

      <H2 id="tables">Read a table</H2>
      <p>
        In a CSV or TSV table, AlignAIR takes the sequence from a column named <code>sequence</code>, <code>seq</code>,{" "}
        <code>nucleotide</code> or <code>read</code> (any case), and the id from one named <code>sequence_id</code>,{" "}
        <code>id</code> or <code>name</code>. Name other columns with <code>--sequence-column</code> and{" "}
        <code>--id-column</code>. A <code>.tsv</code> file, or one whose header has a tab, is read as tab-separated;
        any other as comma-separated.
      </p>
      <CodeTabs
        cli={`$ head -2 reads.csv
read_id,nt,sample
read01,CAGGTGCAGCTGGTGGAGTCTGAGAGAGGCTTGG...,S1
$ alignair predict --model alignair4-igh-human --input reads.csv --out out.tsv \\
    --sequence-column nt --id-column read_id
aligned 3 reads (0 dropped) -> out.tsv`}
        python={`from alignair import Aligner, read_sequences

aligner = Aligner.from_pretrained("alignair4-igh-human")
ids, reads, info = read_sequences("reads.csv", seq_column="nt", id_column="read_id")
aligner.predict(reads, ids=ids).write_airr("out.tsv")`}
      />
      <p>
        Without <code>--sequence-column</code>, this table is refused:{" "}
        <code>
          [ALIGNAIR_INPUT_INVALID] no sequence column in reads.csv; expected one of ('sequence', 'seq', 'nucleotide',
          'read') or select the sequence column explicitly
        </code>
        . The table's other columns are not carried into the output; to carry them, give the same table to{" "}
        <code>--metadata</code> as well (<DocLink to="metadata-join">Join sample and cell metadata</DocLink>).
      </p>
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Align with a donor genotype                                                                                        */
/* ------------------------------------------------------------------------------------------------------------------ */
const donorGenotype: DocPage = {
  slug: "donor-genotype",
  title: "Align with a donor genotype",
  section: SECTION,
  lead: "Call each read only from the alleles one donor carries, from a genotype you have or one AlignAIR infers from the reads.",
  body: () => (
    <>
      <p>
        A genotype narrows the alleles a call can come from: calls the donor cannot carry go away, and so do ties with
        alleles the donor lacks. It never adds an allele; for an allele the model's germline set does not hold, see{" "}
        <DocLink to="own-reference">Use your own germline reference</DocLink>.
      </p>

      <H2 id="genotype-file">The genotype file</H2>
      <p>
        A genotype is a YAML or JSON file keyed by segment, <code>V</code>, <code>D</code> and <code>J</code> (either
        case), each listing allele names of the model's germline set. A segment the file does not name keeps all its
        alleles. With the IG models, the names are OGRDB's labels (see{" "}
        <DocLink to="choose-a-model?h=allele-names">Allele names</DocLink>).
      </p>
      <CodeBlock
        lang="yaml"
        title="donor.yaml"
        code={`V:
  - IGHVF10-G50*02
  - IGHVF10-G42*01
  - IGHVF3-G12*02
  - IGHVF6-G21*07
J:
  - IGHJ4*02
  - IGHJ6*02
  - IGHJ6*03`}
      />
      <p>
        The same in JSON is <code>{`{"V": ["IGHVF10-G50*02", ...], "J": ["IGHJ4*02", ...]}`}</code>. To see the names a
        model knows, export its germline set (<code>alignair reference export</code>, on{" "}
        <DocLink to="own-reference?h=export-the-reference">Use your own germline reference</DocLink>) and list the
        FASTA headers.
      </p>
      <CodeTabs
        cli={`alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv --genotype donor.yaml`}
        python={`from alignair import Aligner, read_sequences

aligner = Aligner.from_pretrained("alignair4-igh-human")
donor = aligner.with_genotype({"V": ["IGHVF10-G50*02", "IGHVF10-G42*01", "IGHVF3-G12*02", "IGHVF6-G21*07"],
                               "J": ["IGHJ4*02", "IGHJ6*02", "IGHJ6*03"]})
ids, reads, info = read_sequences("reads.fasta")
donor.predict(reads, ids=ids).write_airr("out.tsv")`}
      />
      <p>
        <code>with_genotype</code> returns a new <code>Aligner</code> and leaves the original as it was.
      </p>

      <H2 id="outside-the-genotype">Reads of genes the genotype leaves out</H2>
      <p>
        With a genotype, every call comes from its alleles, so the genotype must hold every allele the donor carries.
        Here is the genotype above, which names 4 V alleles, on the 20 example reads (left: no genotype; right: with
        it):
      </p>
      <CodeBlock
        lang="text"
        title="output"
        code={`read     without genotype                          with donor.yaml
read01   IGHVF10-G50*02                 IGHJ6*02   IGHVF10-G50*02  IGHJ6*02  complete
read02   IGHVF10-G42*01,IGHVF10-G42*03  IGHJ4*02   IGHVF10-G42*01  IGHJ4*02  complete
read04   IGHVF9-G32*01                  IGHJ5*02   IGHVF10-G50*02  IGHJ4*02  complete
read07   IGHVF1-G2*01                   IGHJ3*02   (no V call)     IGHJ6*02  partial, no_v`}
      />
      <ul>
        <li>read01 carries an allele of the genotype: its call is unchanged.</li>
        <li>read02's tie is settled: <code>IGHVF10-G42*03</code> is not in the genotype.</li>
        <li>
          read04 is a read of a gene the genotype leaves out. It is called to the closest allele the genotype allows,
          and its row is still <code>complete</code>: nothing marks the call as forced. Its J is forced the same way.
        </li>
        <li>
          read07 matches no allowed V allele closely (under 70% identity over its V, and no 15-nt stretch in common
          with any of them). It gets no V call, its D and J calls stand, and its row is <code>partial</code> with{" "}
          <code>airr_assembly_reason</code> <code>no_v</code>.
        </li>
      </ul>
      <Callout kind="warning" title="A genotype that misses alleles forces wrong calls">
        A read of an allele the genotype leaves out is called to the nearest allele it allows, or gets no V call. Use a
        genotype that covers every gene the donor expresses. Many <code>no_v</code> rows mean the genotype misses
        genes, or the reads are of another locus.
      </Callout>
      <p>
        Partial rows never fail the run by default; the console line counts them (
        <code>aligned 20 reads (0 dropped) -&gt; out.tsv; 0 failed / 2 partial AIRR assemblies tagged</code>).
      </p>

      <H2 id="refused">When a genotype is refused</H2>
      <p>A genotype that names an allele the model's germline set does not hold is refused before any read is aligned:</p>
      <CodeBlock
        title="shell"
        code={`$ alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv --genotype imgt.yaml
[ALIGNAIR_GENOTYPE_ALLELE_NOT_FOUND] the genotype names alleles the reference does not hold: IGHV1-2*02, IGHV3-23*01
Context: {"command": "predict", "missing": ["IGHV1-2*02", "IGHV3-23*01"], "stage": "genotype_validation"}
Hint: Give their sequences in the reference (alignair predict --reference; in Python, Aligner.from_pretrained(reference=) or bound_to).`}
      />
      <p>
        Here the genotype uses IMGT names on an IG model. Write it with the OGRDB names instead (look them up in{" "}
        <Ext href={OGRDB_SETS}>OGRDB's germline sets</Ext>), or bind a germline set that holds these names with{" "}
        <code>--reference</code>: IMGT's set, or the model's own set plus your novel alleles. The genotype can then name
        them. The other refusals:
      </p>
      <DocTable
        head={["The genotype file", "Error"]}
        rows={[
          [
            <>is keyed by gene names (<code>IGHV3-23: [...]</code>)</>,
            <>
              <code>ALIGNAIR_CONFIG_INVALID</code>: donor.yaml is keyed by gene names (IGHV3-23); its keys are segments
              (V, D, J), each listing allele names
            </>,
          ],
          [
            "is not YAML or JSON (a FASTA, say)",
            <>
              <code>ALIGNAIR_CONFIG_INVALID</code>: donor.fasta is not a YAML or JSON genotype: ... (a FASTA goes to{" "}
              <code>--reference</code>)
            </>,
          ],
          [
            <>gives a segment an empty list (<code>V: []</code>)</>,
            <>
              <code>ALIGNAIR_INPUT_INVALID</code>: the genotype leaves gene V with no allele
            </>,
          ],
        ]}
      />
      <p>
        In Python, <code>load_genotype</code> and <code>with_genotype</code> raise the matching errors (
        <code>UnknownAlleleError</code> for an allele the set lacks). Every code is on{" "}
        <DocLink to="error-codes">Error codes</DocLink>.
      </p>

      <H2 id="genotype-from-a-table">A genotype from an AIRR table: alignair genotype</H2>
      <p>
        <code>alignair genotype</code> infers a donor's V genotype from any tool's AIRR table, by TIgGER's frequency
        rule. For each V gene it counts the rows whose <code>v_call</code> is a single allele (a tie counts for none).
        The gene keeps its most frequent alleles until they explain 87.5% of those rows (<code>--explain</code>). A gene
        with fewer than 10 such rows (<code>--min-reads</code>) is left open: no evidence, so no genotype for it. It
        writes an AIRR GenotypeSet; <code>--locus</code> sets its locus (default IGH).
      </p>
      <CodeBlock
        title="shell (output abridged)"
        code={`$ alignair genotype out.tsv --out genotype.json
IGHVF1-G2: open (1 single-allele rows, fewer than 10)
IGHVF10-G50: open (2 single-allele rows, fewer than 10)
...
genotyped 0 of 15 V genes -> genotype.json`}
      />
      <p>
        (The 20 example reads are far too few; a repertoire of thousands of reads genotypes most genes.) A genotyped
        gene prints its alleles, such as <code>IGHVF3-G9: IGHVF3-G9*03, IGHVF3-G9*01</code>. <code>--genotype</code>{" "}
        reads the GenotypeSet as it is, so the round trip is one more run:
      </p>
      <CodeTabs
        cli={`alignair genotype out.tsv --out genotype.json
alignair predict --model alignair4-igh-human --input reads.fasta --out out.donor.tsv --genotype genotype.json`}
        python={`from alignair import Aligner, read_sequences
from alignair.genotype import load_genotype

aligner = Aligner.from_pretrained("alignair4-igh-human")
donor = aligner.with_genotype(load_genotype("genotype.json"))   # a YAML/JSON genotype or a GenotypeSet
ids, reads, info = read_sequences("reads.fasta")
donor.predict(reads, ids=ids).write_airr("out.donor.tsv")`}
      />
      <Callout kind="note" title="Open genes are not in the GenotypeSet">
        The file lists the alleles of the genotyped genes only. Given to <code>--genotype</code>, it allows only those V
        alleles, so the reads of an open gene are called to a genotyped gene's allele, or get no V call. Use the round
        trip when the genes you expect are genotyped, or use <code>--infer-genotype</code>, which keeps every allele of
        an open gene.
      </Callout>

      <H2 id="infer-genotype">Infer the genotype in the same run: --infer-genotype</H2>
      <p>
        <code>--infer-genotype</code> does both steps in one run, in two passes over the file. Pass 1 aligns every read
        and infers the V genotype by the same rule. Pass 2 re-calls only the reads whose call the genotype excludes
        while their <code>v_call</code> lists an allele it allows; an open gene keeps all its alleles. D and J calls
        are not changed.
      </p>
      <CodeBlock
        title="shell"
        code={`$ alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv --infer-genotype
aligned 20 reads (0 dropped) -> out.tsv
$ ls out.tsv*
out.tsv  out.tsv.genotype.json  out.tsv.run.json`}
      />
      <ul>
        <li>
          The genotype is written to <code>--genotype-out</code> (default <code>out.tsv.genotype.json</code>), the same
          GenotypeSet <code>alignair genotype</code> writes. <code>--genotype-explain</code> (0.875) and{" "}
          <code>--genotype-min-reads</code> (10) set the rule.
        </li>
        <li>
          The input is read twice, so it must be a file, and all its reads should come from one donor. It cannot be
          combined with <code>--genotype</code>.
        </li>
        <li>It genotypes V only. It does not find novel alleles, mark deleted genes or separate donors.</li>
      </ul>
      <p>The Python API has no inferring pass; run it from the command line.</p>
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Use your own germline reference                                                                                    */
/* ------------------------------------------------------------------------------------------------------------------ */
const ownReference: DocPage = {
  slug: "own-reference",
  title: "Use your own germline reference",
  section: SECTION,
  lead: "Call from your own germline set, novel alleles included, with a pretrained model and no training.",
  body: () => (
    <>
      <p>
        <code>--reference germline.fasta</code> replaces the model's germline set for the run: every call is then an
        allele of your FASTA. What the model learned carries over to alleles it never saw, so a curated set, IMGT's set
        for an IG locus, or a set with your donor's novel alleles all work. The set must be of the model's locus; for
        another species or locus, see <DocLink to="do-i-need-to-train">Do I need to train?</DocLink>
      </p>

      <H2 id="fasta-requirements">What the FASTA needs</H2>
      <ul>
        <li>
          <strong>Every segment the model calls.</strong> V, D and J alleles for the IGH, TRB and TRD models; V and J for
          IGK, IGL, TRA and TRG.
        </li>
        <li>
          <strong>Names that show their segment.</strong> The first word of each header is the allele's name. The
          segment is read from the letter after the locus code (<code>IGHV1-2*02</code> is V, <code>TRBJ2-1*01</code> is
          J), else from the one V, D or J letter in the name. IMGT/GENE-DB headers hold several fields joined by{" "}
          <code>|</code>: cut them to the allele name.
        </li>
        <li>
          <strong>One sequence per name.</strong> A name given twice with two sequences is refused.
        </li>
        <li>
          <strong>Gapped or not.</strong> Sequences may be IMGT-gapped (with <code>.</code> or <code>-</code>) or
          ungapped. A gapped V keeps its gaps. Give new V alleles gapped if you can (next section).
        </li>
      </ul>
      <p>The easiest start is the model's own set, to which you add your alleles:</p>
      <CodeTabs
        cli={`alignair reference export "$(alignair models path alignair4-igh-human)" --fasta germline.fasta
cat novel_alleles.fasta >> germline.fasta
alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv --reference germline.fasta`}
        python={`from alignair import Aligner, ReferenceSet, read_sequences

aligner = Aligner.from_pretrained("alignair4-igh-human", reference=ReferenceSet.from_fasta("germline.fasta"))
ids, reads, info = read_sequences("reads.fasta")
aligner.predict(reads, ids=ids).write_airr("out.tsv")`}
      />
      <p>
        In Python, <code>aligner.bound_to(ReferenceSet.from_fasta("germline.fasta"))</code> does the same for an{" "}
        <code>Aligner</code> already loaded, and returns a new one.
      </p>

      <H2 id="novel-alleles">Novel alleles</H2>
      <p>
        A V allele of your set that matches the read at more bases than the model's call becomes the call, so a novel
        allele the read supports is called by its own name. For example, add to the IGH set a novel allele of{" "}
        <code>IGHVF10-G50*02</code> with two substitutions, named the TIgGER way:
      </p>
      <CodeBlock
        lang="text"
        title="novel_alleles.fasta (sequence shortened here)"
        code={`>IGHVF10-G50*02_G23A_G25A
CAGGTGCAGCTGGTGGAGTCTGAGAGAGGCTTGGTCAAGCCTGGAGGGTCCCTGAGACTCTCCTGTGCAG...`}
      />
      <CodeBlock
        title="shell"
        code={`$ alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv --reference germline.fasta
aligned 20 reads (0 dropped) -> out.tsv
$ cut -f1,5 out.tsv | head -3
sequence_id	v_call
read01	IGHVF10-G50*02_G23A_G25A
read02	IGHVF10-G42*01,IGHVF10-G42*03`}
      />
      <p>
        read01 carries both substitutions and is now called to the novel allele; the other reads are unchanged. A{" "}
        <DocLink to="donor-genotype">genotype</DocLink> can name the novel allele once it is in the bound set.
      </p>

      <H2 id="gaps-and-anchors">How missing gaps and anchors are filled</H2>
      <p>
        Each V and J allele needs its anchor (the first base of the V Cys or the J Trp/Phe codon), and each V its
        IMGT-gapped form. AlignAIR fills what your FASTA does not give, in this order:
      </p>
      <ol>
        <li>what the FASTA gives: a gapped V keeps its gaps;</li>
        <li>the model's allele with the same sequence: its gaps and anchor;</li>
        <li>
          for a V, the model's allele of the same length that differs at the fewest bases, when that is at most 10% of
          them: its gaps and anchor (this is how the novel allele above got its own);
        </li>
        <li>else, a V's anchor from its own IMGT gaps, and a J's from its one F/W-G-X-G motif.</li>
      </ol>
      <p>Alleles left without an anchor are named in a warning before the run:</p>
      <CodeBlock
        lang="text"
        title="stderr"
        code={`warning: --reference germline.fasta: V 194/195, J 7/7 anchored; no anchor: IGHVF10-G50*99 (unresolved)`}
      />
      <p>
        Here a novel V with a 3-nt insertion was given ungapped, so no rule could place it. Give such an allele
        IMGT-gapped, and check its sequence.
      </p>

      <H2 id="reference-refusals">References that are refused</H2>
      <DocTable
        head={["The FASTA", "What happens"]}
        rows={[
          [
            <>lacks a segment the model calls (an IGK set on the IGH model has no D)</>,
            <>
              Refused: <code>[ALIGNAIR_CONFIG_INVALID] the reference has no D alleles, but this model calls V, D and J</code>
              . Give all the segments (<code>alignair reference export</code> writes the model's own), or use the model of
              your set's locus.
            </>,
          ],
          [
            <>holds alleles of another locus besides the model's (IGH and IGK together)</>,
            <>
              Refused:{" "}
              <code>
                [ALIGNAIR_CONFIG_INVALID] the reference holds alleles of IGK, a locus the model does not call (it calls
                IGH)
              </code>
              . Split the set by locus and bind each part to its own model.
            </>,
          ],
          [
            <>is wholly of another locus that has the model's segments (a TRB set on the IGH model)</>,
            <>
              Binds, with a warning:{" "}
              <code>
                warning: the bound reference is TRB, but the model was trained on IGH: binding is allowed, but its calls
                on TRB have not been measured; records are labelled TRB
              </code>
              . Use the model of that locus.
            </>,
          ],
          [
            "has names that show no segment",
            <>
              Refused: <code>[ALIGNAIR_INPUT_INVALID]</code> germline.fasta: the segment (V, D or J) of N alleles cannot
              be read from their names. Rename them.
            </>,
          ],
          [
            "is of another species",
            <>
              Not checked: it binds when it has the model's segments, but how well a human model calls another species'
              alleles has not been measured. Train a model for that species (<DocLink to="train">Train a model</DocLink>
              ).
            </>,
          ],
        ]}
      />

      <H2 id="export-the-reference">Export a model's germline set</H2>
      <p>
        <code>alignair reference export</code> writes a model file's germline set as one FASTA of its V, D and J alleles
        (238 for the IGH model). It takes the model file, so resolve an id to its cached file first with{" "}
        <code>alignair models path</code>, which downloads the model if needed. <code>--gapped</code> writes each V
        IMGT-gapped, as Change-O's <code>CreateGermlines.py</code> needs; D and J are written as they are.
      </p>
      <CodeBlock
        title="shell (output shortened)"
        code={`$ alignair reference export "$(alignair models path alignair4-igh-human)" --fasta germline_gapped.fasta --gapped
wrote germline_gapped.fasta
$ head -2 germline_gapped.fasta
>IGHVF1-G1*01
CAGGTCACCTTGAAGGAGTCTGGTCCT...GTGCTGGTGAAACCCACAGAGACCCTCACGCTGACCTGCACCGTCTCTGGGTTCTCACTCAGC......AATGCTAGAATG …`}
      />
      <p>
        With a <code>--reference</code> run, the germline set to give downstream tools is your FASTA, not this export.
      </p>
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Join sample and cell metadata                                                                                      */
/* ------------------------------------------------------------------------------------------------------------------ */
const metadataJoin: DocPage = {
  slug: "metadata-join",
  title: "Join sample and cell metadata",
  section: SECTION,
  lead: "Carry cell barcodes, UMI counts, sample ids or any other per-read column from a side table into the AIRR table.",
  body: () => (
    <>
      <H2 id="10x-example">A 10x example</H2>
      <p>
        <code>--metadata</code> joins a CSV or TSV table to the reads by read id. With Cell Ranger's output, the contigs
        go to <code>--input</code> and the annotations to <code>--metadata</code>:
      </p>
      <CodeTabs
        cli={`alignair predict --model alignair4-igh-human --input filtered_contig.fasta --out out.tsv \\
  --metadata filtered_contig_annotations.csv`}
        python={`from alignair import Aligner, AirrWriter
from alignair.io import build_metadata_index, iter_sequence_batches, lookup_metadata_rows

aligner = Aligner.from_pretrained("alignair4-igh-human")
index, columns = build_metadata_index("filtered_contig_annotations.csv", normalize_10x=True)
with index, AirrWriter("out.tsv", locus=aligner.default_locus, extra_columns=columns) as writer:
    for batch in iter_sequence_batches("filtered_contig.fasta"):
        records = aligner.predict(batch.sequences).to_dicts()
        writer.write(batch.ids, batch.sequences, records, metas=lookup_metadata_rows(index, batch.source_ids))`}
      />
      <p>The annotation columns follow AlignAIR's own (first row shown, a few columns):</p>
      <CodeBlock
        lang="text"
        title="out.tsv"
        code={`sequence_id                  v_call         c_call  barcode             umis  reads  chain  cell_id             umi_count  consensus_count
AAACCTGAGAAACCAT-1_contig_1  IGHVF6-G21*07  IGHM    AAACCTGAGAAACCAT-1  12    4521   IGH    AAACCTGAGAAACCAT-1  12         4521`}
      />
      <p>
        Cell Ranger's <code>barcode</code>, <code>umis</code>, <code>reads</code> and <code>c_gene</code> are also
        written under their AIRR names, <code>cell_id</code>, <code>umi_count</code>, <code>consensus_count</code> and{" "}
        <code>c_call</code>, which Change-O, SCOPer and Scirpy read. The metadata is carried as given; it is not
        checked against the calls. The 10x <code>chain</code> column is also how you split a mixed library by locus (
        <DocLink to="choose-a-model?h=pick-by-locus">Pick by locus</DocLink>).
      </p>

      <H2 id="id-column">Which column holds the read id</H2>
      <p>
        By default, the id column is the table's first column named <code>sequence_id</code>, <code>contig_id</code>,{" "}
        <code>cell_id</code>, <code>id</code> or <code>name</code> (any case); in Cell Ranger's table that is{" "}
        <code>contig_id</code>. Name another with <code>--metadata-id-column</code>:
      </p>
      <CodeBlock
        code={`alignair predict --model alignair4-igh-human --input reads.tsv --out out.tsv \\
  --metadata samples.tsv --metadata-id-column read_name`}
      />
      <ul>
        <li>
          Ids are matched as the input file gives them, before a repeated id gets its <code>_dup1</code> suffix, so
          both copies of a repeated read get its row.
        </li>
        <li>A read whose id is not in the table gets empty metadata columns.</li>
        <li>
          An id twice in the table is refused (<code>ALIGNAIR_METADATA_ID_INVALID</code>): choose a column whose values
          are unique.
        </li>
      </ul>

      <H2 id="kept-columns">Which columns are kept</H2>
      <p>
        By default, AlignAIR keeps those of these columns the table has: <code>cell_id</code>, <code>barcode</code>,{" "}
        <code>sample_id</code>, <code>umi_count</code>, <code>umis</code>, <code>reads</code>,{" "}
        <code>duplicate_count</code>, <code>consensus_count</code>, <code>raw_clonotype_id</code>,{" "}
        <code>raw_consensus_id</code>, <code>chain</code>, <code>c_call</code>, <code>is_cell</code> and{" "}
        <code>high_confidence</code>, plus the AIRR names above. <code>--keep-columns</code> chooses them yourself, by
        their names in the file (Cell Ranger's <code>umis</code>, not <code>umi_count</code>); a name the file does not
        have is an error.
      </p>
      <CodeTabs
        cli={`alignair predict --model alignair4-igh-human --input reads.tsv --out out.tsv \\
  --metadata sample_metadata.tsv --keep-columns sample_id,subject_id,tissue,timepoint`}
        python={`from alignair import Aligner, AirrWriter
from alignair.io import build_metadata_index, iter_sequence_batches, lookup_metadata_rows

aligner = Aligner.from_pretrained("alignair4-igh-human")
index, columns = build_metadata_index("sample_metadata.tsv",
                                      keep_columns=["sample_id", "subject_id", "tissue", "timepoint"])
with index, AirrWriter("out.tsv", locus=aligner.default_locus, extra_columns=columns) as writer:
    for batch in iter_sequence_batches("reads.tsv"):
        records = aligner.predict(batch.sequences).to_dicts()
        writer.write(batch.ids, batch.sequences, records, metas=lookup_metadata_rows(index, batch.source_ids))`}
      />
      <p>
        The metadata is per read. To give every read of a file the same sample id, add the column to the output
        afterwards instead.
      </p>

      <H2 id="meta-prefix">Columns AlignAIR also writes: the meta_ prefix</H2>
      <p>
        Metadata never overwrites a value AlignAIR writes. A kept column whose name is one of AlignAIR's own columns is
        written as <code>meta_&lt;name&gt;</code>: keep Cell Ranger's <code>productive</code> and the table gets both
        AlignAIR's <code>productive</code> and Cell Ranger's <code>meta_productive</code>.
      </p>
      <p>
        <code>c_call</code> is the exception. AlignAIR does not call the constant region, so a metadata{" "}
        <code>c_call</code> (or Cell Ranger's <code>c_gene</code>) keeps its name and fills the column's empty cells.
      </p>
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Choose output columns                                                                                              */
/* ------------------------------------------------------------------------------------------------------------------ */
const MINIMAL = ["sequence_id", "sequence", "locus", "v_call", "d_call", "j_call", "productive"];
const AIRR = ["sequence_id", "sequence", "rev_comp", "productive", "v_call", "d_call", "j_call", "sequence_alignment",
  "germline_alignment", "junction", "junction_aa", "v_cigar", "d_cigar", "j_cigar"];

const outputColumnsChoice: DocPage = {
  slug: "output-columns-choice",
  title: "Choose output columns",
  section: SECTION,
  lead: "Write all 99 columns of the AIRR table, a smaller preset, or exactly the columns you name.",
  body: () => (
    <>
      <H2 id="presets">The presets</H2>
      <p>
        <code>--columns</code> takes a preset or a list of columns. Every column is described on{" "}
        <DocLink to="column-dictionary">Output column dictionary</DocLink>, and what they tell you on{" "}
        <DocLink to="output-columns">Output columns explained</DocLink>.
      </p>
      <DocTable
        head={["Preset", "Columns", "What it holds"]}
        rows={[
          [
            <><code>full</code> (default)</>,
            "99",
            <>
              Every column: calls, junction, alignments, coordinates, CIGARs, regions, identities, the candidate sets (
              <code>*_call_set</code>), orientation and the assembly status.
            </>,
          ],
          [
            <code>core</code>,
            "27",
            <>
              <code>sequence_id</code>, <code>sequence</code>, <code>rev_comp</code> and <code>locus</code>; the calls
              (with <code>c_call</code>); <code>productive</code>; <code>junction</code>, <code>junction_aa</code> and{" "}
              <code>junction_length</code>; each segment's start and end in the read and in the germline; the CIGARs.
            </>,
          ],
          [<code>airr</code>, "14", "The fields the AIRR rearrangement schema requires."],
          [
            <code>minimal</code>,
            "7",
            "The calls and productivity only. It skips the AIRR assembly, so it is the fastest.",
          ],
        ]}
      />
      <H3 id="preset-minimal">minimal</H3>
      <ColumnTable names={MINIMAL} />
      <H3 id="preset-airr">airr</H3>
      <ColumnTable names={AIRR} />
      <Callout kind="note" title="Only full has the assembly status">
        <code>airr_assembly_status</code> and <code>airr_assembly_reason</code>, which mark the partial rows, are in{" "}
        <code>full</code> only. With a smaller preset the partial rows are still written, untagged; the console line
        still counts them.
      </Callout>

      <H2 id="for-immcantation">For Change-O, TIgGER and SCOPer</H2>
      <p>
        Keep <code>full</code>. Between them these tools read the IMGT-gapped <code>sequence_alignment</code> and{" "}
        <code>germline_alignment</code>, the junction and <code>junction_length</code>, the germline coordinates (
        <code>v_germline_start</code> ...) and the N-region lengths (<code>np1_length</code>, <code>np2_length</code>):
        only <code>full</code> has them all. It also has <code>airr_assembly_status</code>, to keep the complete rows
        before clonal analysis. <code>v_call</code> lists the alleles that match a read equally well after the call,
        comma-separated, as TIgGER and Change-O expect. More on{" "}
        <DocLink to="from-immcantation">Coming from Immcantation</DocLink>.
      </p>

      <H2 id="column-list">Name your own columns</H2>
      <p>
        A comma-separated list writes exactly those columns, in that order. A name AlignAIR does not know is written as
        an empty column, so check the spelling. A list that names no junction, alignment, region or identity column
        (nor <code>vj_in_frame</code> or <code>stop_codon</code>) skips the AIRR assembly, as <code>minimal</code>{" "}
        does.
      </p>
      <CodeTabs
        cli={`alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv \\
  --columns sequence_id,v_call,d_call,j_call,junction_aa,productive,airr_assembly_status`}
        python={`from alignair import Aligner, read_sequences

aligner = Aligner.from_pretrained("alignair4-igh-human")
ids, reads, info = read_sequences("reads.fasta")
result = aligner.predict(reads, ids=ids)
result.write_airr("out.tsv", columns=["sequence_id", "v_call", "d_call", "j_call", "junction_aa", "productive",
                                      "airr_assembly_status"])
result.write_airr("out.core.tsv", columns="core")      # a preset by name`}
      />

      <H2 id="run-json">The run record beside the table</H2>
      <p>
        Each command-line run also writes <code>out.tsv.run.json</code>: the model's id, version and checksum, the
        fingerprints of the germline set the calls came from (yours, with <code>--reference</code>), the genotype, the
        options, and the counts of complete, partial and failed rows. Keep it with the table for your methods section.{" "}
        <code>--no-run-metadata</code> leaves it out.
      </p>

      <H2 id="t-f-columns">Reading the T/F columns</H2>
      <p>
        <code>productive</code>, <code>rev_comp</code>, <code>vj_in_frame</code>, <code>stop_codon</code>,{" "}
        <code>segmentation_low_quality</code> and <code>length_cropped</code> hold the AIRR strings <code>T</code> and{" "}
        <code>F</code>; an empty cell means unknown, not false. pandas reads them as strings, so compare with{" "}
        <code>"T"</code>, never with <code>.astype(bool)</code>, which turns <code>"F"</code> into True:
      </p>
      <CodeBlock
        lang="python"
        code={`import pandas as pd

table = pd.read_csv("out.tsv", sep="\\t")
usable = table[(table["airr_assembly_status"] == "complete") & (table["segmentation_low_quality"] != "T")]
productive = usable[usable["productive"] == "T"]`}
      />
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Run at scale                                                                                                       */
/* ------------------------------------------------------------------------------------------------------------------ */
const runAtScale: DocPage = {
  slug: "run-at-scale",
  title: "Run at scale",
  section: SECTION,
  lead: "Use a GPU or every CPU core, keep memory flat on large files, and align many samples in a loop or in nf-core/airrflow.",
  body: () => (
    <>
      <H2 id="gpu">On a GPU</H2>
      <p>
        <code>--device auto</code>, the default, uses an NVIDIA GPU (CUDA) when there is one, else Apple's GPU (MPS),
        else the CPU; <code>alignair doctor</code> shows which as <code>device_auto</code>. Name a device with{" "}
        <code>--device cuda</code>, <code>cuda:1</code>, <code>mps</code> or <code>cpu</code>. On a GPU, reads go in
        batches of 256; raise <code>--batch-size</code> on a large GPU, lower it if the GPU runs out of memory.
      </p>
      <CodeTabs
        cli={`alignair predict --model alignair4-igh-human --input reads.fastq.gz --out out.tsv --device cuda --batch-size 512`}
        python={`from alignair import Aligner, read_sequences

aligner = Aligner.from_pretrained("alignair4-igh-human", device="cuda")
ids, reads, info = read_sequences("reads.fastq.gz")
aligner.predict(reads, ids=ids, batch_size=512).write_airr("out.tsv")`}
      />

      <H2 id="cpu-threads">On the CPU: --threads</H2>
      <p>
        Without a GPU, <code>--threads N</code> runs N worker processes of one thread each, with batches of 16 reads per
        worker; this scales better than one process using many threads. A number near your physical core count is a
        good start; leave some cores free on a shared machine. Without <code>--threads</code>, one process runs with
        batches of 64. On a GPU, <code>--threads</code> only sets the CPU threads of the one process.
      </p>
      <CodeBlock code={`alignair predict --model alignair4-trb-human --input tcr.fasta --out tcr.tsv --threads 8`} />
      <p>Worker processes are a command-line feature; in Python, one process predicts.</p>

      <H2 id="memory">Large files in constant memory</H2>
      <p>
        The command line streams its input: it reads 20,000 reads at a time (<code>--chunk-size</code>), aligns and
        writes them, then reads the next chunk. Memory depends on the chunk and batch sizes, not on the size of the file
        (only the read ids are remembered, to rename repeats). The table is written to a temporary file and moved into
        place when the run ends, so a run that stops halfway leaves no half-written table.
      </p>
      <CodeTabs
        cli={`alignair predict --model alignair4-igh-human --input big.fastq.gz --out big.tsv --chunk-size 50000`}
        python={`from alignair import Aligner, AirrWriter, iter_sequences

aligner = Aligner.from_pretrained("alignair4-igh-human")
with AirrWriter("big.tsv", locus=aligner.default_locus) as writer:
    for ids, reads, n_dropped in iter_sequences("big.fastq.gz", chunk_size=20000):
        writer.write(ids, reads, aligner.predict(reads).to_dicts())`}
      />
      <p>
        <code>--columns minimal</code> skips the AIRR assembly and writes much smaller rows, when the calls are all you
        need (<DocLink to="output-columns-choice">Choose output columns</DocLink>).
      </p>

      <H2 id="many-samples">Many samples</H2>
      <p>
        Run one command per sample. Download the model once before you start, and run the jobs with{" "}
        <code>--offline</code>, so parallel jobs never fetch it at the same time; <code>--quiet</code> drops the
        update notice. With a tab-separated manifest whose columns are <code>sample</code> and <code>input</code>:
      </p>
      <CodeTabs
        cli={`alignair models get alignair4-igh-human
mkdir -p results
tail -n +2 manifest.tsv | while IFS=$'\\t' read -r sample input; do
  alignair predict --model alignair4-igh-human --input "$input" --out "results/\${sample}.tsv" --offline --quiet
done`}
        python={`import csv
import os
from alignair import Aligner, read_sequences

aligner = Aligner.from_pretrained("alignair4-igh-human")    # loaded once for every sample
os.makedirs("results", exist_ok=True)
with open("manifest.tsv") as manifest:
    for row in csv.DictReader(manifest, delimiter="\\t"):
        ids, reads, info = read_sequences(row["input"])
        aligner.predict(reads, ids=ids).write_airr(f"results/{row['sample']}.tsv")`}
      />
      <p>
        Each command-line run loads the model again; for many small samples, the Python loop loads it once. Each run
        of the command line writes its sample's own <code>.run.json</code>.
      </p>

      <H2 id="airrflow">nf-core/airrflow</H2>
      <p>
        airrflow takes AlignAIR's tables in its <code>assembled</code> mode, one per sample, listed in the samplesheet's{" "}
        <code>filename</code> column. Its other required columns (subject, species, tissue, locus and more) are listed
        in <Ext href={AIRRFLOW_USAGE}>airrflow's usage docs</Ext>: check them for the version you run.
      </p>
      <Callout kind="warning" title="Pass --reassign false">
        On an AIRR table, airrflow re-assigns the genes by default (<code>--reassign</code>, default true), which
        replaces AlignAIR's calls. Pass <code>--reassign false</code> to keep them.
      </Callout>
      <CodeBlock
        code={`nextflow run nf-core/airrflow -r <version> -profile docker \\
  --mode assembled --input samplesheet.tsv --reassign false --outdir results`}
      />
      <p>
        airrflow's later steps look calls up in its own germline set, IMGT's by default (
        <code>--reference_fasta</code>, on <Ext href={AIRRFLOW_PARAMETERS}>its parameters page</Ext>). The IG models
        write OGRDB names, which that set does not hold: check how your airrflow version uses its germline set before
        you rely on those steps.
      </p>
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Fix common problems                                                                                                */
/* ------------------------------------------------------------------------------------------------------------------ */
const fixCommonProblems: DocPage = {
  slug: "fix-common-problems",
  title: "Fix common problems",
  section: SECTION,
  lead: "Find what you see, its cause, and what to do about it.",
  body: () => (
    <>
      <p>
        Every failure prints a code in brackets, such as <code>[ALIGNAIR_GENOTYPE_ALLELE_NOT_FOUND]</code>, then the
        message, and often a hint. <DocLink to="error-codes">Error codes</DocLink> lists every code with what to do.{" "}
        <code>--debug</code> adds the full traceback, and <code>--error-format json</code> prints the error as one line
        of JSON for a pipeline to read.
      </p>

      <H2 id="genotype-and-reference">Genotype and reference</H2>
      <DocTable
        head={["What you see", "Cause", "What to do"]}
        rows={[
          [
            <code>[ALIGNAIR_GENOTYPE_ALLELE_NOT_FOUND] the genotype names alleles the reference does not hold</code>,
            "The genotype names alleles the model's germline set lacks: IMGT names on an IG model, or novel alleles.",
            <>
              Use the OGRDB names (<DocLink to="choose-a-model?h=allele-names">Allele names</DocLink>), or bind a set
              that holds them with <code>--reference</code> (<DocLink to="donor-genotype?h=refused">refusals</DocLink>
              ).
            </>,
          ],
          [
            <>
              <code>[ALIGNAIR_CONFIG_INVALID]</code> ... is keyed by gene names, or ... is not a YAML or JSON genotype
            </>,
            "The genotype file has the wrong shape, or is a FASTA.",
            <>
              Key it by <code>V</code>, <code>D</code>, <code>J</code>, each listing allele names; a FASTA goes to{" "}
              <code>--reference</code>.
            </>,
          ],
          [
            <code>[ALIGNAIR_CONFIG_INVALID] the reference has no D alleles, but this model calls V, D and J</code>,
            "The --reference FASTA lacks a segment the model calls, often a light-chain set on the IGH model.",
            "Give V, D and J, or use the model of the set's locus.",
          ],
          [
            <code>[ALIGNAIR_CONFIG_INVALID] the reference holds alleles of IGK, a locus the model does not call</code>,
            "The FASTA mixes loci.",
            "Split it by locus; bind each part to its own model.",
          ],
          [
            <code>warning: the bound reference is TRB, but the model was trained on IGH</code>,
            "The FASTA is of another locus than the model's.",
            "Use the model of that locus.",
          ],
        ]}
      />

      <H2 id="rows-and-run">Rows and runs</H2>
      <DocTable
        head={["What you see", "Cause", "What to do"]}
        rows={[
          [
            <>
              Many <code>partial</code> rows with <code>airr_assembly_reason</code> <code>no_v</code>
            </>,
            "No V close enough among the alleles allowed: a genotype that misses genes, reads of another locus, or reads that start inside the junction.",
            <>
              Check the genotype covers the donor's genes (
              <DocLink to="donor-genotype?h=outside-the-genotype">reads of genes it leaves out</DocLink>) and the model's
              locus.
            </>,
          ],
          [
            <>
              Many <code>partial</code> rows with another reason
            </>,
            "Short, damaged or off-target reads: the calls are there, but the junction or alignment could not be built.",
            <>
              Keep <code>airr_assembly_status == "complete"</code> for junction and clonal analysis. See{" "}
              <DocLink to="trust-the-calls">How far to trust the calls</DocLink>.
            </>,
          ],
          [
            <>
              The run exits with <code>[ALIGNAIR_INPUT_INVALID] AIRR assembly FAILED for</code> ... reads
            </>,
            <>
              More than 1% of the rows failed assembly (<code>--max-assembly-failures 0.01</code>). The table is written
              and every row tagged; only the exit status reports it.
            </>,
            <>
              Read <code>airr_assembly_error</code> of the failed rows. Allow more with{" "}
              <code>--max-assembly-failures 0.05</code>, or never fail with <code>--permissive</code>. Report a failure
              that repeats, with <code>--debug</code>.
            </>,
          ],
          [
            <code>[ALIGNAIR_INPUT_INVALID] no valid reads in reads.fasta</code>,
            "Every read was dropped: the wrong file or format.",
            <>
              Look at the reasons with <code>--rejects-out</code> (<DocLink to="prepare-input?h=dropped-reads">dropped
              reads</DocLink>).
            </>,
          ],
          [
            <code>[ALIGNAIR_INPUT_INVALID] no sequence column in reads.csv</code>,
            "The table's sequence column has another name.",
            <>
              Name it with <code>--sequence-column</code> (and the ids with <code>--id-column</code>).
            </>,
          ],
          [
            <>
              Rows with <code>length_cropped</code> = <code>T</code> lack their J or the start of their V
            </>,
            "The reads are longer than 575 nt.",
            <>
              Trim reads to the V(D)J region first (<DocLink to="prepare-input?h=long-reads">long reads</DocLink>).
            </>,
          ],
          [
            <>
              <code>[ALIGNAIR_METADATA_ID_INVALID] duplicate metadata id</code>
            </>,
            "The metadata table repeats an id.",
            <>
              Choose a column of unique ids with <code>--metadata-id-column</code>.
            </>,
          ],
        ]}
      />

      <H2 id="models-and-machines">Models and machines</H2>
      <DocTable
        head={["What you see", "Cause", "What to do"]}
        rows={[
          [
            <code>[ALIGNAIR_MODEL_UNAVAILABLE] offline: model is not available in the local cache or registries</code>,
            <>
              <code>--offline</code> (or <code>offline=True</code>) and the model is not in this cache.
            </>,
            <>
              Run <code>alignair models get &lt;id&gt;</code> once with internet, with the same{" "}
              <code>ALIGNAIR_CACHE_DIR</code> the offline job uses.
            </>,
          ],
          [
            <code>[ALIGNAIR_REGISTRY_UNAVAILABLE]</code>,
            "Hugging Face could not be reached when the model was first needed.",
            "Check the connection and try again, or download the model ahead and run offline.",
          ],
          [
            <code>[ALIGNAIR_MODEL_NOT_FOUND]</code>,
            "A misspelled model id or path.",
            <>
              Check the id against <DocLink to="choose-a-model">the model table</DocLink> or{" "}
              <code>alignair models list</code>.
            </>,
          ],
          [
            "CUDA out of memory, or the process is killed",
            "The batch is too large for the GPU, or the chunk for the RAM.",
            <>
              Lower <code>--batch-size</code> (GPU memory) or <code>--chunk-size</code> (RAM).
            </>,
          ],
          [
            "Slow on the CPU",
            "One process aligns all the reads.",
            <>
              Add <code>--threads N</code>; use a GPU if you have one; write <code>--columns minimal</code> if the calls
              are all you need (<DocLink to="run-at-scale">Run at scale</DocLink>).
            </>,
          ],
        ]}
      />

      <H2 id="t-f-in-pandas">A T/F column read wrong in pandas</H2>
      <p>
        A filter on <code>productive</code> or <code>segmentation_low_quality</code> keeps every row, or none. These
        columns hold the strings <code>T</code> and <code>F</code>, and <code>.astype(bool)</code> turns any non-empty
        string, <code>"F"</code> included, into True. Compare with the string:
      </p>
      <CodeBlock
        lang="python"
        code={`import pandas as pd

table = pd.read_csv("out.tsv", sep="\\t")
table = table[table["segmentation_low_quality"] != "T"]     # not: ~table[...].astype(bool)
productive = table[table["productive"] == "T"]               # an empty cell is unknown, not F`}
      />
      <p>
        Still stuck? <DocLink to="support">Support</DocLink> says where to ask; include the command, its full output
        with <code>--debug</code>, and the <code>out.tsv.run.json</code>.
      </p>
    </>
  ),
};

export const usePages: DocPage[] = [
  chooseModel,
  prepareInput,
  donorGenotype,
  ownReference,
  metadataJoin,
  outputColumnsChoice,
  runAtScale,
  fixCommonProblems,
];
