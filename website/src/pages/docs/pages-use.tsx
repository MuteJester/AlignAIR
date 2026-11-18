import type { ReactNode } from "react";
import { CodeBlock, Callout } from "../../components/ui";
import CodeTabs from "../../components/CodeTabs";
import { DocTable, DocLink, H2, H3, type DocPage } from "./doc-kit";
import ModelTable from "../../components/data/ModelTable";
import ColumnTable from "../../components/data/ColumnTable";

const SECTION = "Use on your data";

const OGRDB_SETS = "https://ogrdb.airr-community.org/germline_sets/Homo%20sapiens";
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
        model calls only alleles of its own locus, so the calls it gives a read of another locus mean nothing, even
        when the row looks complete. For 10x TCR contigs, the whole recipe (split by chain, one model per chain, the
        annotations carried, the chains paired by cell) is on{" "}
        <DocLink to="metadata-join?h=10x-tcr">Join sample and cell metadata</DocLink>.
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
        It aligns single reads: merge paired R1/R2 reads before aligning. BAM and CRAM files are not read. The models
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
        <code>ALIGNAIR_GENOTYPE_ALLELE_NOT_FOUND</code>). You have two ways to work with IMGT-named data: call with
        IMGT's set, or map the model's labels to IMGT names.
      </p>

      <H3 id="bind-imgt">Call with IMGT's set</H3>
      <p>
        Bind IMGT's germline set with <code>--reference</code>, and the calls carry IMGT's names. Immcantation's
        germline files (IMGT/GENE-DB, one FASTA per segment) are read as they are, headers included:
      </p>
      <CodeBlock
        code={`cat imgt_human_IGHV.fasta imgt_human_IGHD.fasta imgt_human_IGHJ.fasta > imgt_igh.fasta
alignair predict --model alignair4-igh-human --input reads.fasta --out out.imgt.tsv --reference imgt_igh.fasta`}
      />
      <CodeBlock
        lang="text"
        title="v_call on the example reads"
        code={`read     model's own set (OGRDB)        IMGT's set bound
read01   IGHVF10-G50*02                 IGHV3-11*04
read02   IGHVF10-G42*01,IGHVF10-G42*03  IGHV3-43D*04,IGHV3-43D*05
read03   IGHVF3-G12*02                  IGHV4-30-2*05`}
      />
      <p>
        The junctions of all 20 example reads are unchanged, and so are their D and J calls but one: read10's J becomes{" "}
        <code>IGHJ6*04</code>, an allele IMGT's set has and the model's lacks. A warning before the run names any
        allele whose anchor could not be placed.
      </p>

      <H3 id="what-the-set-lacks">What the IG sets lack</H3>
      <p>
        OGRDB's sets hold fewer alleles than IMGT's. The IGH model has 198 V, 33 D and 7 J alleles; IMGT's human IGH set
        (as GenAIRR ships it, pseudogenes and ORFs included) has 307, 47 and 14. The model lacks these J alleles:{" "}
        <code>IGHJ3*01</code>, <code>IGHJ4*01</code>, <code>IGHJ4*03</code>, <code>IGHJ5*01</code>,{" "}
        <code>IGHJ5*04</code>, <code>IGHJ6*01</code> and <code>IGHJ6*04</code>.
      </p>
      <p>
        A read of an allele the set lacks is called to the nearest allele it holds, usually another allele of the same
        gene (<code>IGHJ6*04</code> becomes <code>IGHJ6*02</code>), and its identity stays below 1 even when the read
        is unmutated. On 3,000 simulated unmutated reads drawn from IMGT's set:
      </p>
      <ul>
        <li>
          fewer than 2% of the reads of a J allele the model lacks got <code>j_identity</code> 1, against over 99% of
          the reads of a J it holds;
        </li>
        <li>
          with IMGT's set bound, 1,288 of the 1,305 reads of a V allele the model lacks got their V allele back, but only
          589 of the 1,503 reads of a missing J allele got their J allele.
        </li>
      </ul>

      <H3 id="map-names">Map the labels to IMGT names</H3>
      <p>
        OGRDB's current human sets, at <Ext href={OGRDB_SETS}>ogrdb.airr-community.org/germline_sets</Ext>, name their
        alleles by their IUIS (IMGT) names (an allele IUIS has not named carries an <code>i</code>, as in{" "}
        <code>IGHV3-30*i02</code>), and they hold the IGH model's sequences: each of its 238 alleles has the
        same sequence, or one containing it, in OGRDB's <code>IGH_VDJ</code> set (release 10). So matching sequences
        gives each label its IMGT name. Export the model's set, download OGRDB's "FASTA Ungapped" of the same locus
        (here saved as <code>ogrdb_igh.fasta</code>), and match them with this script:
      </p>
      <CodeBlock
        lang="python"
        title="map_names.py"
        code={`import sys

def read_fasta(path):
    seqs, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = ""
        elif name:
            seqs[name] += line.upper().replace(".", "").replace("-", "")
    return seqs

model, named = read_fasta(sys.argv[1]), read_fasta(sys.argv[2])
for label, seq in model.items():
    same = [n for n, s in named.items() if s == seq] or [n for n, s in named.items() if seq in s or s in seq]
    print(label, ",".join(same), sep="\\t")`}
      />
      <CodeBlock
        title="shell"
        code={`$ alignair reference export "$(alignair models path alignair4-igh-human)" --fasta igh_germline.fasta
wrote igh_germline.fasta
$ python map_names.py igh_germline.fasta ogrdb_igh.fasta > names.tsv
$ grep -E "G50\\*02|G42\\*0[13]" names.tsv
IGHVF10-G42*01	IGHV3-43D*05
IGHVF10-G42*03	IGHV3-43D*04
IGHVF10-G50*02	IGHV3-11*04`}
      />
      <p>
        <code>names.tsv</code> translates either way: an IMGT-named genotype or gene-usage table into the model's
        labels, or AlignAIR's calls into IMGT names. For IGK and IGL it is partial: about half of the model's alleles
        find a name this way (83 of 176 for IGK, 100 of 191 for IGL).
      </p>

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

      <H2 id="another-species">Mouse and other species</H2>
      <p>
        The pretrained models are human. <code>--reference</code> binds another germline set of the same species and
        locus, or your novel alleles; it does not make a human model call another species well. For mouse or any other
        species, train a model on its germline set, or fine-tune the human model of the same locus on it (
        <DocLink to="fine-tune">Fine-tune a model</DocLink>). On 300 simulated mouse TRB reads:
      </p>
      <DocTable
        head={["Model", "V allele right", "J allele right", "Junction exact", "Partial rows"]}
        rows={[
          ["Human TRB model, mouse TRB set bound with --reference", "296", "236", "246", "58"],
          ["Human TRB model fine-tuned on the mouse TRB set, 100 steps", "300", "293", "259", "0"],
        ]}
      />
      <p>
        The built-in germline sets to train on are named by species, locus and source, with <code>TCR</code> for the
        T-cell loci: <code>MOUSE_IGH_IMGT</code>, <code>MOUSE_IGK_IMGT</code>, <code>MOUSE_IGL_IMGT</code>,{" "}
        <code>MOUSE_TCRA_IMGT</code>, <code>MOUSE_TCRB_IMGT</code>, <code>MOUSE_TCRD_IMGT</code> and{" "}
        <code>MOUSE_TCRG_IMGT</code>, with a C57BL/6J set beside each. <code>alignair reference list</code> lists them:
      </p>
      <CodeBlock
        title="shell"
        code={`$ alignair reference list --species Mouse --chain TRB
2 built-in reference(s):
  name                         chain_type             species                has_d
  MOUSE_C57BL6J_TCRB_IMGT      TCR_BETA               Mouse C57BL/6J         True
  MOUSE_TCRB_IMGT              TCR_BETA               Mouse                  True`}
      />
      <p>
        To fine-tune the human TRB model on the mouse set, check the plan first: <code>--plan</code> prints what the run
        would do, without training. Drop it to train.
      </p>
      <CodeBlock
        title="shell (output abridged)"
        code={`$ alignair train --dataconfig MOUSE_TCRB_IMGT --init-from alignair4-trb-human --out mouse_trb --plan
training plan:
{
  "loci": [
    "TRB"
  ],
  "alleles": {
    "V": 24,
    "D": 2,
    "J": 16
  },
  ...
  "steps": 50000,
  "batch_size": 64
}`}
      />
      <p>
        The fine-tune in the table above ran 100 steps (<code>--steps 100</code>), not the preset's 50,000: about
        200 s on the CPU of an 8-core desktop (Intel i7-10700F, <code>--data-workers 4</code>). Check a model you
        trained with <DocLink to="evaluate">Evaluate your model</DocLink> before you rely on it.
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
              Train a model from your germline set, or fine-tune the human model of the same locus (
              <DocLink to="choose-a-model?h=another-species">Mouse and other species</DocLink>). Start at{" "}
              <DocLink to="do-i-need-to-train">Do I need to train?</DocLink>
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
        and its row says <code>length_cropped</code> = <code>T</code> (the cell is blank for a read that was not cut).
        The cut happens before AlignAIR orients the read, so which end is lost depends on the read's orientation:
      </p>
      <ul>
        <li>
          a <strong>forward</strong> read loses its 3' end. That is often only constant region; but when much sequence
          comes before the V (a long leader, UTR or primer), the J and the junction can be lost;
        </li>
        <li>
          a <strong>reverse-complemented</strong> read loses the 5' end of the molecule: the start of the V. Its J and
          junction are kept, and its V alignment starts further into the germline.
        </li>
      </ul>
      <p>
        A cut that falls after the J changes nothing AlignAIR reports. A cut that reaches the J shows in the row: the J
        alignment ends at the last base read, <code>j_sequence_end</code> = 575. The J call and the junction of such a
        row can be wrong even when it says <code>complete</code>, and a deeper cut leaves the row{" "}
        <code>partial</code>. In a test that put 100 to 320 nt of random sequence before four example reads, every
        row whose junction changed was one of these. So after a run, check these rows:
      </p>
      <CodeBlock
        lang="python"
        code={`import pandas as pd

table = pd.read_csv("out.tsv", sep="\\t")
check = table[(table["length_cropped"] == "T")
              & ((table["j_sequence_end"] >= 575) | (table["airr_assembly_status"] != "complete"))]`}
      />
      <p>
        When there are many, trim the reads to the V(D)J region before aligning (remove the leader, primers and
        constant region in your preprocessing).
      </p>
      <p>
        <strong>10x contigs.</strong> Cell Ranger's contigs carry the 5' UTR and leader before the V and part of the
        constant region after the J, so many run past 575 nt and come out <code>length_cropped</code> = <code>T</code>.
        That is expected: the contigs are forward, and the cut usually takes only constant region, which AlignAIR does
        not call (<code>c_call</code> comes from Cell Ranger's annotations, through{" "}
        <DocLink to="metadata-join">--metadata</DocLink>). In a test with 300 simulated TRB contigs given 10x-like
        flanks (80 to 160 nt before the V, 100 to 250 nt after the J, of random sequence), 271 came out cropped; every
        one kept the V call, J call and junction of the same contig without flanks, and none had{" "}
        <code>j_sequence_end</code> = 575. Contigs with a longer stretch before the V are the ones the check above
        finds.
      </p>

      <H2 id="presto-headers">Reads from pRESTO: CONSCOUNT and DUPCOUNT</H2>
      <p>
        A read's id is the first word of its header, so a pRESTO header such as{" "}
        <code>&gt;read01|CONSCOUNT=2|DUPCOUNT=5|PRCONS=IGHG</code> becomes the whole <code>sequence_id</code>, and its
        annotations do not become columns, as they do with Change-O's MakeDb. To get MakeDb's{" "}
        <code>consensus_count</code> and <code>duplicate_count</code>, split the headers into a plain id and a table
        of the annotations, then join the table with <code>--metadata</code>:
      </p>
      <CodeBlock
        code={`awk -F'|' 'BEGIN {OFS = "\\t"; print "sequence_id", "consensus_count", "duplicate_count", "prcons" > "annotations.tsv"}
  /^>/ {c = d = p = ""
        for (i = 2; i <= NF; i++) {split($i, kv, "="); if (kv[1] == "CONSCOUNT") c = kv[2]
                                   if (kv[1] == "DUPCOUNT") d = kv[2]; if (kv[1] == "PRCONS") p = kv[2]}
        id = substr($1, 2); print id, c, d, p > "annotations.tsv"; print ">" id; next}
  {print}' reads_presto.fasta > reads.fasta
alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv \\
  --metadata annotations.tsv --keep-columns consensus_count,duplicate_count,prcons`}
      />
      <CodeBlock
        lang="text"
        title="out.tsv (a few columns)"
        code={`sequence_id  v_call                         consensus_count  duplicate_count  prcons
read01       IGHVF10-G50*02                 2                5                IGHG
read02       IGHVF10-G42*01,IGHVF10-G42*03  3                6                IGHG`}
      />
      <p>
        The first command writes <code>reads.fasta</code> with plain ids and <code>annotations.tsv</code> with one
        row per read. Another annotation goes the same way: add it to the awk and to <code>--keep-columns</code>.
        Without <code>--keep-columns</code>, the two counts are kept and <code>prcons</code> is not (
        <DocLink to="metadata-join?h=kept-columns">which columns are kept</DocLink>).
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
# with_genotype returns a new Aligner; aligner itself is unchanged
donor = aligner.with_genotype({"V": ["IGHVF10-G50*02", "IGHVF10-G42*01", "IGHVF3-G12*02", "IGHVF6-G21*07"],
                               "J": ["IGHJ4*02", "IGHJ6*02", "IGHJ6*03"]})
ids, reads, info = read_sequences("reads.fasta")
donor.predict(reads, ids=ids).write_airr("out.tsv")`}
      />

      <H2 id="outside-the-genotype">Reads of genes the genotype leaves out</H2>
      <p>
        With a YAML or JSON genotype, every call of a segment it names comes from its alleles, so the genotype must
        hold every allele the donor carries. Here is the genotype above, which names 4 V alleles, on the 20 example
        reads, without it and with it:
      </p>
      <DocTable
        head={["Read", "No genotype: V, J", "With donor.yaml: V, J", "v_identity: without, with", "Status with donor.yaml"]}
        rows={[
          [
            "read01",
            <><code>IGHVF10-G50*02</code>, <code>IGHJ6*02</code></>,
            <><code>IGHVF10-G50*02</code>, <code>IGHJ6*02</code></>,
            "0.959, 0.959",
            "complete",
          ],
          [
            "read02",
            <><code>IGHVF10-G42*01,IGHVF10-G42*03</code>, <code>IGHJ4*02</code></>,
            <><code>IGHVF10-G42*01</code>, <code>IGHJ4*02</code></>,
            "0.969, 0.969",
            "complete",
          ],
          [
            "read04",
            <><code>IGHVF9-G32*01</code>, <code>IGHJ5*02</code></>,
            <><code>IGHVF10-G50*02</code>, <code>IGHJ4*02</code></>,
            "0.955, 0.790",
            "complete",
          ],
          [
            "read07",
            <><code>IGHVF1-G2*01</code>, <code>IGHJ3*02</code></>,
            <>no V call, <code>IGHJ6*02</code></>,
            "0.956, blank",
            <>partial, <code>no_v</code></>,
          ],
        ]}
      />
      <ul>
        <li>read01 carries an allele of the genotype: its call is unchanged.</li>
        <li>read02's tie is settled: <code>IGHVF10-G42*03</code> is not in the genotype.</li>
        <li>
          read04 is a read of a gene the genotype leaves out. It is called to the closest allele the genotype allows,
          and its row is still <code>complete</code>. Its J is forced the same way. What gives the forced call away is
          its identity: <code>v_identity</code> falls from 0.955 to 0.790, and <code>j_identity</code> from 1.0 to
          0.914 (<DocLink to="donor-genotype?h=spot-a-forced-call">spot a forced call</DocLink>).
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

      <H3 id="spot-a-forced-call">Spot a forced call</H3>
      <p>
        A forced call matches the read less well than its true allele would, so its <code>v_identity</code> drops.
        Align the reads once without the genotype and once with it, and compare the two read by read:
      </p>
      <CodeBlock
        lang="python"
        code={`import pandas as pd

columns = ["sequence_id", "v_call", "v_identity"]
before = pd.read_csv("out.tsv", sep="\\t", usecols=columns)
after = pd.read_csv("out.donor.tsv", sep="\\t", usecols=columns)
both = before.merge(after, on="sequence_id", suffixes=("", "_genotype"))
forced = both[both["v_identity"] - both["v_identity_genotype"] > 0.05]`}
      />
      <p>
        On the example reads this finds read04 (0.955 to 0.790), read10 (0.962 to 0.841), read11 (0.969 to 0.843) and
        nine more of the reads whose gene <code>donor.yaml</code> leaves out; two more forced calls drop by less than
        0.05, so choose the threshold for your data. Within one run, rows whose{" "}
        <code>v_identity</code> sits far below the other rows of their gene point the same way.
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
        Here the genotype uses IMGT names on an IG model. Write it with the model's labels instead (
        <DocLink to="choose-a-model?h=map-names">map the names</DocLink>), or bind a germline set that holds these
        names with{" "}
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
        <code>alignair genotype</code> infers a donor's V genotype from any tool's AIRR table, by a frequency rule like
        that of TIgGER's <code>inferGenotype</code>. For each V gene it counts the rows whose <code>v_call</code> is a
        single allele (a tie counts for none). The gene keeps its most frequent alleles until they explain 87.5% of
        those rows (<code>--explain</code>; TIgGER's default share too). It writes an AIRR GenotypeSet;{" "}
        <code>--locus</code> sets its locus (default IGH). Two things differ from TIgGER:
      </p>
      <ul>
        <li>
          it counts every single-allele row, mutated or not, where <code>inferGenotype</code> by default counts only
          unmutated reads. Somatic mutations can move a read onto a sibling allele, and those reads count here;
        </li>
        <li>
          a gene with fewer than 10 such rows (<code>--min-reads</code>, AlignAIR's own cutoff, not TIgGER's) is left
          open: no evidence, so no genotype for it.
        </li>
      </ul>
      <p>
        On a mutated B-cell repertoire, compare a few genes with TIgGER's genotype, or keep only the rows with a low{" "}
        <code>mutation_rate</code> in the table you give <code>alignair genotype</code> (it reads any AIRR table).
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
      <p>
        A GenotypeSet restricts only the genes it names: a gene it left open keeps all its alleles (TIgGER's
        reassignAlleles, by contrast, moves the reads of genes its genotype leaves out). A YAML or JSON genotype is
        stricter: each segment it names is limited to the alleles it lists.
      </p>

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
        for an IG locus (<DocLink to="choose-a-model?h=bind-imgt">call with IMGT's set</DocLink>), or a set with your
        donor's novel alleles all work. The set must be of the model's locus and species: for another species, see{" "}
        <DocLink to="choose-a-model?h=another-species">Mouse and other species</DocLink>; for another locus,{" "}
        <DocLink to="do-i-need-to-train">Do I need to train?</DocLink>
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
          J), else from the one V, D or J letter in the name. An IMGT/GENE-DB header, as in Immcantation's germline
          files (<code>&gt;X00369|IGHV3-11*04|Homo sapiens|F|V-REGION|...</code>), is read as it is: the allele's
          name is its first <code>|</code> field that holds a <code>*</code>, as Change-O reads it.
        </li>
        <li>
          <strong>One sequence per name.</strong> A name given twice with two sequences is refused.
        </li>
        <li>
          <strong>Gapped or not.</strong> Sequences may be IMGT-gapped (with <code>.</code> or <code>-</code>) or
          ungapped. A gapped V keeps its gaps. Gapped is best: Change-O's CreateGermlines.py and TIgGER need the V
          alleles gapped, and a gapped V needs no gaps filled in (see below).
        </li>
      </ul>

      <H3 id="one-file">One germline file for the whole analysis</H3>
      <p>
        The easiest start is the model's own set, exported gapped, to which you add your novel alleles (TIgGER's novel
        alleles, the <code>novel_imgt</code> sequences of <code>findNovelAlleles</code>, are already gapped):
      </p>
      <CodeTabs
        cli={`alignair reference export "$(alignair models path alignair4-igh-human)" --fasta germline.fasta --gapped
cat novel_alleles.fasta >> germline.fasta
alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv --reference germline.fasta`}
        python={`from alignair import Aligner, ReferenceSet, read_sequences

reference = ReferenceSet.from_fasta("germline.fasta")
aligner = Aligner.from_pretrained("alignair4-igh-human", reference=reference)
# or, for an Aligner already loaded (returns a new one):
# aligner = aligner.bound_to(reference)
ids, reads, info = read_sequences("reads.fasta")
aligner.predict(reads, ids=ids).write_airr("out.tsv")`}
      />
      <p>
        Then give the same <code>germline.fasta</code> to every tool downstream: to TIgGER as its germline set, and to
        CreateGermlines.py (<code>-r germline.fasta</code>), which reads AlignAIR's table as it is. The gapped export
        alone calls exactly as the model's own set does, and an ungapped novel allele binds too, but CreateGermlines.py
        needs it gapped.
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
CAGGTGCAGCTGGTGGAGTCTGAGAGA...GGCTTGGTCAAGCCTGGAGGGTCCCTGAGACTCTCCTGTGCAG...`}
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
              Not checked: it binds when it has the model's segments, but a human model calls another species' reads
              worse than a model trained or fine-tuned for it (
              <DocLink to="choose-a-model?h=another-species">Mouse and other species</DocLink>).
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
        With a <code>--reference</code> run, the germline set to give downstream tools is your FASTA, not this export
        (<DocLink to="own-reference?h=one-file">one germline file</DocLink>).
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
        go to <code>--input</code> and the annotations to <code>--metadata</code>. Cell Ranger's contig file holds
        every chain of the run, and a model calls one locus, so give each model the contigs of its own chain (
        <code>contigs_IGH.fasta</code> here; the next section shows the split):
      </p>
      <CodeTabs
        cli={`alignair predict --model alignair4-igh-human --input contigs_IGH.fasta --out out.tsv \\
  --metadata filtered_contig_annotations.csv`}
        python={`from alignair import Aligner, AirrWriter
from alignair.io import build_metadata_index, iter_sequence_batches, lookup_metadata_rows

aligner = Aligner.from_pretrained("alignair4-igh-human")
index, columns = build_metadata_index("filtered_contig_annotations.csv", normalize_10x=True)
with index, AirrWriter("out.tsv", locus=aligner.default_locus, extra_columns=columns) as writer:
    for batch in iter_sequence_batches("contigs_IGH.fasta"):
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
        checked against the calls.
      </p>

      <H2 id="10x-tcr">10x TCR contigs, chain by chain</H2>
      <p>
        A model of one locus gives reads of another locus rows that look real. Of 300 simulated TRA contigs aligned
        with the TRB model, 85 came out <code>complete</code> with a TRBV call and 21 <code>productive</code>, and the
        run gave no warning; only their <code>v_identity</code> (median 0.79, against 1.0 for the TRB contigs) gave
        them away. So split Cell Ranger's contigs by the annotations' <code>chain</code> column, and align each chain
        with its own model. The split keeps each contig's header as it is, so the ids still match the annotations:
      </p>
      <CodeBlock
        code={`for chain in TRA TRB; do
  awk -F, -v chain="$chain" 'NR == 1 {for (i = 1; i <= NF; i++) {if ($i == "chain") c = i; if ($i == "contig_id") k = i}; next}
      $c == chain {print $k}' filtered_contig_annotations.csv > "$chain.ids"
  awk 'NR == FNR {keep[$1]; next} /^>/ {p = (substr($1, 2) in keep)} p' "$chain.ids" filtered_contig.fasta > "contigs_$chain.fasta"
  model="alignair4-$(echo "$chain" | tr 'A-Z' 'a-z')-human"
  alignair predict --model "$model" --input "contigs_$chain.fasta" --out "$chain.tsv" \\
    --metadata filtered_contig_annotations.csv --threads 4
done
head -n 1 TRA.tsv > tcr.tsv
for chain in TRA TRB; do tail -n +2 "$chain.tsv"; done >> tcr.tsv`}
      />
      <ul>
        <li>
          The first <code>awk</code> lists the <code>contig_id</code> of each annotation row of the chain; the second
          keeps those contigs of the FASTA.
        </li>
        <li>
          Add <code>TRG</code> and <code>TRD</code> to the loop for gamma-delta libraries, or loop over{" "}
          <code>IGH IGK IGL</code> for B cells. Contigs whose <code>chain</code> is <code>Multi</code>, or a chain no
          model covers, are left out.
        </li>
        <li>
          Each table carries <code>cell_id</code>, <code>umi_count</code>, <code>consensus_count</code> and{" "}
          <code>c_call</code> from the annotations. The tables have the same columns, so <code>tcr.tsv</code> is one
          table under one header, with each row's <code>locus</code>.
        </li>
      </ul>
      <p>
        AlignAIR does not pair chains. To pair them, join the rows of the two loci on <code>cell_id</code>; a cell
        with two chains of one locus keeps here the one with the most UMIs:
      </p>
      <CodeBlock
        lang="python"
        code={`import pandas as pd

table = pd.read_csv("tcr.tsv", sep="\\t")
table = table[table["airr_assembly_status"] == "complete"]
top = table.sort_values("umi_count", ascending=False).drop_duplicates(["cell_id", "locus"])
pairs = top[top["locus"] == "TRA"].merge(top[top["locus"] == "TRB"], on="cell_id", suffixes=("_tra", "_trb"))
print(pairs[["cell_id", "v_call_tra", "junction_aa_tra", "v_call_trb", "junction_aa_trb"]])`}
      />
      <p>
        Tables from several 10x runs need a <code>sample_id</code> before you merge them (
        <DocLink to="metadata-join?h=sample-id">A sample id for every row</DocLink>), and cells are then keyed by{" "}
        <code>sample_id</code> and <code>cell_id</code> together.
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

      <H3 id="sample-id">A sample id for every row</H3>
      <p>
        The metadata is per read. To give every row of a sample's table the same <code>sample_id</code>, add the
        column after the run. Tables you merge need it: 10x barcodes repeat across runs, so{" "}
        <code>AAACCTGAGAAACCAT-1</code> can be a cell of every sample. With one table per sample:
      </p>
      <CodeBlock
        code={`for sample in S1 S2 S3; do
  awk -v id="$sample" 'BEGIN {FS = OFS = "\\t"} NR == 1 {print $0, "sample_id"; next} {print $0, id}' \\
    "$sample.tsv" > "$sample.with_id.tsv"
done
head -n 1 S1.with_id.tsv > all.tsv
for sample in S1 S2 S3; do tail -n +2 "$sample.with_id.tsv"; done >> all.tsv`}
      />
      <p>
        Skip this when the metadata table already gave each read a <code>sample_id</code> (it is one of the columns
        kept by default).
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
        <code>productive</code>, <code>rev_comp</code>, <code>vj_in_frame</code>, <code>stop_codon</code> and{" "}
        <code>segmentation_low_quality</code> hold the AIRR strings <code>T</code> and <code>F</code>; an empty cell
        means unknown, not false. <code>length_cropped</code> is <code>T</code> or empty, and empty means the read was
        not cut. pandas reads them as strings, so compare with <code>"T"</code>, never with{" "}
        <code>.astype(bool)</code>, which turns <code>"F"</code> into True:
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
      <p>
        A GPU and the CPU give the same calls and junctions, but their arithmetic differs in the last digits: on the
        example reads, <code>mutation_rate</code> differs by up to 0.00001, and on a short fragment a candidate set (
        <code>v_call_set</code>) can differ. The run record (<code>out.tsv.run.json</code>) keeps the device each run
        used.
      </p>

      <H2 id="cpu-threads">On the CPU: --threads</H2>
      <p>
        Without <code>--threads</code>, one process runs, with batches of 64 reads, on PyTorch's default number of
        threads: one per physical core of the machine, or of the CPUs the job is bound to. On a shared node whose
        scheduler does not bind jobs to their CPUs, that is more cores than your job was given.
      </p>
      <p>
        <code>--threads N</code> runs N worker processes of one thread each, with batches of 16 reads per worker; this
        scales better than one process using many threads. Set N to the cores your job was given (on SLURM,{" "}
        <code>$SLURM_CPUS_PER_TASK</code>), and leave some free on a shared desktop. On a GPU,{" "}
        <code>--threads</code> only sets the CPU threads of the one process.
      </p>
      <CodeBlock code={`alignair predict --model alignair4-trb-human --input tcr.fasta --out tcr.tsv --threads 8`} />
      <p>
        Worker processes are a command-line feature: in Python, one process predicts. On a CPU cluster, the command
        line with <code>--threads</code> suits large samples; the Python loop, which loads the model once, suits many
        small ones (a command-line run of 20 reads takes about 2 s, start-up and model load included).
      </p>

      <H3 id="cpu-sizing">Speed and memory on the CPU</H3>
      <p>
        To size a job on a CPU cluster or an HPC node: the throughput (reads per second) and peak memory of each
        setting, measured on 24,000 simulated IGH reads of about 380 nt, on the CPU of an 8-core desktop (Intel
        i7-10700F, 16 threads, with other jobs running; a range is of two runs):
      </p>
      <DocTable
        head={["Run", "Reads per second", "Peak memory"]}
        rows={[
          ["Without --threads", "195 to 265", "1.4 GB"],
          [<code>--threads 1</code>, "113", "1.3 GB"],
          [<code>--threads 4</code>, "250", "1.6 GB"],
          [<code>--threads 8</code>, "340 to 415", "2.2 GB"],
        ]}
      />
      <p>
        So with <code>--threads 8</code>, a million reads take 40 to 50 minutes on such a machine. TRB contigs of about
        350 nt ran at much the same speed. Memory is for the default chunk of 20,000 reads, the model included; it
        does not grow with the file. Ask a scheduler for 4 GB with 8 threads.
      </p>

      <H2 id="memory">Large files in constant memory</H2>
      <p>
        The command line streams its input: it reads 20,000 reads at a time (<code>--chunk-size</code>), aligns and
        writes them, then reads the next chunk. Memory depends on the chunk and batch sizes and the number of workers,
        not on the size of the file (only the read ids are remembered, to rename repeats); the figures are{" "}
        <DocLink to="run-at-scale?h=cpu-sizing">above</DocLink>. The table is written to a temporary file and moved into
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
  alignair predict --model alignair4-igh-human --input "$input" --out "results/\${sample}.tsv" \\
    --threads 8 --offline --quiet
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
        of the command line writes its sample's own <code>.run.json</code>. To give each table its sample's id, see{" "}
        <DocLink to="metadata-join?h=sample-id">A sample id for every row</DocLink>.
      </p>

      <H3 id="slurm-array">A SLURM job array</H3>
      <p>
        On a cluster, run one job per sample. With the same manifest, this script aligns the sample on line{" "}
        <code>$SLURM_ARRAY_TASK_ID</code> + 1 (line 1 is the header) on the cores the job was given:
      </p>
      <CodeBlock
        lang="bash"
        title="align.sh"
        code={`#!/bin/bash
#SBATCH --job-name=alignair
#SBATCH --array=1-24
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=02:00:00

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" manifest.tsv)   # line 1 is the header
sample=$(printf '%s\\n' "$line" | cut -f1)
input=$(printf '%s\\n' "$line" | cut -f2)
alignair predict --model alignair4-igh-human --input "$input" --out "results/\${sample}.tsv" \\
  --threads 8 --offline --quiet`}
      />
      <p>
        Keep <code>--threads</code> equal to <code>--cpus-per-task</code>, and set <code>--array</code> to{" "}
        <code>1-</code> the number of samples. Download the model once on a node with
        internet, into the cache every job reads (<code>ALIGNAIR_CACHE_DIR</code> on a shared disk), then submit:
      </p>
      <CodeBlock
        code={`alignair models get alignair4-igh-human
mkdir -p results
sbatch align.sh`}
      />
      <p>
        4 GB and two hours cover a sample of a million reads at the speeds measured above; scale them to your
        samples.
      </p>

      <H3 id="apptainer">Apptainer or Singularity</H3>
      <p>
        Clusters that do not run Docker usually run Apptainer (Singularity), which runs the Docker image as it is.
        Pull it once into a file, then put <code>apptainer exec alignair.sif</code> before any command on these pages:
      </p>
      <CodeBlock
        code={`apptainer pull alignair.sif docker://alignair/alignair:<version>
apptainer exec alignair.sif alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv \\
  --threads 8`}
      />
      <p>
        Apptainer runs as you, with your home directory mounted, so the model is cached in your own{" "}
        <code>~/.cache/alignair</code> (or <code>ALIGNAIR_CACHE_DIR</code>), and the image runs on the CPU. With
        Singularity, write <code>singularity</code> for <code>apptainer</code>.
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
              Rows with <code>length_cropped</code> = <code>T</code> are <code>partial</code>, or their{" "}
              <code>j_sequence_end</code> is 575
            </>,
            "The reads are longer than 575 nt, and the cut reached the J: the J call and junction may be wrong.",
            <>
              Trim reads to the V(D)J region first (<DocLink to="prepare-input?h=long-reads">long reads</DocLink>).
              Cropped rows whose J ends before 575, as on most 10x contigs, are fine.
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
              Add <code>--threads N</code>, N the cores you have; use a GPU if you have one; write{" "}
              <code>--columns minimal</code> if the calls
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
