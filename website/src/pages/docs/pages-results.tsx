import type { ReactNode } from "react";
import { CodeBlock, Callout } from "../../components/ui";
import ComparisonTable from "../../components/data/ComparisonTable";
import { DocTable, DocLink, H2, H3, type DocPage } from "./doc-kit";

const SECTION = "Understand the results";

const ASSAY = "https://github.com/MuteJester/AlignAIR/tree/main/src/alignair_assay";
const PRJNA555323 = "https://www.ncbi.nlm.nih.gov/bioproject/PRJNA555323";
const RODRIGUEZ = "https://doi.org/10.1038/s41467-023-40070-x";

function Ext({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noreferrer" className="text-brand-600 underline">
      {children}
    </a>
  );
}

/* ------------------------------------------------------------------------------------------------------------------ */
/* Output columns explained                                                                                           */
/* ------------------------------------------------------------------------------------------------------------------ */
const outputColumns: DocPage = {
  slug: "output-columns",
  title: "Output columns explained",
  section: SECTION,
  lead: "What the columns you read first say, how far each one goes, and how to keep only the rows your analysis can use.",
  body: () => (
    <>
      <p>
        Each row of the table is one read. The columns on this page carry most of what a repertoire analysis uses; the{" "}
        <DocLink to="column-dictionary">output column dictionary</DocLink> lists all 99. The examples are the reads of{" "}
        <DocLink to="first-alignment">Your first alignment</DocLink>, aligned with the IGH model.
      </p>

      <H2 id="calls">The calls: v_call, d_call, j_call</H2>
      <CodeBlock
        lang="text"
        title="out.tsv"
        code={`sequence_id  v_call                         d_call       j_call
read01       IGHVF10-G50*02                 IGHD3-10*01  IGHJ6*02
read02       IGHVF10-G42*01,IGHVF10-G42*03  IGHD2-2*02   IGHJ4*02
read03       IGHVF3-G12*02                  IGHD5-18*02  IGHJ6*03`}
      />
      <ul>
        <li>
          <strong>
            <code>v_call</code>
          </strong>{" "}
          names the called V allele first. The alleles after it, comma-separated, match as many of the read's V bases
          as the call does, so the read alone cannot choose among them; IgBLAST lists equally scoring alleles the same
          way, and TIgGER and Change-O read the list as it is. read02 cannot tell <code>IGHVF10-G42*01</code> from{" "}
          <code>IGHVF10-G42*03</code>.
        </li>
        <li>
          <strong>
            <code>d_call</code> and <code>j_call</code>
          </strong>{" "}
          name one allele each, with no list after it. The other D or J alleles AlignAIR weighed are in the candidate
          set (next section).
        </li>
        <li>
          <strong>Allele names.</strong> The IG models write OGRDB labels (<code>IGHVF10-G50*02</code>), the TCR models
          IMGT names; see <DocLink to="concepts?h=allele-names">OGRDB and IMGT allele names</DocLink>, and{" "}
          <DocLink to="choose-a-model?h=map-names">Map the labels to IMGT names</DocLink> to translate them.
        </li>
        <li>
          <strong>A blank call.</strong> A blank <code>v_call</code> means the read holds no V that AlignAIR could
          place (its row is <code>partial</code>, reason <code>no_v</code>). A blank <code>d_call</code> means the read
          shows no D, or AlignAIR found fewer than 8 of its bases (<DocLink to="trust-the-calls?h=short-d">D calls on short D
          segments</DocLink>).
        </li>
        <li>
          <code>locus</code> is the model's locus, on every row. <code>c_call</code> stays blank: AlignAIR does not
          call the constant region, unless a <DocLink to="metadata-join">metadata table</DocLink> fills it.
        </li>
      </ul>

      <H2 id="candidate-set">The candidate set: v_call_set, d_call_set, j_call_set</H2>
      <p>
        For each segment, the candidate set holds the alleles the model scored at least half as likely as its best one,
        best first, three at most. It is made before the read is aligned to the germline, and the call comes after
        that alignment. So the two can differ:
      </p>
      <ul>
        <li>
          the call can fall outside the set: the alignment can replace the model's first choice by an allele that
          matches more of the read's bases, such as a novel allele you bound with <code>--reference</code>;
        </li>
        <li>
          <code>v_call</code> can list more alleles than the set, as its list of equal matches has no limit.
        </li>
      </ul>
      <p>
        On a short read both happen. The last 150 nt of read01 (from FR3 to the J) match 17 V alleles equally well,
        the full read's allele (<code>IGHVF10-G50*02</code>) among them, while the set holds three alleles without it:
      </p>
      <CodeBlock
        lang="text"
        title="out.tsv"
        code={`sequence_id     v_call                                                          v_call_set
read01_last150  IGHVF10-G49*01,IGHVF10-G37*09,IGHVF10-G37*10,... (17 alleles)  IGHVF10-G49*01,IGHVF10-G51*02,IGHVF10-G49*03`}
      />
      <p>Read the set as what the model weighed, not as a proof:</p>
      <ul>
        <li>several members mean the model could not settle on one allele;</li>
        <li>it is not exhaustive: it stops at three, and a read may support more alleles than that;</li>
        <li>one member does not by itself make a confident call: no other allele came within half the best's score;</li>
        <li>
          its members need not share a gene or a family, so collapse it to a gene only when every member has that
          gene.
        </li>
      </ul>
      <p>
        In Python, <code>predict(threshold=0.5, cap=3)</code> sets this rule (<DocLink to="python-api">Python
        API</DocLink>). A <DocLink to="donor-genotype">donor genotype</DocLink> narrows both the call and the set to the
        donor's alleles.
      </p>

      <H2 id="junction-and-productivity">The junction and productivity</H2>
      <CodeBlock
        lang="text"
        title="out.tsv"
        code={`sequence_id  junction_aa                  junction_length  vj_in_frame  stop_codon  productive
read01       CATFYYGSGSYYNFVSNYNYYGMDVW   78               T            F           T
read02       CAKGTSCNSTSCYTIDYW           54               T            F           T`}
      />
      <DocTable
        head={["Column", "What it says"]}
        rows={[
          [
            <>
              <code>junction</code>, <code>junction_aa</code>, <code>junction_length</code>
            </>,
            <>
              The junction, as AIRR defines it: from the first base of the V's conserved Cys codon to the last base of
              the J's Trp or Phe codon. AlignAIR finds the two codons in the read itself. The junction is blank when
              either codon is not wholly inside the read, or the row has no V call. <code>cdr3</code> is the junction
              without the two codons.
            </>,
          ],
          [<code>vj_in_frame</code>, <><code>T</code> when the junction's length is a multiple of 3.</>],
          [
            <code>stop_codon</code>,
            <>
              <code>T</code> when a stop codon lies in the junction's reading frame anywhere from the start of the V to
              the end of the J in the read.
            </>,
          ],
          [
            <code>productive</code>,
            <>
              <code>T</code> when in frame with no stop codon, <code>F</code> otherwise, and blank when the read has no
              junction: blank means unknown, not unproductive. <code>productive_prediction</code> holds the same value.
            </>,
          ],
        ]}
      />
      <p>
        All three read the junction's frame: <code>vj_in_frame</code> from the junction's length, and{" "}
        <code>stop_codon</code> from the codons in that frame between the V's start and the J's end. An indel inside the
        V or the J that shifts the frame is not seen (<DocLink to="trust-the-calls?h=other-limits">other limits</DocLink>
        ).
      </p>

      <H2 id="identity-and-mutation">Identity and mutation</H2>
      <CodeBlock
        lang="text"
        title="out.tsv"
        code={`sequence_id  v_identity          d_identity  j_identity          mutation_rate
read01       0.958904109589041   1.0         0.9830508474576272  0.032617684453725815`}
      />
      <ul>
        <li>
          <code>v_identity</code>, <code>d_identity</code> and <code>j_identity</code> are the fraction of the aligned
          segment's bases that match the called germline, from 0 to 1, as AIRR defines them. IgBLAST writes a
          percentage: a filter <code>{"v_identity >= 98"}</code> becomes <code>{"v_identity >= 0.98"}</code>.{" "}
          <code>d_identity</code> is blank for a D too short to call.
        </li>
        <li>
          <code>mutation_rate</code> is AlignAIR's estimate of the share of the read's V, D and J bases that differ
          from the germline, from 0 to 1. The model estimates it from the read; it is not counted from the alignment,
          so it need not equal 1 − <code>v_identity</code>.
        </li>
        <li>
          T cells do not hypermutate: on the TCR models an identity well below 1 points to sequencing errors or to an
          allele the germline set lacks (<DocLink to="trust-the-calls?h=novel-alleles">novel alleles</DocLink>).
        </li>
      </ul>

      <H2 id="assembly-status">Assembly status: is the row complete?</H2>
      <p>
        After the calls, AlignAIR builds the rest of the row: the IMGT-gapped alignments, the regions, the identities.
        Three columns say how that went:
      </p>
      <DocTable
        head={["Column", "Values"]}
        rows={[
          [
            <code>airr_assembly_status</code>,
            <>
              <code>complete</code>: every field was derived. <code>partial</code>: the calls stand, but some fields
              are blank. <code>failed</code>: an unexpected error stopped the row; it keeps its calls.
            </>,
          ],
          [
            <code>airr_assembly_reason</code>,
            <>
              Why a row is partial, such as <code>no_v</code> (no V call) or <code>collapsed_segment</code>. Every reason
              is explained in <DocLink to="trust-the-calls?h=partial-rows">Partial rows</DocLink>.
            </>,
          ],
          [<code>airr_assembly_error</code>, <>The error that stopped a <code>failed</code> row.</>],
          [
            <code>segmentation_low_quality</code>,
            <>
              <code>T</code> when the read's V or J region came out empty. Such a row is never <code>complete</code>,
              and its call for the empty segment means nothing.
            </>,
          ],
          [
            <code>length_cropped</code>,
            <>
              <code>T</code> when the read was longer than 575 nt and only its first 575 nt were aligned; blank
              otherwise, never <code>F</code> (<DocLink to="prepare-input?h=long-reads">reads longer than 575 nt</DocLink>
              ). Cell Ranger's contigs often pass 575 nt, as they carry the constant region after the J: many 10x rows
              are <code>T</code>, and the cut usually takes only constant region, which AlignAIR does not call.{" "}
              <DocLink to="prepare-input">Prepare your input</DocLink> shows how to find the rows where it took more.
            </>,
          ],
        ]}
      />
      <p>
        The console line counts the partial and failed rows, for example{" "}
        <code>aligned 19 reads (0 dropped) -&gt; out.tsv; 0 failed / 10 partial AIRR assemblies tagged</code>. Only
        the <code>full</code> preset, the default, writes these columns (
        <DocLink to="output-columns-choice?h=presets">Choose output columns</DocLink>).
      </p>

      <H2 id="orientation">Orientation: rev_comp, orientation, input_sequence</H2>
      <p>
        AlignAIR turns every read forward itself, so you never orient reads first. The table says how each one came:
      </p>
      <DocTable
        head={["The read came", <code>orientation</code>, <code>sequence</code>, <code>rev_comp</code>, "Positions count on"]}
        rows={[
          ["forward", <code>forward</code>, "the read", <code>F</code>, <code>sequence</code>],
          [
            "reverse-complemented",
            <code>reverse_complement</code>,
            "the read as given",
            <code>T</code>,
            <>
              the reverse complement of <code>sequence</code>
            </>,
          ],
          [
            "complemented, or reversed",
            <>
              <code>complement</code>, <code>reverse</code>
            </>,
            <>
              the read turned forward (the read as given is in <code>input_sequence</code>)
            </>,
            <code>F</code>,
            <code>sequence</code>,
          ],
        ]}
      />
      <p>
        <code>rev_comp</code> follows the AIRR convention, as IgBLAST's does: when it is <code>T</code>, every
        position and alignment refers to the reverse complement of <code>sequence</code>.
      </p>

      <H2 id="filter-usable-rows">Keep the rows your analysis can use</H2>
      <p>
        The true/false columns (<code>productive</code>, <code>rev_comp</code>, <code>vj_in_frame</code>,{" "}
        <code>stop_codon</code>, <code>segmentation_low_quality</code>) hold the AIRR strings <code>T</code> and{" "}
        <code>F</code>, and a blank cell means unknown. <code>length_cropped</code> is <code>T</code> or blank, and
        blank means the read was not cut. Compare them with the string <code>"T"</code>.
      </p>
      <p>
        First, with 10x contigs, keep the rows whose <code>chain</code> (Cell Ranger's, joined with{" "}
        <code>--metadata</code>) equals <code>locus</code>: a model gives reads of another locus rows that can look
        real (<DocLink to="trust-the-calls?h=off-target">reads of another locus</DocLink>). Then, which rows to keep
        depends on the analysis:
      </p>
      <DocTable
        head={["Analysis", "Keep"]}
        rows={[
          [
            "Junctions, CDR3s, clones (Change-O, SCOPer)",
            <>
              Rows with <code>airr_assembly_status</code> <code>complete</code>.
            </>,
          ],
          [
            "The productive repertoire",
            <>
              Complete rows with <code>productive</code> <code>T</code>.
            </>,
          ],
          [
            "Gene usage, genotyping (TIgGER)",
            <>
              Rows with a <code>v_call</code>, partial ones included, except those with{" "}
              <code>segmentation_low_quality</code> <code>T</code>. Count D and J on these rows only. On TCR rows, also
              leave out a low <code>v_identity</code> (below 0.9 here).
            </>,
          ],
        ]}
      />
      <CodeBlock
        lang="python"
        code={`import pandas as pd

table = pd.read_csv("out.tsv", sep="\\t")
if "chain" in table:                                   # 10x: Cell Ranger's chain column, joined by --metadata
    table = table[table["chain"] == table["locus"]]    # drops the other chains and Multi
complete = table[table["airr_assembly_status"] == "complete"]
productive = complete[complete["productive"] == "T"]   # blank: unknown, not F
usage = table[table["v_call"].notna() & (table["segmentation_low_quality"] != "T")]
tcr = usage["locus"].str.startswith("TR")
usage = usage[~tcr | (usage["v_identity"] >= 0.9)]     # T cells do not hypermutate
j_usage = usage["j_call"].str.split("*").str[0].value_counts()   # J genes, on rows with a V call`}
      />
      <p>
        A complete row never has <code>segmentation_low_quality</code> <code>T</code>, so the first two filters need no
        check on it. It matters when you keep partial rows: there it marks the rows whose V or J call means nothing.
      </p>
      <p>
        A J call stands on a row without a V call, but that row is often a read of another locus. On 300 simulated TRA
        contigs aligned with the TRB model, counting J on every row with a J call (except{" "}
        <code>segmentation_low_quality</code> <code>T</code>) kept 277 meaningless TRBJ calls; the filter above keeps
        4. T cells do not hypermutate, so a TCR row far from its germline is suspect: every simulated TRA and TRB contig
        aligned with its own model had a <code>v_identity</code> of at least 0.98, while the reads of another locus
        that got a V call had a median of 0.75 to 0.79. The 0.9 is an example: set it from your own rows.
      </p>
      <Callout kind="warning" title="Never .astype(bool) a T/F column">
        Any non-empty string is true in Python, so <code>.astype(bool)</code> turns every <code>"F"</code> into{" "}
        <code>True</code>. The filter <code>~table["segmentation_low_quality"].fillna(False).astype(bool)</code> keeps
        no row at all. Compare with <code>"T"</code>, or read the file with{" "}
        <code>pd.read_csv(..., true_values=["T"], false_values=["F"])</code>.
      </Callout>

      <H2 id="every-other-column">Every other column</H2>
      <p>
        The rest of the table follows the AIRR standard, counting as Change-O's MakeDb does where the standard leaves a
        choice:
      </p>
      <ul>
        <li>
          the IMGT-gapped <code>sequence_alignment</code> and <code>germline_alignment</code>, with N in the germline
          for the untemplated bases between the segments;
        </li>
        <li>
          each segment's start and end in the read and in its germline, 1-based and inclusive (the V's germline
          positions count in the IMGT-gapped V);
        </li>
        <li>the CIGAR strings, each over the whole read;</li>
        <li>
          the N regions (<code>np1</code>, <code>np2</code>); a row without a D call holds every base between V and J in{" "}
          <code>np1</code>;
        </li>
        <li>the framework and CDR regions, with their positions in the read.</li>
      </ul>
      <p>
        The{" "}
        <DocLink to="column-dictionary">output column dictionary</DocLink> describes all 99 columns, with their units
        and the presets that hold them; <DocLink to="output-columns-choice">Choose output columns</DocLink> shows how to
        write fewer.
      </p>
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* How far to trust the calls                                                                                         */
/* ------------------------------------------------------------------------------------------------------------------ */
const trustTheCalls: DocPage = {
  slug: "trust-the-calls",
  title: "How far to trust the calls",
  section: SECTION,
  lead: "Where AlignAIR's calls hold and where they do not: each known limit, how to spot it in the table, what to do about it, and when to check with IgBLAST.",
  body: () => (
    <>
      <p>
        AlignAIR reports what a read supports and marks what it cannot settle. Each limit below says what happens, how
        it shows in the table and what to do. How often AlignAIR is right, measure by measure, against IgBLAST on reads
        with known truth, is on <DocLink to="comparison">Comparison with IgBLAST</DocLink>.
      </p>
      <p>
        Most examples are cut from the reads of <DocLink to="first-alignment">Your first alignment</DocLink> (for
        example, the last 150 nt of read06), and all are aligned with the IGH model.
      </p>

      <H2 id="signals">What each row tells you about itself</H2>
      <DocTable
        head={["You see", "It means"]}
        rows={[
          [
            <>
              Several alleles in <code>v_call</code>
            </>,
            "The read matches them all equally well: it cannot tell them apart.",
          ],
          [
            <>
              Several alleles in a <code>*_call_set</code>
            </>,
            "The model could not settle on one allele for that segment.",
          ],
          [
            <>
              <code>airr_assembly_status</code> <code>partial</code> or <code>failed</code>
            </>,
            <>
              Some fields are blank; <code>airr_assembly_reason</code> says why (
              <DocLink to="trust-the-calls?h=partial-rows">partial rows</DocLink>).
            </>,
          ],
          [
            <>
              <code>segmentation_low_quality</code> <code>T</code>
            </>,
            "The read's V or J region came out empty: that segment's call means nothing.",
          ],
          [
            <>
              <code>length_cropped</code> <code>T</code>
            </>,
            "The read was cut to its first 575 nt.",
          ],
          [
            <>
              A blank <code>productive</code> or <code>junction</code>
            </>,
            "The read does not hold both conserved codons: productivity is unknown.",
          ],
          [
            <>
              A low <code>v_identity</code>, a high <code>mutation_rate</code>
            </>,
            "Heavy hypermutation, sequencing errors, an allele the germline set lacks, or a read of another locus.",
          ],
        ]}
      />

      <H2 id="short-reads">Short reads and fragments</H2>
      <p>
        A read that covers only part of its V holds fewer bases that tell alleles apart, so <code>v_call</code> lists
        more alleles. The junction is not affected. read06, whole and cut to its last 150 and last 100 nt:
      </p>
      <CodeBlock
        lang="text"
        title="out.tsv (v_call summarized)"
        code={`sequence_id     v_call                                                 junction_aa
read06          IGHVF3-G9*03                                           CARLYRNITGTRFYWYFNLW
read06_last150  IGHVF3-G9*01,IGHVF3-G9*02,IGHVF3-G9*03,IGHVF3-G9*04,   CARLYRNITGTRFYWYFNLW
                IGHVF3-G9*04_A319G
read06_last100  IGHVF3-G6*03,... (68 alleles of families F3, F5, F6    CARLYRNITGTRFYWYFNLW
                and F10, IGHVF3-G9*03 among them)`}
      />
      <p>
        On the 150-nt fragment, the five listed alleles share one gene, so the gene call is safe. On the 100-nt
        fragment the list crosses families: the first name, <code>IGHVF3-G6*03</code>, is no more likely than the 67
        after it. Shorter still, and the read has no V at all for AlignAIR:
      </p>
      <ul>
        <li>
          <strong>No V call.</strong> A read gets no V call when it shares no 15-nt stretch with any V allele of the
          germline set and its V alignment covers fewer than 20 bases or matches under 70% of them. Its row is{" "}
          <code>partial</code> with reason <code>no_v</code>; its D and J calls stand. The last 100 nt of read01 end
          this way.
        </li>
        <li>
          <strong>A read that ends before its J.</strong> Its J region comes out empty: the row has{" "}
          <code>segmentation_low_quality</code> <code>T</code>, is <code>partial</code> with reason{" "}
          <code>collapsed_segment</code>, and still carries a <code>j_call</code>, which means nothing. The first 200 nt
          of read01 get <code>IGHJ1*01</code>; the whole read is <code>IGHJ6*02</code>.
        </li>
        <li>
          <strong>No junction</strong> unless both conserved codons, the V's Cys and the J's Trp or Phe, lie wholly
          inside the read.
        </li>
      </ul>
      <p>
        <strong>What to do.</strong> Report fragments at the gene level when the listed alleles share a gene, and keep
        the whole list for TIgGER. Leave out rows with <code>segmentation_low_quality</code> <code>T</code> when you
        count gene usage, and use complete rows for junction and clone work (
        <DocLink to="output-columns?h=filter-usable-rows">filters</DocLink>).
      </p>

      <H2 id="hypermutation">Heavy hypermutation</H2>
      <p>
        AlignAIR calls the V allele of the germline set that matches the most of the read's V bases. Somatic
        mutations can make a read match a sibling allele, one or a few bases away from the true one, as well or
        better: the sibling is then listed after the call, or becomes the call. The more mutations and the closer the
        siblings, the more often this happens. IgBLAST faces the same limit; the comparison measures both tools on it,
        as the mutation load rises (the "V allele siblings" suite).
      </p>
      <p>
        <strong>How it shows.</strong> A low <code>v_identity</code>, a high <code>mutation_rate</code>, and longer
        lists in <code>v_call</code>.
      </p>
      <p>
        <strong>What to do.</strong> Call from the donor's own alleles: a sibling the donor does not carry can then not
        be called (<code>--genotype</code>, or <code>--infer-genotype</code> to infer the genotype from the same reads;
        see <DocLink to="donor-genotype">Align with a donor genotype</DocLink>). Make allele-level claims on heavily
        mutated reads only after a <DocLink to="trust-the-calls?h=cross-check">cross-check</DocLink>; gene-level
        analyses are much less affected.
      </p>

      <H2 id="long-reads">Reads longer than 575 nt</H2>
      <p>
        A model reads at most 575 nt. A longer read is aligned on its first 575 nt, as written in the file, and its{" "}
        <code>length_cropped</code> is <code>T</code>. The cut comes before AlignAIR turns the read around: a forward
        read with a long stretch before its V loses its J and junction; a reverse-complemented read loses the start of
        its V. read01 with 300 extra bases in front of it is cut near the end of its V, before the junction: it gets no
        junction and no meaningful J call, and the row is <code>partial</code> (<code>collapsed_segment</code>).
      </p>
      <p>
        <strong>What to do.</strong> Trim reads to the V(D)J region before aligning (
        <DocLink to="prepare-input?h=long-reads">Prepare your input</DocLink>), and check the rows with{" "}
        <code>length_cropped</code> <code>T</code>. Cell Ranger's contigs are forward and carry the constant region
        after the J, so many are cut, mostly in the constant region, which changes no call: Prepare your input shows
        how to find the few where the cut reached the J.
      </p>

      <H2 id="outside-the-read">Alleles that differ only outside the read</H2>
      <p>
        Most reads show only part of their V: a primer inside the V cuts its start, and recombination trims its end.
        Alleles that differ only in the part a read lacks match it equally well, and <code>v_call</code> lists them all
        after the call; the first name is not more likely than the others. Alleles of different genes, even of
        different families, can be identical over a short stretch, as read06's last 100 nt show above. Alleles that
        differ only past IMGT position 318, at the very end of the V, can rarely be told apart in any rearranged read.
      </p>
      <p>
        D and J calls name one allele, with no such list. Two pairs of IGH D alleles have identical sequences (
        <code>IGHD4-4*01</code> and <code>IGHD4-11*01</code>, <code>IGHD5-5*01</code> and <code>IGHD5-18*01</code>):{" "}
        <code>d_call</code> names one of each pair, so count each pair as one. <code>d_call_set</code> and{" "}
        <code>j_call_set</code> show the other alleles the model weighed.
      </p>
      <p>
        <strong>What to do.</strong> Collapse a list to a gene only when every allele in it has that gene. A donor
        genotype settles most lists: <code>--infer-genotype</code> re-calls the reads whose call falls outside the donor's
        genotype while their list holds an allele of it (<DocLink to="donor-genotype?h=infer-genotype">infer the
        genotype</DocLink>).
      </p>

      <H2 id="short-d">D calls on short D segments</H2>
      <p>
        The D is the least certain call. D alleles are short (11 to 37 nt in the IGH set; 12 and 16 nt in TRB; 8, 9 and
        13 nt in TRD), recombination trims both their ends, and the N nucleotides around them can look like D bases.
      </p>
      <ul>
        <li>
          AlignAIR calls a D only when it finds at least 8 of the D's germline bases in the read. With fewer,{" "}
          <code>d_call</code> is blank, the D's place in the read (<code>d_sequence_start</code>, <code>d_sequence_end</code>) is kept and{" "}
          <code>d_identity</code> is blank. Either way, a row without a D call has no D in its germline:{" "}
          <code>np1</code> runs from the V to the J, as Change-O reads such a row. With no D bases at all, the row has no
          D coordinates either.
        </li>
        <li>
          So a TRD D is called only when almost all of it remains: <code>TRDD1*01</code> is 8 nt long.
        </li>
        <li>
          <code>d_call</code> names one allele; <code>d_call_set</code> lists the others the model weighed.
        </li>
      </ul>
      <p>
        <strong>What to do.</strong> Report D usage at the gene level, count the identical pairs above as one, and read
        a blank <code>d_call</code> as "too little D to call", not "no D". For a study that rests on D alleles,
        cross-check with IgBLAST. The comparison measures D calls as the D is trimmed shorter ("D observability").
      </p>

      <H2 id="novel-alleles">Novel alleles</H2>
      <p>
        AlignAIR calls only alleles of the germline set it runs with. A read of an allele the set lacks is called to the
        allele that matches it at the most bases, and its differences count as mutations: they lower{" "}
        <code>v_identity</code> and raise <code>mutation_rate</code>. Nothing in the row marks the allele as novel.
      </p>
      <ul>
        <li>
          AlignAIR does not discover novel alleles, and neither does <code>--infer-genotype</code>. Find them with a
          tool built for it, such as TIgGER, on AlignAIR's table (
          <DocLink to="from-immcantation?h=tigger">TIgGER on AlignAIR's table</DocLink>).
        </li>
        <li>
          Then add them to the germline set with <code>--reference</code>: a read that matches the novel allele at
          more bases than the old call is called by the novel allele's name (
          <DocLink to="own-reference?h=novel-alleles">Novel alleles</DocLink>). Export the model's set with{" "}
          <code>--gapped</code>, append the novel alleles IMGT-gapped (TIgGER writes them so), and give that one file
          to AlignAIR, TIgGER and CreateGermlines. Read the warning that names any allele left without its anchor (
          <DocLink to="own-reference?h=gaps-and-anchors">gaps and anchors</DocLink>).
        </li>
      </ul>

      <H2 id="off-target">Reads of another locus, and other sequences</H2>
      <p>
        A model calls only its own locus, and every row carries that locus in <code>locus</code>. A read of another
        locus, or a sequence that is not a receptor at all, most often gets no V call: the row is <code>partial</code>{" "}
        with reason <code>no_v</code>, and still has a <code>j_call</code>, which means nothing. Most such rows have{" "}
        <code>segmentation_low_quality</code> <code>F</code>, so that column does not find them. And a fifth to a third
        of these reads come out as ordinary <code>complete</code> rows, with a V call of the model's locus, often a
        junction, sometimes <code>productive</code> <code>T</code>, and the run gives no warning. Simulated 10x-like
        contigs aligned with the model of another locus:
      </p>
      <DocTable
        head={["Reads, model", <><code>no_v</code> rows (with <code>segmentation_low_quality</code> T)</>, <><code>complete</code> rows</>, <>of them <code>productive</code> T</>]}
        rows={[
          ["300 TRA contigs, TRB model", "206 (23)", "85 (28%)", "21"],
          ["300 TRB contigs, IGH model", "246 (92)", "54 (18%)", "2"],
          ["300 TRB contigs, TRA model", "188 (4)", "98 (33%)", "13"],
        ]}
      />
      <CodeBlock
        lang="text"
        title="TRA contigs aligned with the TRB model (a few columns)"
        code={`sequence_id                  locus  v_call    j_call      v_identity          junction_aa       productive  airr_assembly_status  airr_assembly_reason
TAAATGAGACATCTTA-1_contig_1  TRB    TRBV2*01  TRBJ1-6*01  0.8292682926829268  CAMREGINTNAGKSTF  T           complete
CAGATTTTCATATTAT-1_contig_1  TRB              TRBJ2-4*01                                                    partial               no_v`}
      />
      <p>
        <strong>How it shows.</strong> Two signals tell these rows apart. With 10x data, Cell Ranger's{" "}
        <code>chain</code> (joined with <code>--metadata</code>) differs from <code>locus</code>. On the TCR models,{" "}
        <code>v_identity</code> is low: the off-locus rows with a V call had a median of 0.75 to 0.79, every true TRA
        and TRB contig 0.98 or more. A few off-locus reads keep only a short stretch of V that matches many alleles
        exactly, and reach a <code>v_identity</code> of 1: the <code>chain</code> column still finds those.
      </p>
      <p>
        <strong>What to do.</strong> Split mixed libraries by locus and align each part with its own model (
        <DocLink to="choose-a-model?h=pick-by-locus">Pick by locus</DocLink>). Keep the rows whose{" "}
        <code>chain</code> equals <code>locus</code>, leave out TCR rows with a low <code>v_identity</code>, and count
        J only on rows with a V call (<DocLink to="output-columns?h=filter-usable-rows">filters</DocLink>). Many{" "}
        <code>no_v</code> rows in a run point to reads of another locus, or to a genotype that misses genes.
      </p>

      <H2 id="other-limits">Other limits</H2>
      <ul>
        <li>
          <strong>Frameshifts inside V or J.</strong> <code>productive</code> is read from the junction: its frame and
          the stop codons in its frame. An indel inside the V or the J that shifts the frame is not, by itself, called
          unproductive.
        </li>
        <li>
          <strong>A genotype that misses alleles</strong> forces the reads of the missing alleles onto the nearest
          allele it allows, with nothing in the row to say so (
          <DocLink to="donor-genotype?h=outside-the-genotype">reads of genes the genotype leaves out</DocLink>).
        </li>
        <li>
          <strong>A germline set of another species</strong> binds to a human model when it has the model's segments,
          but how well the model calls it has not been measured (
          <DocLink to="own-reference?h=reference-refusals">references that are refused</DocLink>). Train a model for
          that species.
        </li>
        <li>
          <strong>Models from <code>alignair demo</code> or <code>--preset quick</code></strong> are for trying the
          steps, not for results.
        </li>
      </ul>

      <H2 id="partial-rows">Partial rows and segmentation_low_quality</H2>
      <p>
        A <code>partial</code> row keeps its calls but has some fields blank. <code>airr_assembly_reason</code> says
        why:
      </p>
      <DocTable
        head={["Reason", "What happened", "What the row keeps"]}
        rows={[
          [
            <code>no_v</code>,
            <>
              The read holds no V AlignAIR could place (<DocLink to="trust-the-calls?h=short-reads">short reads</DocLink>
              ): a read that starts inside the junction, a D-J rearrangement, a read of another locus, or a read of a
              gene the <code>--genotype</code> leaves out.
            </>,
            "The D and J calls. No junction, alignment or regions.",
          ],
          [
            <code>nonproductive_indel</code>,
            "The junction is out of frame or holds a stop codon, and AlignAIR estimates more than one insertion or deletion in the read: it does not build the read's alignment.",
            "The calls and the junction.",
          ],
          [
            <code>missing_calls_or_coordinates</code>,
            "The V call, or the V or J position in the read, is missing.",
            "What was called.",
          ],
          [
            <code>collapsed_segment</code>,
            <>
              The read's V or J region came out empty (<code>segmentation_low_quality</code> <code>T</code>): a read that
              ends before its J, or a long read cut short.
            </>,
            "The calls; the one for the empty segment means nothing.",
          ],
          [
            <code>incomplete_alignment</code>,
            "The IMGT-gapped alignment could not be built.",
            "The calls, positions and junction.",
          ],
        ]}
      />
      <p>
        A <code>failed</code> row met an unexpected error while its fields were built: <code>airr_assembly_error</code>{" "}
        holds the error, and the row keeps its calls and positions. Report one that repeats, with the read and{" "}
        <code>--debug</code> (<DocLink to="support">Support</DocLink>).
      </p>
      <p>
        <code>segmentation_low_quality</code> is <code>T</code> on every row whose V or J region came out empty. Such a
        row is never <code>complete</code>: its reason is <code>collapsed_segment</code>, or one found before it, such
        as <code>no_v</code>.
      </p>

      <H2 id="quality-gates">The quality gates</H2>
      <p>
        A run stops with an error when too many of its rows could not be assembled. The table is written either way,
        every row tagged; only the exit status (1) and the message report the breach.
      </p>
      <DocTable
        head={["Option", "Default", "The run fails when"]}
        rows={[
          [<code>--max-assembly-failures</code>, <code>0.01</code>, "more than this share of the rows failed"],
          [<code>--max-partial-assemblies</code>, <><code>1.0</code> (never)</>, "more than this share of the rows is partial"],
          [<code>--permissive</code>, "off", "never: the run always succeeds, rows still tagged"],
        ]}
      />
      <CodeBlock
        lang="text"
        title="stderr"
        code={`[ALIGNAIR_INPUT_INVALID] AIRR assembly FAILED for 3/200 reads (1.5% > --max-assembly-failures 1.0%); reasons {...}. Wrote tagged output to out.tsv.`}
      />
      <p>
        A pipeline that should stop when the reads look wrong, say a library of another locus, can fail on partial rows
        too; a run on dirty reads can turn the gates off:
      </p>
      <CodeBlock
        code={`alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv --max-partial-assemblies 0.2
alignair predict --model alignair4-igh-human --input dirty.fasta --out dirty.tsv --permissive`}
      />

      <H2 id="cross-check">When to cross-check with IgBLAST</H2>
      <p>Check a sample of the reads against IgBLAST when:</p>
      <ul>
        <li>a claim rests on V alleles of heavily mutated reads (lineages, mutation studies);</li>
        <li>a claim rests on D alleles;</li>
        <li>
          the reads differ from what the models were trained on: very long reads, unusual library designs, or a germline
          set of another species bound with <code>--reference</code>;
        </li>
        <li>
          the <DocLink to="comparison">comparison</DocLink> shows IgBLAST ahead on a measure your analysis rests on.
        </li>
      </ul>
      <p>
        Give IgBLAST the germline set the model calls from, so both tools choose among the same alleles (
        <DocLink to="from-igblast?h=fair-comparison">A fair side-by-side</DocLink>), then see where they agree:
      </p>
      <CodeBlock code={`alignair compare --a alignair.tsv --b igblast.tsv --a-name AlignAIR --b-name IgBLAST --out report.md`} />
      <p>
        Agreement is not accuracy: on real reads neither tool is the truth. Look at the reads where they differ, and at
        whether IgBLAST's call is in AlignAIR's candidate set (the report's set rescue), which means shared doubt rather
        than a conflict.
      </p>
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Comparison with IgBLAST                                                                                            */
/* ------------------------------------------------------------------------------------------------------------------ */
const SUITES: [string, string, string][] = [
  ["S1", "Core repertoire", "Realistic donor repertoires: full-length reads and FR1, FR2 and FR3 primer designs, at three mutation levels."],
  ["S2", "V allele siblings", "Telling a V allele from a sibling one to five bases away, as the mutation load rises."],
  ["S3", "Junction sweeps", "The junction and the segment edges, as N lengths and V, D and J trims are changed one at a time."],
  ["S4", "Read truncation", "Calls, germline edges and abstaining, as reads lose a fixed number of bases at one end."],
  ["S5", "V indels", "Finding insertions and deletions in V, the V edge and the junction, as indels in V rise from one to six."],
  ["S6", "D observability", "D calls and D edges, as the D is trimmed shorter or inverted."],
  ["S7", "Novel alleles", "Reads of donors who carry V alleles the germline set lacks: calls, genes and mutation counts."],
  ["S8", "Donor genotypes", "How often a tool calls alleles the donor does not carry, alleles of deleted genes included."],
  ["S9", "Clonal lineages", "Whether every member of a clone gets the same V gene, J gene and junction length as the members diverge."],
  ["S10", "Held-out settings", "Accuracy when the mutation model, the N lengths or the trims leave the default distributions."],
  ["S11", "Orientation pairs", "Finding a read's orientation, and annotating both strands of a read the same way."],
  ["S12", "Sequencing noise", "Accuracy under sequencing errors, PCR errors, ambiguous bases and polymerase indels."],
  ["S13", "Library flanks", "Where V starts and J ends when the read carries a 5' leader, a masked primer or a 3' constant region."],
];

const comparison: DocPage = {
  slug: "comparison",
  title: "Comparison with IgBLAST",
  section: SECTION,
  lead: "How AlignAIR's calls compare with IgBLAST's on all seven loci, every measure shown, ties included; how that was measured, on simulated reads and on real IGH repertoires; and when IgBLAST is still the better choice.",
  body: () => (
    <>
      <p>
        Both tools align the same reads, against the same germline set, and are scored by the same rules. The
        measurement is AlignAIR's assay, <code>alignair_assay</code>: it lives in the{" "}
        <Ext href={ASSAY}>source repository</Ext>, not in the pip package, and its README gives the full method. This
        page says what it measures and how to read the results.
      </p>

      <H2 id="method">How it was measured</H2>

      <H3 id="same-germline-set">The same germline set</H3>
      <p>
        IgBLAST's V, D and J databases are built from the model's own germline set, with an auxiliary file of J reading
        frames written from the same germlines, and IgBLAST runs with <code>-organism human</code>,{" "}
        <code>-ig_seqtype Ig</code> or <code>TCR</code>, and AIRR output (<code>-outfmt 19</code>). Both tools so choose
        among the same alleles. For a locus without D, IgBLAST gets a one-sequence placeholder D database, which it
        needs and does not search. The IgBLAST release is recorded, with checksums, alongside the results.
      </p>

      <H3 id="known-truth">Simulated reads with known truth</H3>
      <p>
        GenAIRR simulates each read from known alleles, trims, N nucleotides, mutations and sequencing artefacts, so the
        truth of every read is known. Each read is scored on what it shows:
      </p>
      <ul>
        <li>
          a segment counts only when the read holds some of its germline bases;
        </li>
        <li>
          the right answer for an allele is the set of alleles a perfect aligner could defend from that read: the true
          allele and those that match the read as well over the bases it shows;
        </li>
        <li>a segment edge is right anywhere in its range of equally defensible positions;</li>
        <li>
          a missing call counts as wrong, except on a segment too short to call (fewer than 20 V, 8 D or 10 J germline
          bases), where abstaining counts as right.
        </li>
      </ul>
      <p>
        The reads come from a locked test split: generated from seeds never used while the models were developed, and
        scored once. Results from development runs are not shown. The models were trained on GenAIRR simulations too,
        so two parts of the comparison step outside them: a suite whose mutation model, N lengths and trims leave the
        defaults (S10), and the real repertoires <DocLink to="comparison?h=real-data">below</DocLink>.
      </p>

      <H3 id="suites">The 13 suites</H3>
      <DocTable head={["Suite", "Name", "What it measures"]} rows={SUITES.map(([id, title, what]) => [id, title, what])} />
      <p>
        IG reads carry somatic hypermutation; TCR reads do not, as T cells do not hypermutate. A suite or part of a
        suite that needs a D or hypermutation is left out on a locus without it: D observability runs on IGH, TRB and
        TRD, clonal lineages on the IG loci.
      </p>

      <H3 id="measures">The measures</H3>
      <p>
        A measure's name is the segment, then what is scored. The tables show every measure the run compared; the main
        ones:
      </p>
      <DocTable
        head={["Measure", "Scores"]}
        rows={[
          [
            <>
              <code>v.top1_in_set</code>, <code>d.top1_in_set</code>, <code>j.top1_in_set</code>
            </>,
            <>The called allele (the first one listed) is one the read supports.</>,
          ],
          [<code>v.gene_top1</code>, "The called gene is right."],
          [
            <>
              <code>v.seq_start_exact</code>, <code>v.seq_end_exact</code>, <code>j.seq_start_exact</code>, ...
            </>,
            "A segment's start or end in the read is exact. germ_start and germ_end do the same in the germline.",
          ],
          [
            <>
              <code>junction.nt_exact</code>, <code>junction.aa_exact</code>, <code>junction.length_exact</code>
            </>,
            "The junction is exactly right, in nucleotides, in amino acids, in length.",
          ],
          [
            <code>clone_key.exact</code>,
            "The V gene, the J gene and the junction length are all right: what clonal clustering groups on.",
          ],
          [
            <>
              <code>productive.correct</code>, <code>vj_in_frame.correct</code>, <code>stop_codon.correct</code>
            </>,
            "The productivity calls are right.",
          ],
          [<code>orientation.correct</code>, "The read's orientation is right."],
          [
            <>
              <code>v.no_call</code>, <code>d.spurious_call</code>, <code>v_short.abstain_ok</code>
            </>,
            "A missing call; a call for a segment the read does not hold; abstaining on a segment too short to call.",
          ],
          [
            <>
              Names ending in <code>_err</code>, such as <code>v.mutation_count_err</code>
            </>,
            "A mean distance from the truth, in nucleotides or mutations: lower is better.",
          ],
        ]}
      />
      <p>
        For the measures where lower is better (the <code>_err</code> measures, <code>no_call</code>,{" "}
        <code>spurious_call</code>, <code>indel_false</code>, <code>germline_errors</code>,{" "}
        <code>outside_genotype</code> and <code>deleted_gene_call</code>), a negative difference favours AlignAIR; the verdict already takes the direction
        into account.
      </p>

      <H3 id="paired-tests">Paired tests</H3>
      <ul>
        <li>
          Both tools are scored on the same reads. The difference is AlignAIR's score minus IgBLAST's, over the reads
          both were scored on, with its 95% confidence interval. The IgBLAST and AlignAIR columns are each tool's mean
          over the reads the measure applies to.
        </li>
        <li>
          A right-or-wrong measure is tested with an exact McNemar test, on the reads where only one tool is right; a
          measure in nucleotides or counts, with a paired z-test. The reads of one clone (S9) and the two strands of
          one read (S11) are not independent, so they are tested by cluster.
        </li>
        <li>
          The p-values of all the measures of one suite are corrected together (Holm, at 5%). The verdict is "AlignAIR
          better" or "IgBLAST better" when a difference survives the correction, and "tie" otherwise, which includes
          a measure with fewer than 10 paired reads.
        </li>
      </ul>

      <H2 id="results">Results on simulated reads</H2>
      {/* PENDING-V4-NUMBERS: the headline results from the locked test-split run go here, one line per measure with its
          numbers, before the tables (owner: headline wins up front, the full tables with ties below). */}
      <p>One table per suite, every measure, ties included.</p>
      <H3 id="results-igh">IGH</H3>
      <ComparisonTable locus="IGH" />
      <H3 id="results-igk">IGK</H3>
      <ComparisonTable locus="IGK" />
      <H3 id="results-igl">IGL</H3>
      <ComparisonTable locus="IGL" />
      <H3 id="results-tra">TRA</H3>
      <ComparisonTable locus="TRA" />
      <H3 id="results-trb">TRB</H3>
      <ComparisonTable locus="TRB" />
      <H3 id="results-trd">TRD</H3>
      <ComparisonTable locus="TRD" />
      <H3 id="results-trg">TRG</H3>
      <ComparisonTable locus="TRG" />

      <H2 id="real-data">Real IGH repertoires</H2>
      <p>
        Simulation is not the whole story, so the comparison also scores both tools on real human IGH repertoires where
        each donor's own alleles are known:
      </p>
      <ul>
        <li>
          <strong>The data.</strong> AIRR-seq of donors from BioProject <Ext href={PRJNA555323}>PRJNA555323</Ext> (
          <Ext href={RODRIGUEZ}>Rodriguez et al., Nature Communications 2023</Ext>), whose IGH loci were also
          sequenced with long reads, so each donor's alleles are known from the genome (VDJbase). Four donors are held
          back for this run. The reads are prepared with pRESTO, as the study did, and both tools align them against
          the IGH model's germline set.
        </li>
        <li>
          <strong>The measure, declared before the run.</strong> The share of each tool's V calls that the donor's
          genome does not carry (lower is better). A call counts as carried when it equals one of the donor's alleles
          up to IMGT position 318. Reads called into genes where the genomic truth is incomplete are left out for
          both tools. The tools are compared on the same reads, per library protocol (5' RACE and multiplex PCR), with
          tests clustered by clonotype.
        </li>
        <li>
          <strong>Beside it.</strong> On the reads whose V matches exactly one of the donor's alleles, how often each
          tool calls that allele.
        </li>
      </ul>
      <p>
        The other loci have no such check on real reads: for IGK, IGL and the four TCR loci, the simulated results
        above are the measure.
      </p>
      {/* PENDING-V4-NUMBERS: the locked run's real-data results (its pre-declared measure per protocol, with the
          known-answer accuracy beside it), reported as the run's report gives them, wins, ties or losses alike. */}
      <p className="text-sm text-slate-500">Results of the locked test run appear here.</p>

      <H2 id="your-own-data">On your own data</H2>
      <p>
        Your reads have no known truth, so a comparison there measures agreement, not accuracy. To see where the two
        tools agree on your reads, run both on the same germline set and compare their tables (
        <DocLink to="from-igblast?h=fair-comparison">A fair side-by-side</DocLink>). To measure a model you trained,
        see <DocLink to="evaluate">Evaluate your model</DocLink>.
      </p>

      <H2 id="when-igblast">When IgBLAST is still the better choice</H2>
      <ul>
        <li>
          <strong>A species or locus no AlignAIR model covers,</strong> when you cannot train one (
          <DocLink to="do-i-need-to-train">Do I need to train?</DocLink>): IgBLAST aligns against any germline database
          you build.
        </li>
        <li>
          <strong>Reads longer than 575 nt</strong> that you cannot trim to the V(D)J region: AlignAIR aligns only their
          first 575 nt.
        </li>
        <li>
          <strong>A D call wherever any D bases remain:</strong> AlignAIR leaves a D uncalled when it finds fewer than 8
          of its bases.
        </li>
        <li>
          <strong>IgBLAST's own output:</strong> its alignment scores and e-values, or a pipeline or reviewer that asks
          for IgBLAST specifically.
        </li>
        <li>
          <strong>A measure your study rests on</strong> where the tables above say "IgBLAST better".
        </li>
      </ul>
      <p>
        The limits behind these are on <DocLink to="trust-the-calls">How far to trust the calls</DocLink>.
      </p>
    </>
  ),
};

export const resultsPages: DocPage[] = [outputColumns, trustTheCalls, comparison];
