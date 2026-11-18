import { CodeBlock, Callout } from "../../components/ui";
import CodeTabs from "../../components/CodeTabs";
import { DocTable, DocLink, H2, H3, type DocPage } from "./doc-kit";
import ColumnTable from "../../components/data/ColumnTable";

const PREDICT_PYTHON = `from alignair import Aligner, read_sequences

aligner = Aligner.from_pretrained("alignair4-igh-human")
ids, reads, info = read_sequences("reads.fasta")
aligner.predict(reads, ids=ids).write_airr("alignair.tsv")`;

const TIGGER_RUN = `library(tigger)

db <- airr::read_rearrangement("alignair.tsv")
germline_db <- readIgFasta("germlines.fasta")      # alignair reference export ... --gapped

# pos_range leaves out the V positions the primers cover (these reads start near IMGT position 40)
novel <- findNovelAlleles(db, germline_db, pos_range = 50:312, nproc = 4)
genotype <- inferGenotype(db, germline_db = germline_db, novel = novel)`;

const TIGGER_BACK = `# The novel alleles, gapped, after the gapped germlines: one file for AlignAIR, TIgGER and CreateGermlines
found <- selectNovel(novel)
file.copy("germlines.fasta", "germlines_with_novel.fasta", overwrite = TRUE)
cat(sprintf(">%s\\n%s\\n", found$polymorphism_call, found$novel_imgt),
    file = "germlines_with_novel.fasta", sep = "", append = TRUE)

# The genotype as YAML: every V allele of it, gene*allele
alleles <- unlist(Map(function(gene, calls) paste0(gene, "*", strsplit(calls, ",")[[1]]),
                      genotype$gene, genotype$alleles), use.names = FALSE)
writeLines(c("V:", paste0("  - ", alleles)), "donor.yaml")`;

/* ------------------------------------------------------------------------------------------------------------------ */
/* IgBLAST                                                                                                         */
/* ------------------------------------------------------------------------------------------------------------------ */

const fromIgblast: DocPage = {
  slug: "from-igblast",
  title: "Coming from IgBLAST",
  section: "Coming from…",
  lead: "Swap your igblastn command for alignair predict, keep reading the same AIRR table, and learn the few columns that differ.",
  body: () => (
    <>
      <p>
        Both tools write an AIRR rearrangement table, so Change-O, TIgGER, SCOPer and your own scripts read AlignAIR's
        output as they read IgBLAST's <code>-outfmt 19</code>. Three things change: there is no germline database to
        build, the IG models name alleles the OGRDB way, and a few columns differ.
      </p>

      <H2 id="the-command">The command</H2>
      <p>Your IgBLAST command for human IGH reads looks like this:</p>
      <CodeBlock
        title="IgBLAST"
        code={`igblastn -germline_db_V human_V -germline_db_D human_D -germline_db_J human_J \\
  -auxiliary_data optional_file/human_gl.aux -organism human -ig_seqtype Ig \\
  -outfmt 19 -query reads.fasta -out igblast.tsv`}
      />
      <p>The same reads through AlignAIR:</p>
      <CodeTabs
        cli={`$ alignair predict --model alignair4-igh-human --input reads.fasta --out alignair.tsv
aligned 20 reads (0 dropped) -> alignair.tsv`}
        python={PREDICT_PYTHON}
      />
      <p>
        The output line is from the 20 example reads of <DocLink to="first-alignment">Your first alignment</DocLink>.
        The model carries its germline set, so there is nothing to build. The first run downloads the model and caches
        it. Beside the table, <code>alignair.tsv.run.json</code> records the model, the reference and the options of
        the run.
      </p>

      <H2 id="option-by-option">IgBLAST's options, one by one</H2>
      <DocTable
        head={["IgBLAST", "AlignAIR"]}
        rows={[
          [
            <><code>-germline_db_V</code>, <code>-germline_db_D</code>, <code>-germline_db_J</code></>,
            <>
              The model's own germline set, chosen with <code>--model</code>. For another set, novel alleles included,
              bind a FASTA with <code>--reference</code> (<DocLink to="from-igblast?h=bind-a-reference">below</DocLink>).
            </>,
          ],
          [
            <code>-auxiliary_data</code>,
            <>
              Nothing to give. The model's J alleles carry their reading frame and CDR3 end. A bound{" "}
              <code>--reference</code> takes a J's from the model's allele with the same sequence, else from the J's own
              Trp/Phe-Gly-X-Gly motif.
            </>,
          ],
          [
            <><code>-organism human</code>, <code>-ig_seqtype Ig</code> or <code>TCR</code></>,
            <>
              The model: one per locus, all human (<DocLink to="choose-a-model">Choose a model</DocLink>). For another
              species, see <DocLink to="do-i-need-to-train">Do I need to train?</DocLink>
            </>,
          ],
          [
            <code>-domain_system imgt</code>,
            <>
              Always IMGT: <code>sequence_alignment</code> and <code>germline_alignment</code> are IMGT-gapped, and the
              FWR and CDR boundaries are IMGT's.
            </>,
          ],
          [
            <code>-query reads.fasta</code>,
            <>
              <code>--input reads.fasta</code>: FASTA, FASTQ, CSV or TSV, gzipped or not, or <code>-</code> for standard
              input.
            </>,
          ],
          [
            <><code>-outfmt 19</code>, <code>-out</code></>,
            <>
              The default: an AIRR rearrangement TSV with 99 columns, written to <code>--out</code>.{" "}
              <code>--columns airr</code>, <code>core</code>, <code>minimal</code> or a comma-separated list write fewer.
            </>,
          ],
          [
            <code>-num_threads N</code>,
            <>
              <code>--threads N</code>: N worker processes on the CPU. A GPU is used when there is one (
              <code>--device auto</code>, the default).
            </>,
          ],
        ]}
      />

      <H2 id="germline-database">Your germline database</H2>
      <p>
        IgBLAST searches the databases you build. An AlignAIR model calls from the germline set it was trained with:
        OGRDB sets for the human IG models, IMGT sets for the TCR models.{" "}
        <DocLink to="choose-a-model">Choose a model</DocLink> lists each model's set and allele counts.
      </p>

      <H3 id="allele-names">Allele names on the IG models</H3>
      <p>
        The IG models call alleles by their OGRDB labels. IGH V alleles read <code>IGHVF10-G50*02</code> (F for the
        family, G for the gene group), where IgBLAST with IMGT germlines writes names such as <code>IGHV3-23*01</code>.
        IGH D and J alleles keep IMGT-style names (<code>IGHD3-10*01</code>, <code>IGHJ6*02</code>). IGK and IGL alleles
        have OGRDB labels of their own, such as <code>IGKV1-5TCS*03</code> and <code>IGLJ2/3-HDWH*01</code>. The TCR
        models use IMGT names.
      </p>
      <p>
        So a join on <code>v_call</code> with an IgBLAST table, a gene-usage plot, or a genotype file in IMGT names will
        not match the IG models' calls. Two ways through:
      </p>
      <ul>
        <li>
          Bind IMGT's germline set with <code>--reference</code>: the calls then carry IMGT names (
          <DocLink to="choose-a-model?h=bind-imgt">Call with IMGT's set</DocLink>).
        </li>
        <li>
          Map the model's labels to IMGT names by their sequences, and translate your tables or AlignAIR's calls (
          <DocLink to="choose-a-model?h=map-names">Map the labels to IMGT names</DocLink>).
        </li>
      </ul>

      <H3 id="bind-a-reference">Bind another germline set</H3>
      <p>
        Give a FASTA of V, D and J alleles whose headers are the allele names (<code>&gt;IGHV3-23*01</code>), or
        IMGT/GENE-DB headers, which are read as they are (the field with a <code>*</code> names the allele). Novel
        alleles are welcome. Missing IMGT gaps and anchors are taken from the model's own alleles where it can. Nothing
        is retrained.
      </p>
      <CodeTabs
        cli={`alignair predict --model alignair4-igh-human --input reads.fasta --out alignair.tsv \\
  --reference germline.fasta`}
        python={`from alignair import Aligner, ReferenceSet, read_sequences

reference = ReferenceSet.from_fasta("germline.fasta")
aligner = Aligner.from_pretrained("alignair4-igh-human", reference=reference)
ids, reads, info = read_sequences("reads.fasta")
aligner.predict(reads, ids=ids).write_airr("alignair.tsv")`}
      />
      <p>
        The FASTA must hold every segment the model calls (the IGH model refuses one without D alleles), and keep to the
        model's locus: one that mixes in alleles of another locus is refused. The details are on{" "}
        <DocLink to="own-reference">Use your own germline reference</DocLink>.
      </p>

      <H2 id="columns">Column by column</H2>
      <p>The shared AIRR columns keep their meaning. These are the ones to know:</p>
      <DocTable
        head={["Columns", "What to know"]}
        rows={[
          [
            <>
              <code>sequence_id</code>, <code>sequence</code>, <code>locus</code>, <code>rev_comp</code>,{" "}
              <code>productive</code>, <code>vj_in_frame</code>, <code>stop_codon</code>, <code>junction</code>,{" "}
              <code>junction_aa</code>, <code>junction_length</code>, <code>np1</code>, <code>np2</code>
            </>,
            <>
              The same in both. The T/F columns are the strings <code>T</code> and <code>F</code>: in pandas, compare
              with <code>== "T"</code>.
            </>,
          ],
          [
            <><code>v_sequence_start</code> to <code>j_germline_end</code></>,
            <>
              1-based positions in the read and in the germline allele. AlignAIR's <code>v_germline_start</code> and{" "}
              <code>v_germline_end</code> count in the IMGT-gapped V germline from its first position, as Change-O's
              MakeDb writes them (so <code>v_germline_start</code> is 1); IgBLAST's count in the ungapped allele.
            </>,
          ],
          [
            <><code>v_call</code>, <code>d_call</code>, <code>j_call</code></>,
            <>
              <code>v_call</code> lists the V alleles that match the read equally well, comma-separated after the
              call (read02 below); AlignAIR's <code>d_call</code> and <code>j_call</code> name one allele. On the IG
              models the names are OGRDB labels.
            </>,
          ],
          [
            <><code>v_identity</code>, <code>d_identity</code>, <code>j_identity</code></>,
            <>
              AlignAIR writes a fraction from 0 to 1, as the AIRR standard defines it; IgBLAST writes a percentage. A
              filter <code>{"v_identity >= 98"}</code> becomes <code>{"v_identity >= 0.98"}</code>.
            </>,
          ],
          [
            <><code>sequence_alignment</code>, <code>germline_alignment</code></>,
            "AlignAIR's are IMGT-gapped (dots in the V gaps), the form Change-O works with.",
          ],
          [
            <><code>v_cigar</code>, <code>d_cigar</code>, <code>j_cigar</code></>,
            <>
              Both span the whole read: S for the read's bases outside the segment, N for the germline bases skipped at
              its start, then the alignment (<code>4N291M73S</code>). IgBLAST's also ends with an N for the germline
              bases after the segment, which the AIRR standard leaves optional.
            </>,
          ],
          [
            <><code>fwr1</code> to <code>fwr4</code>, <code>cdr1</code> to <code>cdr3</code>, with their starts and ends</>,
            <>
              The positions are the same: 1-based, in the read. AlignAIR cuts FWR1 to CDR2 from the IMGT-gapped
              alignment, so they hold its dots, as MakeDb's do. FWR4 is IMGT's FR4 in both, from the J's Trp or Phe
              codon.
            </>,
          ],
          [
            <>
              <code>v_score</code>, <code>v_support</code> (and their D and J forms), <code>v_frameshift</code>,{" "}
              <code>complete_vdj</code>, <code>d_frame</code>
            </>,
            <>
              IgBLAST only. AlignAIR gives no e-value: the candidate set and the assembly status (below) are what to read
              instead. <DocLink to="trust-the-calls">How far to trust the calls</DocLink> explains both.
            </>,
          ],
          [
            <code>c_call</code>,
            <>
              AlignAIR does not call the constant region: the column stays blank unless a <code>--metadata</code> table
              fills it (Cell Ranger's <code>c_gene</code>).
            </>,
          ],
        ]}
      />
      <p>A real row (read02 of the example reads), where two V alleles match the read equally well:</p>
      <CodeBlock
        lang="text"
        title="alignair.tsv (some columns)"
        code={`sequence_id  v_call                         v_call_set                     v_identity          j_call    airr_assembly_status
read02       IGHVF10-G42*01,IGHVF10-G42*03  IGHVF10-G42*01,IGHVF10-G42*03  0.9690721649484536  IGHJ4*02  complete`}
      />

      <H3 id="alignair-only-columns">Columns only AlignAIR writes</H3>
      <ColumnTable
        names={[
          "productive_prediction",
          "v_call_set",
          "d_call_set",
          "j_call_set",
          "mutation_rate",
          "orientation",
          "input_sequence",
          "segmentation_low_quality",
          "length_cropped",
          "airr_assembly_status",
          "airr_assembly_reason",
          "airr_assembly_error",
        ]}
      />
      <p>Two habits to add to an IgBLAST pipeline:</p>
      <ul>
        <li>
          Keep the rows with <code>airr_assembly_status</code> <code>complete</code> before junction-level work. A{" "}
          <code>partial</code> row has valid calls but some fields left blank; <code>airr_assembly_reason</code> says
          why.
        </li>
        <li>
          Read a <code>*_call_set</code> with several alleles as "not resolved to one allele". It is the model's set
          before the germline refinement, so the call can fall outside it.
        </li>
      </ul>
      <p>
        Every column, with its unit and the presets that include it, is in the{" "}
        <DocLink to="column-dictionary">output column dictionary</DocLink>.
      </p>

      <H2 id="fair-comparison">A fair side-by-side</H2>
      <p>
        To compare the two tools, give IgBLAST the germline set the model calls from, so both choose among the same
        alleles. The steps for the IGH model:
      </p>
      <CodeBlock
        code={`# 1. The model's germlines, ungapped, in one FASTA
alignair reference export "$(alignair models path alignair4-igh-human)" --fasta igh.fasta

# 2. One FASTA per segment (the letter after IGH in each name: V, D or J)
awk '/^>/ { segment = substr($0, 5, 1) } { print > ("igh_" segment ".fasta") }' igh.fasta

# 3. One BLAST database per segment
for segment in V D J; do
  makeblastdb -parse_seqids -dbtype nucl -in igh_$segment.fasta -out igh_$segment
done`}
      />
      <p>
        <code>reference export</code> reads a model file: <code>alignair models path</code> prints where the model's
        file is (downloading it first if needed). The <code>awk</code> line reads the fifth character of each header,
        the segment letter after the three-letter locus (the same for IGK, IGL and the TCR loci).
      </p>

      <H3 id="auxiliary-j-file">The auxiliary J file</H3>
      <p>
        IgBLAST finds each J allele's reading frame and CDR3 end in its auxiliary file, by allele name. A J allele the
        file does not list gets no junction. This lists the J alleles it lacks:
      </p>
      <CodeBlock
        code={`awk 'NR == FNR { listed[$1]; next } /^>/ { name = substr($1, 2); if (!(name in listed)) print "no line for " name }' \\
  "$IGDATA/optional_file/human_gl.aux" igh_J.fasta`}
      />
      <p>
        <code>IGDATA</code> is IgBLAST's own directory, where <code>igblastn</code> also finds its internal data. For the
        IGH model, the release's <code>human_gl.aux</code> should cover the J alleles. The IGK and IGL models' J alleles
        have OGRDB labels (<code>IGKJ2-27FL*01</code>) it does not list: copy the file and add one line per missing
        allele, in its tab-separated format: the allele name, the first coding frame start, the chain type (
        <code>JK</code> or <code>JL</code>), the CDR3 end and the extra bases after the last codon, positions counted
        from 0.
      </p>

      <H3 id="run-both">Run both, then compare</H3>
      <CodeBlock
        code={`# 4. IgBLAST on your reads, against the model's germlines
igblastn -germline_db_V igh_V -germline_db_D igh_D -germline_db_J igh_J \\
  -auxiliary_data "$IGDATA/optional_file/human_gl.aux" -organism human -ig_seqtype Ig \\
  -outfmt 19 -query reads.fasta -out igblast.tsv

# 5. AlignAIR on the same reads, then where the two agree
alignair predict --model alignair4-igh-human --input reads.fasta --out alignair.tsv
alignair compare --a alignair.tsv --b igblast.tsv --a-name AlignAIR --b-name IgBLAST --out report.md`}
      />
      <p>
        For a locus without D (IGK, IGL, TRA, TRG), <code>igblastn</code> still asks for a D database: a one-sequence
        placeholder is enough, as it searches no D on those chains. Use <code>-ig_seqtype TCR</code> for the TCR models.
      </p>
      <p>
        The report gives, per gene, how often the two tools call the same allele and the same gene, and the set rescue:
        of the reads where the calls differ, how often IgBLAST's call is inside AlignAIR's <code>*_call_set</code>{" "}
        (shared ambiguity rather than a conflict). Agreement is not accuracy: without known truth, neither tool is the
        reference. The measured comparison, on simulated reads with known truth for all seven loci, is on{" "}
        <DocLink to="comparison">Comparison with IgBLAST</DocLink>.
      </p>

      <H2 id="next">Next</H2>
      <ul>
        <li>
          <DocLink to="from-immcantation">Coming from Immcantation</DocLink>: Change-O, TIgGER and SCOPer on AlignAIR's
          table.
        </li>
        <li>
          <DocLink to="donor-genotype">Align with a donor genotype</DocLink>: call only from a donor's alleles.
        </li>
        <li>
          <DocLink to="trust-the-calls">How far to trust the calls</DocLink>: the known failure modes.
        </li>
      </ul>
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* MiXCR                                                                                                              */
/* ------------------------------------------------------------------------------------------------------------------ */

const fromMixcr: DocPage = {
  slug: "from-mixcr",
  title: "Coming from MiXCR",
  section: "Coming from…",
  lead: "What AlignAIR takes over from a MiXCR pipeline (assigning V(D)J alleles to reads), what it leaves to other tools, and how to give it your reads.",
  body: () => (
    <>
      <H2 id="two-jobs">Two different jobs</H2>
      <p>
        MiXCR covers the whole road from raw sequencing data to clonotypes. It aligns reads to V, D, J and C genes,
        corrects UMI and cell barcodes, assembles clonotypes and contigs, and exports clone tables and QC reports,
        usually through one <code>mixcr analyze</code> command with a preset for the library.
      </p>
      <p>
        AlignAIR does one step of that road: it assigns V, D and J alleles to each read, from a named germline set, and
        writes an AIRR rearrangement table with one row per read. It does not assemble clones.
      </p>

      <H2 id="what-maps">What maps and what does not</H2>
      <DocTable
        head={["In MiXCR", "With AlignAIR"]}
        rows={[
          [
            "Aligning reads to V, D and J genes (its align step)",
            <>
              <code>alignair predict</code>: allele calls with their ties, the junction, IMGT-gapped alignments, FWR
              and CDR regions.
            </>,
          ],
          [
            "Its built-in gene library",
            <>
              The model's germline set (OGRDB for the IG models, IMGT for the TCR models), or your own with{" "}
              <code>--reference</code>. Allele names differ between the two tools.
            </>,
          ],
          ["Several loci in one run", "One model per locus: give each model the reads of its locus."],
          [
            "C gene alignment (isotypes)",
            <>
              Not called. <code>c_call</code> comes only from a <code>--metadata</code> table, such as Cell Ranger's{" "}
              <code>c_gene</code>.
            </>,
          ],
          ["Paired-end FASTQ files", "One sequence per read: merge the pairs first."],
          [
            "UMI and cell-barcode correction",
            <>
              None. Build one consensus read per UMI first (<DocLink to="from-mixcr?h=umi-counts">below</DocLink>), and
              carry barcodes and counts in with <code>--metadata</code>.
            </>,
          ],
          [
            "Clonotype and contig assembly",
            <>
              None: AlignAIR does not assemble contigs, pair chains or group reads into clonotypes. For T cells, see{" "}
              <DocLink to="from-mixcr?h=tcr-clonotypes">T-cell clonotypes</DocLink>; for B cells, Change-O or SCOPer (
              <DocLink to="from-immcantation?h=clonal-clustering">how</DocLink>).
            </>,
          ],
          [
            "Clone tables and QC reports",
            <>
              The per-read AIRR table; <code>alignair analyze</code> prints a summary of it.
            </>,
          ],
          ["Receptor reads found in bulk RNA-seq", "Not covered: AlignAIR expects reads that are receptor sequences."],
        ]}
      />
      <p>
        Keep MiXCR when you want its end-to-end pipeline to clonotypes. Reach for AlignAIR when the allele calls
        themselves feed your analysis: donor genotypes (<code>--genotype</code>, <code>--infer-genotype</code>), novel
        alleles (<code>--reference</code>), and the Immcantation tools that work from them.
      </p>

      <H2 id="where-it-fits">Where AlignAIR fits</H2>
      <ol>
        <li>
          <strong>Prepare the reads.</strong> Merge read pairs, and for UMI libraries build one consensus read per UMI (
          <DocLink to="from-mixcr?h=umi-counts">below</DocLink>). Cell Ranger's contigs need neither: align them as
          they are.
        </li>
        <li>
          <strong>Assign alleles.</strong> <code>alignair predict</code> with the model of each locus.
        </li>
        <li>
          <strong>Group clonotypes.</strong> T cells: by V gene, J gene and junction, with no threshold (
          <DocLink to="from-mixcr?h=tcr-clonotypes">below</DocLink>). B cells, which hypermutate: Change-O's{" "}
          <code>DefineClones.py</code> or SCOPer, with a distance threshold (
          <DocLink to="from-immcantation?h=clonal-clustering">Coming from Immcantation</DocLink>).
        </li>
        <li>
          <strong>Go on downstream.</strong> T cells: Scirpy and other tools that read AIRR tables (
          <DocLink to="from-mixcr?h=tcr-tools">below</DocLink>). B cells: germlines, genotypes, mutation and lineage
          analysis with the rest of Immcantation.
        </li>
      </ol>

      <H2 id="feed-reads">Give it your reads</H2>

      <H3 id="merged-reads">Merged reads</H3>
      <CodeTabs
        cli={`alignair predict --model alignair4-igh-human --input merged.fastq.gz --out alignair.tsv`}
        python={`from alignair import Aligner, read_sequences

aligner = Aligner.from_pretrained("alignair4-igh-human")
ids, reads, info = read_sequences("merged.fastq.gz")
aligner.predict(reads, ids=ids).write_airr("alignair.tsv")`}
      />
      <p>
        AlignAIR turns each read the right way round itself (the <code>orientation</code> and <code>rev_comp</code>{" "}
        columns say how), so reads in mixed orientations are fine. IUPAC ambiguity codes are read as N; a read in which
        more than 20% of the symbols are something else (a gap, a digit) is dropped, and <code>--rejects-out</code>{" "}
        keeps the dropped reads in a file.
      </p>
      <Callout kind="warning" title="Reads over 575 nt">
        A read longer than 575 nt is aligned on its first 575 nt only (<code>length_cropped</code> is <code>T</code>).
        The cut comes before the read is turned: a forward read loses its 3' end, a reverse-complemented one the start
        of its V. A row cut inside its J is <code>partial</code> (<code>j_cropped</code>). Trim long reads to the
        V(D)J region first. Cell Ranger's contigs are forward, and the cut usually takes only constant region after the
        J. See <DocLink to="prepare-input?h=long-reads">Prepare your input</DocLink>.
      </Callout>

      <H3 id="umi-counts">UMI consensus reads and their counts</H3>
      <p>
        AlignAIR aligns each read as it is. It does not group reads by UMI, correct UMI or barcode errors, or build a
        consensus, so two reads of one molecule get two rows. Build the consensus reads first: MiXCR builds its own
        inside its pipeline, and AlignAIR needs them as sequences (FASTA, FASTQ or a table). Outside MiXCR, pRESTO's{" "}
        <code>BuildConsensus.py</code> builds one read per UMI from reads whose UMI is in their header (
        <a href="https://presto.readthedocs.io" target="_blank" rel="noreferrer">pRESTO's documentation</a> has
        workflows for UMI libraries); pRESTO is installed with Change-O.
      </p>
      <p>
        Then align the consensus reads, and join their counts by read id with <code>--metadata</code>. A table with a{" "}
        <code>sequence_id</code> column and columns such as <code>umi_count</code>, <code>consensus_count</code>,{" "}
        <code>duplicate_count</code> or <code>sample_id</code> adds them to each row:
      </p>
      <CodeBlock
        code={`alignair predict --model alignair4-igh-human --input consensus.fastq.gz --out alignair.tsv \\
  --metadata counts.tsv`}
      />
      <p>
        By default the join carries the known AIRR and 10x columns it finds; <code>--keep-columns</code> picks them
        yourself. <DocLink to="metadata-join">Join sample and cell metadata</DocLink> has the rules.
      </p>

      <H3 id="reads-in-a-table">Reads in a table</H3>
      <CodeTabs
        cli={`alignair predict --model alignair4-igh-human --input reads.tsv --out alignair.tsv \\
  --sequence-column sequence --id-column sequence_id`}
        python={`from alignair import Aligner, read_sequences

aligner = Aligner.from_pretrained("alignair4-igh-human")
ids, reads, info = read_sequences("reads.tsv", seq_column="sequence", id_column="sequence_id")
aligner.predict(reads, ids=ids).write_airr("alignair.tsv")`}
      />

      <H3 id="single-cell">10x single-cell contigs</H3>
      <CodeBlock
        code={`alignair predict --model alignair4-trb-human --input contigs_TRB.fasta --out TRB.tsv \\
  --metadata filtered_contig_annotations.csv`}
      />
      <p>
        Cell Ranger's <code>barcode</code>, <code>umis</code>, <code>reads</code> and <code>c_gene</code> become{" "}
        <code>cell_id</code>, <code>umi_count</code>, <code>consensus_count</code> and <code>c_call</code>. A contig file
        holds every chain, and each model calls one locus, so split the contigs by the annotations'{" "}
        <code>chain</code> column first and align each chain with its own model: a TRA contig given to the TRB model
        can come out as a complete, productive TRB row. <DocLink to="metadata-join">Join sample and cell metadata</DocLink>{" "}
        gives the split, for TRA, TRB, TRG and TRD alike, and how to put the tables back together.
      </p>

      <H2 id="t-cells">T-cell data</H2>

      <H3 id="tcr-clonotypes">T-cell clonotypes</H3>
      <p>
        T cells do not hypermutate, so a T-cell clonotype is usually the reads, or cells, with the same V gene, J gene
        and junction, base for base or in amino acids. No distance threshold is chosen, unlike the SHazaM threshold of
        B-cell clones. Change-O's <code>DefineClones.py</code> groups them with a distance of 0:
      </p>
      <CodeBlock code={`DefineClones.py -d TRB.tsv --act set --model ham --norm none --dist 0`} />
      <p>
        It writes <code>TRB_clone-pass.tsv</code> with a <code>clone_id</code> column. A row without a junction cannot
        be grouped, and DefineClones counts it as failed. For single cells, a clonotype usually also pairs the chains of
        each cell (by <code>cell_id</code>): Scirpy does that, below.
      </p>

      <H3 id="tcr-tools">Scirpy and other tools</H3>
      <p>
        Scirpy reads AlignAIR's tables as they are, one per locus, and puts the chains with the same{" "}
        <code>cell_id</code> together (the <code>cell_id</code> comes from Cell Ranger's annotations, through{" "}
        <code>--metadata</code>). Checked with Scirpy 0.26.1, on the TRA and TRB tables of 300 simulated cells:
      </p>
      <CodeBlock
        lang="python"
        code={`import scirpy as ir

adata = ir.io.read_airr(["TRA.tsv", "TRB.tsv"])     # one cell per cell_id
ir.pp.index_chains(adata)
ir.tl.chain_qc(adata)
ir.pp.ir_dist(adata)
ir.tl.define_clonotypes(adata, receptor_arms="all", dual_ir="primary_only", same_v_gene=True)`}
      />
      <p>
        <code>define_clonotypes</code> groups the cells whose chains have the same CDR3 in nucleotides;{" "}
        <code>same_v_gene=True</code> also asks for the same V gene. <code>index_chains</code> keeps the productive
        chains of each cell. Other tools that read the AIRR rearrangement format take AlignAIR's table the same way;
        check the columns each one needs.
      </p>

      <H3 id="check-against-cell-ranger">Check the calls against Cell Ranger or MiXCR</H3>
      <p>
        <code>alignair compare</code> matches two AIRR tables row by row on <code>sequence_id</code> and reports, per
        segment, how often the calls agree. Cell Ranger's <code>airr_rearrangement.tsv</code> names each row by its
        contig id, as in <code>filtered_contig.fasta</code>, so it matches AlignAIR's table of the same contigs as it
        is (the rows of the other chains count as Cell Ranger's only). A MiXCR table matches only when its{" "}
        <code>sequence_id</code> holds your read ids. On 300 simulated TRB contigs, against a table in Cell Ranger's
        layout that names each contig's true genes, without alleles:
      </p>
      <CodeBlock
        title="shell (output shortened)"
        code={`$ alignair compare --a TRB.tsv --b airr_rearrangement.tsv --a-name AlignAIR --b-name CellRanger
# AIRR agreement: AlignAIR vs CellRanger

Matched **300** sequences by \`sequence_id\` (AlignAIR: 300, CellRanger: 300; AlignAIR-only 0, CellRanger-only 0).

## Call agreement (rows where both tools called the gene)

| gene | n | allele agreement | gene agreement | set-rescue of disagreements |
| --- | --- | --- | --- | --- |
| V | 300 | 0.0% | 96.7% | 0.0% |
| D | 0 | n/a | n/a | n/a |
| J | 300 | 0.0% | 100.0% | 0.0% |`}
      />
      <p>
        Read the gene agreement column. A table that names genes only, or alleles of another germline set (Cell
        Ranger's reference, MiXCR's library), agrees on no allele name, so its allele agreement says nothing. Where the
        tools differ, either may be right. <DocLink to="evaluate?h=compare">Evaluate your model</DocLink> explains the
        rest of the report.
      </p>

      <H3 id="tcr-measured">What has been measured on T-cell receptors</H3>
      <p>
        The <DocLink to="comparison">comparison with IgBLAST</DocLink> measures the four TCR models, like the IG
        models, on simulated reads with known truth. Its check on real reads covers human IGH repertoires only, whose
        donors' alleles are known from their genomes; no real TCR repertoire has been scored that way. So judge the TCR
        models by the simulated results, and check a sample of your own reads against the tool you use today, as
        above.
      </p>

      <H2 id="next">Next</H2>
      <ul>
        <li>
          <DocLink to="from-immcantation">Coming from Immcantation</DocLink>: genotypes, germlines and clones on
          AlignAIR's table.
        </li>
        <li>
          <DocLink to="choose-a-model">Choose a model</DocLink>: the seven models and their germline sets.
        </li>
        <li>
          <DocLink to="run-at-scale">Run at scale</DocLink>: GPU, worker processes and large inputs.
        </li>
      </ul>
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Immcantation                                                                                                       */
/* ------------------------------------------------------------------------------------------------------------------ */

const fromImmcantation: DocPage = {
  slug: "from-immcantation",
  title: "Coming from Immcantation",
  section: "Coming from…",
  lead: "Replace AssignGenes and MakeDb with one alignair predict, and keep Change-O, TIgGER and SCOPer on the table it writes.",
  body: () => (
    <>
      <H2 id="assign-and-makedb">AssignGenes and MakeDb become one command</H2>
      <p>
        In the usual Change-O route, <code>AssignGenes.py igblast</code> runs IgBLAST on the reads, and{" "}
        <code>MakeDb.py igblast</code> turns IgBLAST's output into an AIRR table with IMGT-gapped alignments. AlignAIR
        does both in one step: its table is already AIRR, with an IMGT-gapped <code>sequence_alignment</code> and{" "}
        <code>germline_alignment</code>.
      </p>
      <CodeTabs
        cli={`$ alignair predict --model alignair4-igh-human --input reads.fasta --out alignair.tsv
aligned 20 reads (0 dropped) -> alignair.tsv`}
        python={PREDICT_PYTHON}
      />
      <p>
        Keep the default columns (all 99). Change-O, TIgGER and SCOPer read <code>sequence_alignment</code>,{" "}
        <code>junction_length</code> and <code>locus</code>, and the lighter presets (<code>core</code>,{" "}
        <code>airr</code>, <code>minimal</code>) each drop some of them.
      </p>
      <p>
        When a read cannot tell V alleles apart, <code>v_call</code> lists the call and then the alleles that match the
        read as well, comma-separated, as IgBLAST does. Change-O and TIgGER read these lists as they read IgBLAST's.
      </p>
      <p>
        Reads from pRESTO keep their whole header as <code>sequence_id</code>: AlignAIR does not split{" "}
        <code>&gt;read01|CONSCOUNT=2|DUPCOUNT=5|PRCONS=IGHG</code> into the <code>consensus_count</code> and{" "}
        <code>duplicate_count</code> columns MakeDb writes. <DocLink to="prepare-input">Prepare your input</DocLink>{" "}
        shows how to cut the headers to the read id and join the annotations back with <code>--metadata</code>.
      </p>

      <H2 id="create-germlines">Germlines for CreateGermlines</H2>
      <p>
        <code>CreateGermlines.py</code> rebuilds each read's germline from IMGT-gapped V alleles named as in the table.
        Export them from the model:
      </p>
      <CodeBlock
        code={`$ alignair reference export "$(alignair models path alignair4-igh-human)" --fasta germlines.fasta --gapped
wrote germlines.fasta`}
      />
      <p>
        <code>--gapped</code> writes each V allele IMGT-gapped and D and J as they are, all in one FASTA, under the names
        the model calls. <code>reference export</code> reads a model file: <code>alignair models path</code> prints where
        the model's file is (downloading it first if needed).
      </p>
      <p>Then build the germlines from AlignAIR's table as it is:</p>
      <CodeBlock code={`CreateGermlines.py -d alignair.tsv -r germlines.fasta`} />
      <p>
        Where the AIRR schema leaves a choice, AlignAIR's table counts as MakeDb does, so CreateGermlines reads it
        unchanged. <code>v_germline_start</code> and <code>v_germline_end</code> are positions in the IMGT-gapped V
        germline, from its first position, as <code>sequence_alignment</code> lays out the V. The untemplated bases are N
        in <code>germline_alignment</code>. A row without a D call holds every base between V and J in{" "}
        <code>np1</code>. The CIGARs count ungapped germline bases, as the schema says.
      </p>
      <p>
        CreateGermlines writes <code>alignair_germ-pass.tsv</code> with a <code>germline_alignment_d_mask</code> column,
        the germline Change-O's later steps read. Its log counts the rows it could not build; Change-O's{" "}
        <code>--failed</code> option writes them to a file. Most of those are partial rows (
        <code>airr_assembly_status</code>), which have no gapped alignment. A complete row also fails when its J
        alignment has more or fewer germline bases than read bases (an insertion or deletion in J). CreateGermlines takes
        the J's length from <code>j_germline_start</code> and <code>j_germline_end</code>, while{" "}
        <code>sequence_alignment</code> keeps the read's own J bases. On 3,000 real IGH reads, 2,855 of the 2,872
        complete rows passed; each of the other 17 has such a J.
      </p>
      <p>
        With novel alleles, give CreateGermlines the FASTA you bound with <code>--reference</code>, the export with
        the novel alleles added (<DocLink to="from-immcantation?h=tigger">TIgGER on AlignAIR's table</DocLink>).
      </p>

      <H2 id="genotypes">Donor genotypes</H2>
      <p>
        TIgGER infers a donor's V genotype from a table, and can re-assign each read's alleles to that genotype.
        AlignAIR's <code>--genotype</code> does the re-assignment inside the alignment: given the donor's alleles, it
        calls only from them. The file is YAML or JSON keyed by segment, <code>{"{V: [...], D: [...], J: [...]}"}</code>:
      </p>
      <CodeBlock
        lang="yaml"
        title="donor.yaml"
        code={`# Every V allele the donor carries (three shown), named as the model names them
V:
  - IGHVF10-G50*02
  - IGHVF3-G9*01
  - IGHVF3-G9*03`}
      />
      <p>
        The V list is the donor's whole V genotype: V alleles it does not list are not called. A segment it leaves out
        (here D and J) keeps all its alleles.
      </p>
      <CodeTabs
        cli={`alignair predict --model alignair4-igh-human --input reads.fasta --out alignair.tsv --genotype donor.yaml`}
        python={`from alignair import Aligner, read_sequences

aligner = Aligner.from_pretrained("alignair4-igh-human").with_genotype(
    {"V": ["IGHVF10-G50*02", "IGHVF3-G9*01", "IGHVF3-G9*03"]})
ids, reads, info = read_sequences("reads.fasta")
aligner.predict(reads, ids=ids).write_airr("alignair.tsv")`}
      />

      <H3 id="imgt-and-ogrdb-names">IMGT names and OGRDB names</H3>
      <p>
        A genotype must name alleles the way the model does. A genotype made from IgBLAST calls against IMGT germlines
        names alleles such as <code>IGHV3-23*01</code>. The IG models call OGRDB labels, so AlignAIR refuses it:
      </p>
      <CodeBlock
        code={`$ alignair predict --model alignair4-igh-human --input reads.fasta --out alignair.tsv --genotype imgt_genotype.yaml
[ALIGNAIR_GENOTYPE_ALLELE_NOT_FOUND] the genotype names alleles the reference does not hold: IGHV1-2*02, IGHV3-23*01
Context: {"command": "predict", "missing": ["IGHV1-2*02", "IGHV3-23*01"], "stage": "genotype_validation"}
Hint: Give their sequences in the reference (alignair predict --reference; in Python, Aligner.from_pretrained(reference=) or bound_to).`}
      />
      <p>Three ways through:</p>
      <ul>
        <li>
          <strong>Make the genotype from AlignAIR's own calls.</strong> Use <code>alignair genotype</code> or{" "}
          <code>--infer-genotype</code> (below), or run TIgGER on AlignAIR's table (
          <DocLink to="from-immcantation?h=tigger">how</DocLink>): the names are then the model's.
        </li>
        <li>
          <strong>Translate the names.</strong> Match the model's alleles to IMGT names by sequence, then write the
          model's labels into the genotype (<DocLink to="choose-a-model?h=map-names">Map the labels to IMGT names</DocLink>
          ).
        </li>
        <li>
          <strong>Keep IMGT names.</strong> Bind IMGT's germline set with <code>--reference</code>; Immcantation's
          IMGT/GENE-DB files are read as they are, headers included. The calls then carry IMGT names, and the IMGT
          genotype fits (<DocLink to="choose-a-model?h=bind-imgt">Call with IMGT's set</DocLink>).
        </li>
      </ul>
      <p>The TCR models call IMGT names, so an IMGT-named genotype fits them as it is.</p>

      <H3 id="alignair-genotype">alignair genotype: a genotype from any table</H3>
      <p>
        <code>alignair genotype</code> reads any tool's AIRR table and infers the V genotype. For each V gene it keeps
        the most frequent alleles among the rows whose <code>v_call</code> is a single allele, until they explain 87.5%
        of those rows (<code>--explain</code>), the share TIgGER's <code>inferGenotype</code> also uses. It differs from{" "}
        <code>inferGenotype</code> in two ways:
      </p>
      <ul>
        <li>
          it counts every such row, mutated or not, where <code>inferGenotype</code> counts only unmutated reads by
          default (<code>find_unmutated = TRUE</code>);
        </li>
        <li>
          a gene with fewer than 10 such rows (<code>--min-reads</code>) has too little evidence and is left open, a
          cutoff of AlignAIR's own.
        </li>
      </ul>
      <p>
        It writes an AIRR GenotypeSet; <code>--locus</code> names its locus (default IGH). On 29,101 reads of one donor
        (the first 30,000 read pairs of ENA run SRR8283604, merged):
      </p>
      <CodeBlock
        title="shell (output abridged)"
        code={`$ alignair genotype alignair.tsv --out genotype.json
IGHVF1-G1: open (2 single-allele rows, fewer than 10)
IGHVF1-G2: IGHVF1-G2*01
IGHVF1-G3: IGHVF1-G3*03
IGHVF10-G33: IGHVF10-G33*03
IGHVF10-G37: IGHVF10-G37*03, IGHVF10-G37*10, IGHVF10-G37*07
...
genotyped 33 of 39 V genes -> genotype.json`}
      />
      <p>
        A gene is left open when few of its rows name a single allele, however many rows it has: 4,367 of the 4,371
        reads called to <code>IGHVF10-G41</code> here list <code>*02</code> and <code>*03</code> together, as they
        start after the bases that tell the two apart. The GenotypeSet goes straight back into{" "}
        <code>--genotype</code>:
      </p>
      <CodeBlock
        code={`alignair predict --model alignair4-igh-human --input reads.fasta --out alignair_genotyped.tsv \\
  --genotype genotype.json`}
      />
      <p>
        A GenotypeSet restricts only the genes it names. A gene it left open keeps all its alleles: in the run above,
        the 4,388 reads of the six open genes all kept their gene.
      </p>
      <p>
        <code>--infer-genotype</code> does both steps in one run. It reads the input twice: once to call, once to
        re-call the reads whose call the genotype excludes while they tie an allele it allows. A gene with fewer than{" "}
        <code>--genotype-min-reads</code> single-allele reads keeps every allele. The genotype goes to{" "}
        <code>alignair.tsv.genotype.json</code> (<code>--genotype-out</code>). More on{" "}
        <DocLink to="donor-genotype">Align with a donor genotype</DocLink>.
      </p>
      <CodeBlock
        code={`alignair predict --model alignair4-igh-human --input reads.fasta --out alignair.tsv --infer-genotype`}
      />

      <H2 id="tigger">TIgGER on AlignAIR's table</H2>
      <p>
        TIgGER reads AlignAIR's table as it reads MakeDb's: <code>v_call</code>, <code>j_call</code>, the IMGT-gapped{" "}
        <code>sequence_alignment</code>, <code>junction</code> and <code>junction_length</code>. Its germline file is
        the <code>--gapped</code> export (<DocLink to="from-immcantation?h=create-germlines">above</DocLink>): IMGT-gapped
        V alleles, named as the table names them. Keep that one file for the whole round trip: TIgGER reads it, the
        novel alleles TIgGER finds are added to it, and AlignAIR and CreateGermlines both read the result.
      </p>

      <H3 id="tigger-run">Novel alleles and the genotype</H3>
      <CodeBlock lang="r" title="R (TIgGER 1.2.0)" code={TIGGER_RUN} />
      <p>
        On the 29,101 reads above, <code>selectNovel(novel)</code> holds two novel alleles of <code>IGHVF6-G21</code>,
        named the TIgGER way (the allele they differ from, then each substitution at its IMGT position):{" "}
        <code>IGHVF6-G21*12_G163A</code> and <code>IGHVF6-G21*16_G54A</code>. The genotype keeps the first. These reads
        start near IMGT position 40, and their first few bases differ from every germline, as a primer's do: with the
        default <code>pos_range</code> (1 to 312), <code>findNovelAlleles</code> also reported "novel" alleles of four
        genes at positions 42 to 46. Leave out the positions your primers cover.
      </p>

      <H3 id="tigger-back">Back into AlignAIR</H3>
      <p>AlignAIR does not read TIgGER's genotype table: written to a TSV and given to <code>--genotype</code>, it is refused:</p>
      <CodeBlock
        code={`$ alignair predict --model alignair4-igh-human --input reads.fasta --out x.tsv --genotype tigger_genotype.tsv
[ALIGNAIR_CONFIG_INVALID] tigger_genotype.tsv is not a YAML or JSON genotype: it does not parse (line 1): found character '\\t' that cannot start any token
Context: {"command": "predict", "path": "tigger_genotype.tsv", "stage": "genotype_validation"}
Hint: A genotype is YAML or JSON keyed by segment (V, D, J), each listing allele names of the bound reference, e.g. 'V: [IGHV1-2*02, IGHV3-23*01]'. Germline sequences (a FASTA) go to --reference.`}
      />
      <p>
        In the same R session, write the genotype as a YAML genotype, and the novel alleles' gapped sequences (TIgGER's{" "}
        <code>novel_imgt</code>) after the gapped germlines:
      </p>
      <CodeBlock lang="r" title="R" code={TIGGER_BACK} />
      <p>Then align with both, and build the germlines from the same file:</p>
      <CodeTabs
        cli={`alignair predict --model alignair4-igh-human --input reads.fasta --out alignair_donor.tsv \\
  --reference germlines_with_novel.fasta --genotype donor.yaml
CreateGermlines.py -d alignair_donor.tsv -r germlines_with_novel.fasta`}
        python={`from alignair import Aligner, ReferenceSet, read_sequences
from alignair.genotype import load_genotype

reference = ReferenceSet.from_fasta("germlines_with_novel.fasta")
aligner = Aligner.from_pretrained("alignair4-igh-human", reference=reference)
donor = aligner.with_genotype(load_genotype("donor.yaml"))
ids, reads, info = read_sequences("reads.fasta")
donor.predict(reads, ids=ids).write_airr("alignair_donor.tsv")`}
      />
      <CodeBlock
        title="shell"
        code={`$ cut -f5 alignair_donor.tsv | cut -d, -f1 | sort | uniq -c | grep _G163A
    840 IGHVF6-G21*12_G163A`}
      />
      <p>
        840 of the 29,101 reads are now called to the novel allele (the fifth column is <code>v_call</code>). A YAML
        genotype calls only the V alleles it lists, as TIgGER's <code>reassignAlleles</code> does: the reads of a gene
        TIgGER left out of the genotype move to another gene or lose their V call (492 reads here). To keep such genes
        whole, use <code>alignair genotype</code>'s GenotypeSet instead (
        <DocLink to="from-immcantation?h=alignair-genotype">above</DocLink>).
      </p>
      <p>
        The novel alleles carry the model's labels, as the table does. To work in IMGT names throughout, bind IMGT's
        gapped germline set (<DocLink to="choose-a-model?h=bind-imgt">Call with IMGT's set</DocLink>) and use that file
        where these steps use <code>germlines.fasta</code>: TIgGER and CreateGermlines read its IMGT/GENE-DB headers
        too. On the same reads, with IMGT's human IGH set bound, TIgGER found the same novel allele as{" "}
        <code>IGHV1-69*04_G54A</code>, and 840 reads were called to it. To translate labels instead, see{" "}
        <DocLink to="choose-a-model?h=map-names">Map the labels to IMGT names</DocLink>.
      </p>

      <H3 id="v-call-genotyped">v_call_genotyped</H3>
      <p>
        TIgGER's <code>reassignAlleles</code> writes its genotyped calls to a new column, <code>v_call_genotyped</code>,
        and the steps after it are pointed at that column (<code>DefineClones.py --vf v_call_genotyped</code>, for
        example). A <code>--genotype</code> run needs no such step: its <code>v_call</code> already holds calls from
        the donor's alleles, and AlignAIR writes no <code>v_call_genotyped</code> column. Point those steps at{" "}
        <code>v_call</code>, or run <code>reassignAlleles</code> on AlignAIR's table if your pipeline expects the
        column: it reads the table as it is.
      </p>

      <H3 id="tigger-or-alignair">TIgGER, alignair genotype or --infer-genotype?</H3>
      <DocTable
        head={["You want", "Use"]}
        rows={[
          [
            "Novel alleles",
            <>
              TIgGER's <code>findNovelAlleles</code>. AlignAIR finds none: it calls only alleles of the germline set it
              runs with.
            </>,
          ],
          [
            "A genotype from unmutated reads only, as in a heavily mutated repertoire",
            <>
              TIgGER's <code>inferGenotype</code> (unmutated reads by default), or its Bayesian form,{" "}
              <code>inferGenotypeBayesian</code>.
            </>,
          ],
          [
            "A genotype without R, from any tool's AIRR table, that leaves genes with little evidence open",
            <>
              <code>alignair genotype</code>, then <code>--genotype genotype.json</code>.
            </>,
          ],
          [
            "The genotype applied in the same run, settling the reads whose v_call ties an allele of the genotype with others",
            <code>--infer-genotype</code>,
          ],
        ]}
      />

      <H2 id="clonal-clustering">Clonal clustering: Change-O and SCOPer</H2>
      <p>Change-O's clonal clustering runs on the table from CreateGermlines:</p>
      <CodeBlock code={`DefineClones.py -d alignair_germ-pass.tsv --act set --model ham --norm len --dist 0.16`} />
      <p>
        <code>--act set</code> groups reads by every gene of a tied call, so AlignAIR's comma-separated calls are used
        whole. The threshold 0.16 is only an example: choose yours from your data's distance-to-nearest distribution
        (SHazaM). The result, <code>alignair_germ-pass_clone-pass.tsv</code>, adds a <code>clone_id</code>{" "}
        column. T cells need no threshold (<DocLink to="from-mixcr?h=tcr-clonotypes">T-cell clonotypes</DocLink>).
      </p>
      <p>
        SCOPer reads the same table in R. Its clone functions group rows by <code>v_call</code> and <code>j_call</code>{" "}
        and compare <code>junction</code>, the column names AlignAIR writes. For single-cell data they also use{" "}
        <code>cell_id</code>, which <code>--metadata</code> carries from Cell Ranger's barcodes (
        <DocLink to="metadata-join">Join sample and cell metadata</DocLink>).
      </p>

      <H2 id="airrflow">nf-core/airrflow</H2>
      <p>
        airrflow takes AlignAIR's table in its assembled mode. By default it re-assigns genes in an AIRR table it is
        given, which would replace AlignAIR's calls: run it with <code>--reassign false</code>.
      </p>
      <CodeBlock
        code={`nextflow run nf-core/airrflow -r <version> -profile docker \\
  --mode assembled --input samplesheet.tsv --outdir results --reassign false`}
      />
      <p>
        Its samplesheet needs more columns than the file name (sample, subject, species, locus and others): take the
        list from{" "}
        <a href="https://nf-co.re/airrflow/usage" target="_blank" rel="noreferrer">airrflow's usage docs</a> for the
        version you run. airrflow then builds germlines and clones with Change-O and SCOPer from its own germline
        reference: give it AlignAIR's table as it is (CreateGermlines reads it,{" "}
        <DocLink to="from-immcantation?h=create-germlines">above</DocLink>), and a reference whose allele names match
        the table's (OGRDB labels for the IG models).{" "}
        <a href="https://nf-co.re/airrflow/parameters" target="_blank" rel="noreferrer">Its parameter docs</a> say how
        to give it one.
      </p>

      <H2 id="next">Next</H2>
      <ul>
        <li>
          <DocLink to="donor-genotype">Align with a donor genotype</DocLink>: genotype files, inference and refusals.
        </li>
        <li>
          <DocLink to="own-reference">Use your own germline reference</DocLink>: novel alleles and other germline sets.
        </li>
        <li>
          <DocLink to="trust-the-calls">How far to trust the calls</DocLink>: the known failure modes.
        </li>
      </ul>
    </>
  ),
};

export const comingFromPages: DocPage[] = [fromIgblast, fromMixcr, fromImmcantation];
