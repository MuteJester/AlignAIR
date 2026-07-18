import { CodeBlock, Callout } from "../../components/ui";
import { DocTable, DocLink, type DocPage } from "./doc-kit";

const workedExample: DocPage = {
  slug: "worked-example",
  title: "Worked example: real repertoire data",
  section: "Get started",
  lead: "A full run on a real, public human IGH repertoire - download, merge, predict, inspect, validate, summarise. Every number here is captured from an actual run.",
  body: () => (
    <>
      <p>
        This takes a real public dataset from raw reads to a validated AIRR table, and reads the output
        honestly - including where it is imperfect. The runnable version, with the bundled merge script, is in{" "}
        <a href="https://github.com/MuteJester/AlignAIR/tree/main/examples/end-to-end" target="_blank" rel="noreferrer">
          examples/end-to-end
        </a>{" "}
        and its commands are checked against the real CLI in CI.
      </p>

      <h2>The dataset</h2>
      <DocTable
        head={["", ""]}
        rows={[
          ["Study", <>Briney et al. 2019, <em>Nature</em> 566:393-397, &ldquo;Commonality despite exceptional diversity in the baseline human antibody repertoire&rdquo;</>],
          ["Accession", <>ENA/SRA run <code>SRR8283604</code>, BioProject <code>PRJNA406949</code></>],
          ["Material", "Human IGH (heavy chain) repertoire, Illumina paired-end 2x250"],
          ["Data access", "INSDC record; under INSDC's data-access policy the sequence data is free to reuse without restriction (a data-access policy, not a software license)"],
          ["Model", <code>alignair-igh-human@1.0.0</code>],
        ]}
      />
      <p>We use a small, reproducible subset: the first 2000 read pairs.</p>

      <h2>1. Download a subset</h2>
      <p>Stream the first 2000 reads of each mate (no full 1.2 GB download - <code>head</code> stops the transfer early):</p>
      <CodeBlock
        code={`mkdir -p aa_example && cd aa_example\nB="https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR828/004/SRR8283604"\ncurl -sSL "$B/SRR8283604_1.fastq.gz" | zcat | head -8000 > R1.sub.fastq\ncurl -sSL "$B/SRR8283604_2.fastq.gz" | zcat | head -8000 > R2.sub.fastq\nmd5sum R1.sub.fastq R2.sub.fastq`}
      />
      <p>The exact subset has a stable checksum, so anyone reproduces the same bytes:</p>
      <CodeBlock lang="text" code={`f855e9a9488ac446f82d5e9fd8413bbe  R1.sub.fastq\n7cb75441bb2bf3821bd84c41656f00c7  R2.sub.fastq`} />
      <Callout kind="note" title="Which checksum is which">
        These md5s cover the <strong>extracted 2000-read subset</strong>, computed here - not ENA/SRA's published
        checksums, which validate the complete <code>SRR8283604_1/2.fastq.gz</code> source files (~1.2 GB each). Verify
        the published hashes only if you download the full files.
      </Callout>
      <p>
        If your network blocks the ENA HTTPS host, use the FTP URL instead (<code>ftp://ftp.sra.ebi.ac.uk/vol1/...</code>);
        the subset and its checksums are identical.
      </p>

      <h2>2. Merge the read pairs</h2>
      <p>
        The reads are 2x250, so a full VDJ amplicon (~350-450 nt) spans both mates. Merge with the bundled
        zero-dependency merger:
      </p>
      <CodeBlock code={`python3 merge_pairs.py R1.sub.fastq R2.sub.fastq merged.fastq\nmd5sum merged.fastq`} />
      <CodeBlock lang="text" code={`pairs=2000 merged=1945          # 97% of pairs overlap and merge\n06c7767cbe5718367d84d2fb624353c6  merged.fastq`} />
      <p>The merged reads span the whole rearrangement (the longest reach ~444 nt).</p>
      <Callout kind="note" title="The bundled merger is demonstration preprocessing, not a pipeline">
        <code>merge_pairs.py</code> is deterministic: it takes the largest R1 / reverse-complement(R2) overlap of at
        least 20 nt with at most 10% mismatch, concatenates on that overlap, and <strong>drops pairs that do not
        merge</strong> (55 of 2000 here). It does no quality-aware consensus and no UMI correction - it exists so this
        example runs with nothing but Python and produces a stable checksum. For real repertoire work use pRESTO
        AssemblePairs, <code>fastp --merge</code>, or vsearch, with UMI-based consensus.
      </Callout>

      <h2>3. Predict</h2>
      <CodeBlock code={`alignair predict --input merged.fastq --out briney_merged.tsv --model alignair-igh-human@1.0.0`} />
      <CodeBlock lang="text" code={`aligned 1945 reads (0 dropped) -> briney_merged.tsv; 0 failed / 43 partial AIRR assemblies tagged`} />
      <Callout kind="tip" title="Orientation was handled for you">
        1937 of 1945 reads (99.6%) come out <code>rev_comp=T</code> - this library's reads are antisense, and
        AlignAIR's orientation head detected and re-framed every one, with no primer trimming and no orientation
        flag. The calls use the model's embedded OGRDB nomenclature (<code>IGHVF10-G41*03</code>), not classic IMGT
        names.
      </Callout>

      <h2>4. Inspect the failures</h2>
      <p>43 of 1945 records assembled only partially:</p>
      <CodeBlock lang="text" code={`assembly_status:       complete 1902, partial 43\nairr_assembly_reason:  nonproductive_indel 42, collapsed_segment 1`} />
      <p>
        These are not dropped - they keep their V/D/J calls, they just could not have a junction assembled (an indel
        broke the reading frame). Filter to <code>airr_assembly_status == complete</code> before junction or clonotype
        analysis. See <DocLink to="known-failure-modes">Known failure modes</DocLink>.
      </p>

      <h2>5. Validate the AIRR output</h2>
      <CodeBlock code={`alignair validate-airr briney_merged.tsv`} />
      <CodeBlock lang="text" code={`OK: briney_merged.tsv - 1945 rows, no coordinate/CIGAR violations`} />
      <p>
        A structural check (columns, coordinate bounds, CIGAR consumption). For formal AIRR-schema validation, also run{" "}
        <code>airr.validate_rearrangement("briney_merged.tsv")</code>.
      </p>

      <h2>6. Summarise</h2>
      <p>Real CDR3s come straight out for the productive rearrangements:</p>
      <CodeBlock
        lang="text"
        code={`IGHVF3-G6*01    IGHD6-13*01  IGHJ5*02  CAKNRVAAAGTMFATW\nIGHVF10-G41*02  IGHD4-11*01  IGHJ5*02  CAKDSTLRLRYNWFDPW`}
      />
      <p>
        Top V genes in the subset: <code>IGHVF10-G41*03</code> (256), <code>IGHVF3-G5*01</code> (215),{" "}
        <code>IGHVF6-G24*04</code> (113).
      </p>
      <Callout kind="note" title="Why only 687 of 1902 complete records are productive">
        That fraction is lower than a polished repertoire study reports, and honestly so: these are <strong>raw reads</strong>{" "}
        with no UMI-based error correction, merged with a demo-grade tool, so uncorrected sequencing errors show up as
        frameshifts. AlignAIR reports them as non-productive rather than hiding them - which is the point of the
        productivity flag. For functional analysis, apply UMI consensus and quality control upstream, then filter to{" "}
        <code>productive == T</code>.
      </Callout>

      <h2>What this shows</h2>
      <ul>
        <li>Real public reads, no manual pre-orientation, no reference to configure: one command aligns them.</li>
        <li>The honesty signals (<code>rev_comp</code>, <code>airr_assembly_status</code>, <code>productive</code>) tell you what to trust.</li>
        <li>The output is standard AIRR TSV, ready for the tools on the <DocLink to="integrations">Integrations</DocLink> page.</li>
      </ul>
    </>
  ),
};

export const tutorialPages: DocPage[] = [workedExample];
