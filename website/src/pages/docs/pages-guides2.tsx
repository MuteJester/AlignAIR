import { CodeBlock, Callout } from "../../components/ui";
import { DocTable, DocLink, type DocPage } from "./doc-kit";

const cli: DocPage = {
  slug: "cli",
  title: "Command-line reference",
  section: "Using AlignAIR",
  lead: "Every command supports alignair <command> --help. The command-line surface is the versioned contract of AlignAIR.",
  body: () => (
    <>
      <DocTable
        head={["Command", "Purpose"]}
        rows={[
          [<code>predict</code>, "align reads into an AIRR rearrangement TSV"],
          [<code>train</code>, "train a model on a built-in dataconfig or custom FASTAs"],
          [<code>demo</code>, "offline end-to-end trial (train, predict, validate, genotype)"],
          [<code>doctor</code>, "check the environment (Python, PyTorch, GenAIRR, parasail)"],
          [<code>info</code>, "print an .alignair model file's card and metadata"],
          [<code>models</code>, "list / download / verify / manage pretrained models in the cache"],
          [<code>reference</code>, "list built-in GenAIRR references, or export a model's reference"],
          [<code>export-reference</code>, "export the germline FASTA / dataconfig from a model file"],
          [<code>convert</code>, "package a legacy .pt checkpoint into a safe, pickle-free .alignair"],
          [<code>validate-airr</code>, "structural check of a rearrangement TSV (columns, coordinate/CIGAR bounds, productivity invariants)"],
          [<code>compare</code>, "agreement (concordance) report between two AIRR TSVs"],
          [<code>analyze</code>, "summarize composition, prediction QC, and validation of a TSV"],
          [<code>benchmark</code>, "evaluate a model on freshly-generated GenAIRR reads (self-check)"],
          [<code>genotype</code>, "infer an individual's IG genotype from repertoire reads (experimental)"],
        ]}
      />

      <h2 className="mt-8">alignair predict</h2>
      <CodeBlock code={`alignair predict --input reads.fasta --out out.tsv --model alignair-igh-human`} />
      <ul className="list-disc pl-6 space-y-1.5 my-4">
        <li><code>--input</code> FASTA / FASTQ / CSV / TSV (<code>.gz</code> ok; <code>-</code> reads stdin). For a table, pass <code>--sequence-column</code> / <code>--id-column</code>.</li>
        <li><code>--model</code> a pretrained id, local <code>.alignair</code>/<code>.pt</code>, or <code>org/name</code> HF repo id. Pin with <code>id@version</code>.</li>
        <li><code>--genotype FILE</code> constrain to a donor subset (<code>--genotype-method mask|softmax|renormalize|redistribute</code>).</li>
        <li><code>--metadata FILE</code> join a per-read side table by id; <code>--keep-columns a,b,c</code> selects columns.</li>
        <li><code>--columns full|core|minimal|airr</code> output field set. <code>core</code> is a compact 27-field row (still assembled); <code>minimal</code> is calls + <code>productive</code> only and is the one preset that skips the AIRR assembly.</li>
        <li><code>--chunk-size N</code> stream in chunks (bounded memory; default 20000). <code>--device cpu|cuda|mps</code> (auto picks CUDA, then MPS, then CPU).</li>
        <li><code>--rejects-out FILE</code> write dropped/invalid input records.</li>
      </ul>

      <h3>Input handling</h3>
      <p>Every read is validated before alignment:</p>
      <ul className="list-disc pl-6 space-y-1.5 my-4">
        <li>Sequences are uppercased (alignment is case-insensitive).</li>
        <li>IUPAC ambiguity codes are mapped to <code>N</code>.</li>
        <li>A read that is empty, or more than 20% non-ACGTN, is dropped and recorded in <code>--rejects-out</code>. The reader never silently truncates.</li>
        <li>A read longer than the model window (576 nt) is cropped and flagged <code>length_cropped=T</code>.</li>
        <li>Duplicate read ids are de-duplicated. A <code>--metadata</code> join requires unique ids.</li>
      </ul>
      <p>AlignAIR aligns single reads: it does not assemble paired R1/R2 reads and does not read BAM/CRAM. FASTQ quality strings are read but not used for alignment.</p>

      <h3>Quality gates and exit status</h3>
      <ul className="list-disc pl-6 space-y-1.5 my-4">
        <li><code>--max-assembly-failures FRAC</code> (default <code>0.01</code>): exit non-zero if the failed rate exceeds <code>FRAC</code>.</li>
        <li><code>--max-partial-assemblies FRAC</code> (default <code>1.0</code>, never): exit non-zero if the partial rate exceeds <code>FRAC</code>.</li>
        <li><code>--permissive</code>: never fail on these rates.</li>
      </ul>
      <p>On a gate breach the output is still written with every row tagged; only the exit status signals the problem.</p>

      <h2>alignair train</h2>
      <CodeBlock code={`alignair train --dataconfig HUMAN_IGH_OGRDB --out runs/my_igh --preset desktop`} />
      <p>See <DocLink to="training">Training a custom model</DocLink> for the full guide.</p>

      <h2>alignair genotype (experimental)</h2>
      <p>Infer an individual's immunoglobulin genotype from their repertoire reads:</p>
      <CodeBlock code={`alignair genotype repertoire.fasta --model alignair-igh-human --out genotype_output`} />
      <ul className="list-disc pl-6 space-y-1.5 my-4">
        <li><code>input</code> positional argument containing the read sequences.</li>
        <li><code>--model</code> the model path or shipped registry id.</li>
        <li><code>--min-support FRAC</code> (default <code>0.003</code>) minimum fraction of reads supporting an allele to retain it.</li>
        <li><code>--locus LOCUS</code> (default <code>IGH</code>) target locus to genotype.</li>
        <li><code>--germline-set-ref CURIE</code> optional curated GermlineSet reference.</li>
      </ul>

      <h2>alignair benchmark</h2>
      <p>Evaluate call accuracy, coord MAE, and junction match rates against generated GenAIRR ground truth:</p>
      <CodeBlock code={`alignair benchmark --model alignair-igh-human --n 200 --seed 0 --out benchmark.json`} />
      <ul className="list-disc pl-6 space-y-1.5 my-4">
        <li><code>--model</code> model to evaluate (id or path).</li>
        <li><code>--n</code> (default <code>200</code>) reads to generate per stratum.</li>
        <li><code>--strata</code> comma-separated list of strata to restrict generation.</li>
        <li><code>--format text|json</code> output layout format.</li>
      </ul>

      <h2>alignair analyze</h2>
      <p>Summarize an AIRR TSV, yielding composition counts, quality checks, and validation flags:</p>
      <CodeBlock code={`alignair analyze out.tsv --format text`} />

      <h2>alignair info</h2>
      <p>Print a model file's card details (params count, creation timestamp, training hyperparameters, embedded references) without loading weights into memory:</p>
      <CodeBlock code={`# 'models info' resolves a catalog id:\nalignair models info alignair-igh-human\n\n# 'info' reads a model FILE directly (it does not resolve ids):\nalignair info runs/my_igh/bundle/model.alignair`} />

      <h2>alignair models</h2>
      <p>List, download, verify, or prune cache elements:</p>
      <CodeBlock code={`alignair models list                             # list live models and status\nalignair models get alignair-igh-human           # download model to cache\nalignair models path alignair-igh-human          # print local path in cache\nalignair models info alignair-igh-human          # print cached card\nalignair models verify                           # verify cache SHA-256 hashes\nalignair models prune --keep 1                   # clean old version directories`} />

      <h2>alignair reference &amp; export-reference</h2>
      <p>List built-in dataconfigs or dump reference sequences from a model:</p>
      <CodeBlock code={`alignair reference list --species Human\n\n# 'reference export' takes a model FILE, not a catalog id - resolve the id to a path first:\nalignair models get alignair-igh-human\nMODEL="$(alignair models path alignair-igh-human)"\nalignair reference export "$MODEL" --fasta ref.fasta\n\n# export-reference is the same command, standalone:\nalignair export-reference "$MODEL" --fasta ref.fasta`} />

      <h2>Other utility commands</h2>
      <CodeBlock code={`alignair demo                                        # offline end-to-end sandbox\nalignair convert model.pt model.alignair --dataconfig HUMAN_IGH_OGRDB --trust-pickle\nalignair validate-airr out.tsv                       # structural check (not the official airr validator)\nalignair compare --a a.tsv --b b.tsv --a-name A --b-name B --out report.md`} />
      <p>
        <code>alignair compare</code> measures concordance between two tools, not accuracy (no ground truth is involved).
      </p>
    </>
  ),
};

const pythonApi: DocPage = {
  slug: "python-api",
  title: "Python API",
  section: "Using AlignAIR",
  lead: "A small, stable Python API. The CLI builds on these functions.",
  body: () => (
    <>
      <CodeBlock
        lang="python"
        code={`from alignair import Aligner, read_sequences\n\naligner = Aligner.from_pretrained("runs/my_model/bundle/model.alignair")\nids, reads, info = read_sequences("reads.fastq")\nprint(f"Loaded {len(reads)} reads (dropped {info['n_dropped']} invalid reads)")\nresult = aligner.predict(reads)\nresult.ids = ids\nresult.write_airr("out.tsv")`}
      />
      <p>
        The stable import surface is the top-level <code>alignair</code> namespace plus the documented{" "}
        <code>alignair.genotype</code> helpers. Everything else is implementation detail.
      </p>

      <h2>Streaming large inputs</h2>
      <p>For repertoire-scale files, stream input and write AIRR incrementally so memory stays bounded:</p>
      <CodeBlock
        lang="python"
        code={`from alignair import Aligner, iter_sequences, AirrWriter\n\naligner = Aligner.from_pretrained("alignair-igh-human")\nwith AirrWriter("out.tsv", locus=aligner.default_locus) as writer:\n    for ids, seqs, n_dropped in iter_sequences("big.fastq.gz", chunk_size=20000):\n        preds = aligner.predict(seqs).to_dicts()\n        writer.write(ids, seqs, preds)`}
      />
      <p>
        <code>AirrWriter.write</code> raises <code>PredictionCountMismatch</code> if the id / sequence / prediction lists
        differ in length, rather than silently dropping reads.
      </p>

      <h2>Loading: device, offline, private repos</h2>
      <CodeBlock
        lang="python"
        code={`Aligner.from_pretrained("alignair-igh-human", device="cuda")   # cpu | cuda | mps | auto\nAligner.from_pretrained("alignair-igh-human", offline=True)    # cache only\nAligner.from_pretrained("hf://org/private-model", token="hf_...")\nAligner.from_pretrained("alignair-igh-human@1.0.0")            # pin a version`}
      />
      <p>
        <code>device="auto"</code> picks CUDA, then Apple MPS, then CPU. <code>offline=True</code> uses only what is
        cached and raises if the model is not present locally.
      </p>

      <h2>Partial and null records</h2>
      <p>
        Not every record is fully assembled. Each carries <code>airr_assembly_status</code> (<code>complete</code> /{" "}
        <code>partial</code> / <code>failed</code>), and a blank field means unknown, not <code>False</code>:
      </p>
      <CodeBlock
        lang="python"
        code={`records = aligner.predict(reads).to_dicts()\nusable = [r for r in records\n          if r.get("airr_assembly_status") == "complete"\n          and not r.get("segmentation_low_quality")]`}
      />
      <p>See <DocLink to="airr-fields">AIRR output fields</DocLink> for the status codes and orientation contract.</p>

      <h2>Errors to handle</h2>
      <ul>
        <li><code>FileNotFoundError</code> / <code>ValueError</code> from <code>from_pretrained</code> on a missing path or unresolvable id.</li>
        <li>A load-time error when a caller-supplied reference does not match the model's embedded, fingerprinted reference.</li>
        <li><code>DuplicateMetadataId</code> when a metadata join table has a duplicate key.</li>
        <li><code>PredictionCountMismatch</code> when writing with mismatched counts.</li>
      </ul>

      <h2>Training from Python</h2>
      <CodeBlock
        lang="python"
        code={`from alignair import train_model, Aligner\nimport GenAIRR.data as gd\nimport os\n\nos.makedirs("runs/igh", exist_ok=True)\npath = train_model([gd.HUMAN_IGH_OGRDB], out_path="runs/igh/model.alignair", steps=50_000, device="cuda")\naligner = Aligner.from_pretrained(path)`}
      />
      <p>
        Note that <code>train_model</code> writes a single resumable checkpoint file directly. The directories in <code>out_path</code> must exist. The returned model contains training states (pickle data) and can be converted to an inference-only, pickle-free model using the <code>alignair convert</code> CLI or by setting <code>include_trusted_pickle=False</code> when exporting.
      </p>

      <h2>API stability</h2>
      <p>
        The stable surface is the top-level <code>alignair</code> namespace plus <code>alignair.genotype</code> helpers;
        the 3.x line keeps these signatures backward-compatible. Submodules may change in any release.
      </p>
    </>
  ),
};

const integrations: DocPage = {
  slug: "integrations",
  title: "Integrations",
  section: "Using AlignAIR",
  lead: "AlignAIR writes standard AIRR, so its output feeds the AIRR ecosystem directly.",
  body: () => (
    <>
      <Callout kind="note">
        What is CI-verified is the AlignAIR side of the contract: the emitted output validates against the official{" "}
        <code>airr</code> schema and carries the columns these importers read. The external tools are not bundled; the
        commands below are templates - pin the tool version you run.
      </Callout>

      <h2>What AlignAIR supplies vs. what you supply</h2>
      <DocTable
        head={["From AlignAIR (predict)", "From you (--metadata)"]}
        rows={[
          [<><code>sequence_id</code>, <code>sequence</code>, <code>rev_comp</code>, <code>locus</code></>, <><code>cell_id</code> / <code>barcode</code> (single-cell grouping)</>],
          [<>V/D/J calls (+ <code>*_call_set</code>)</>, <><code>umi_count</code>, <code>duplicate_count</code>, <code>consensus_count</code></>],
          [<>junction, np1/np2, coordinates, CIGARs, alignments</>, <><code>sample_id</code>, <code>subject_id</code>, study metadata</>],
          [<><code>productive</code> (derived), identities</>, <><code>c_call</code> (constant region, from the assembler)</>],
        ]}
      />
      <p>Two rules apply to every integration:</p>
      <ul>
        <li>Filter to complete records first (<code>airr_assembly_status == "complete"</code>) before clonotype or productivity analysis.</li>
        <li>AlignAIR is per-read; single-cell grouping (<code>cell_id</code>) and counts come from your <code>--metadata</code> table.</li>
      </ul>
      <CodeBlock code={`alignair predict --input contigs.fasta --out out.tsv --model alignair-igh-human \\\n  --metadata filtered_contig_annotations.csv --keep-columns barcode,umis,c_gene`} />
      <Callout kind="note" title="Name the RAW columns, not the normalized ones">
        <code>--keep-columns</code> selects columns as they appear in your metadata file, and naming one that is not
        there is an error. Cell Ranger writes <code>barcode</code> and <code>umis</code>, so ask for those.
        AlignAIR normalizes the well-known 10x names on the way out, so the output carries <code>cell_id</code> and{" "}
        <code>umi_count</code> (and <code>c_call</code> from <code>c_gene</code>) as Scirpy and Change-O expect.
      </Callout>

      <h2>Scirpy (single-cell)</h2>
      <CodeBlock
        lang="python"
        code={`import scirpy as ir, pandas as pd\n\nt = pd.read_csv("out.tsv", sep="\\t")\nt = t[t["airr_assembly_status"] == "complete"]\nt.to_csv("out.complete.tsv", sep="\\t", index=False)\nadata = ir.io.read_airr("out.complete.tsv")   # cell_id supplied via --metadata\nir.tl.chain_qc(adata)`}
      />

      <h2>Change-O / Immcantation</h2>
      <CodeBlock code={`# 1. keep only the fully-assembled records (see the rule above)\npython -c "import pandas as pd; t=pd.read_csv('out.tsv',sep='\\t'); \\\n  t[t.airr_assembly_status=='complete'].to_csv('out.complete.tsv',sep='\\t',index=False)"\n\n# 2. export the exact germline the model used, so germline reconstruction matches its calls\nMODEL="$(alignair models path alignair-igh-human)"\nalignair reference export "$MODEL" --fasta germline.fasta\n\n# 3. hand off to Change-O\nDefineClones.py -d out.complete.tsv --act set --model ham --norm len --dist 0.16\nCreateGermlines.py -d out.complete_clone-pass.tsv -r germline.fasta`} />

      <h2>IgBLAST (for comparison)</h2>
      <CodeBlock code={`# give IgBLAST the exact germline the model uses (resolve the id to a file first)\nMODEL="$(alignair models path alignair-igh-human)"\nalignair reference export "$MODEL" --fasta germline.fasta\n\nigblastn -germline_db_V ... -germline_db_D ... -germline_db_J ... \\\n  -outfmt 19 -query reads.fasta > igblast.tsv\nalignair compare --a out.tsv --b igblast.tsv --a-name AlignAIR --b-name IgBLAST --out report.md`} />

      <h2>nf-core/airrflow (assembled mode)</h2>
      <p>AlignAIR produces the per-sample TSV; you provide the samplesheet:</p>
      <CodeBlock
        lang="text"
        code={`filename\tsample_id\tsubject_id\tspecies\tpcr_target_locus\tsingle_cell\nout.tsv\tsample1\tdonorA\thuman\tIGH\tFALSE`}
      />
      <CodeBlock code={`nextflow run nf-core/airrflow -r <pinned-version> -profile docker \\\n  --mode assembled --input samplesheet.tsv --outdir results`} />

      <h2>Workflow engines</h2>
      <CodeBlock
        lang="groovy"
        code={`process alignair {\n  container 'ghcr.io/mutejester/alignair:3.0.0'\n  input:  path reads\n  output: path 'out.tsv'\n  script: 'alignair predict --input $reads --out out.tsv --model alignair-igh-human@1.0.0'\n}`}
      />
      <p>Pre-cache the model and mount the cache (or run with <code>--offline</code>) so the pipeline never fetches at run time.</p>
    </>
  ),
};

export const guides2Pages: DocPage[] = [cli, pythonApi, integrations];
