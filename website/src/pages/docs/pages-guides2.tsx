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
          [<code>demo</code>, "offline end-to-end trial (train, predict, predict under a donor genotype)"],
          [<code>doctor</code>, "check the environment (Python, PyTorch, GenAIRR, cache and config dirs)"],
          [<code>info</code>, "print an .alignair model file's card and metadata"],
          [<code>models</code>, "list / download / verify / manage catalog models in the cache (alias model)"],
          [<code>reference</code>, "list built-in GenAIRR references, or export a model's reference"],
          [<code>export-reference</code>, "export the germline FASTA (or, from a training checkpoint, the dataconfig) from a model file"],
          [<code>convert</code>, "convert a lab chip model file (lab-chip-v1, lab-chipw64-v1) into a safe, pickle-free chip-v1 .alignair"],
          [<code>validate-airr</code>, "structural check of a rearrangement TSV (columns, coordinate/CIGAR bounds, productivity invariants)"],
          [<code>compare</code>, "agreement (concordance) report between two AIRR TSVs"],
          [<code>analyze</code>, "summarize composition, prediction QC, and validation of a TSV"],
          [<code>benchmark</code>, "evaluate a model on freshly-generated GenAIRR reads (self-check)"],
          [<code>genotype</code>, "a donor's V genotype from an AIRR table, by the rule of predict --infer-genotype"],
        ]}
      />

      <h2 className="mt-8">alignair predict</h2>
{/* PENDING-V4-PUBLISH: alignair4-igh-human is not published yet. */}
      <CodeBlock code={`alignair predict --input reads.fasta --out out.tsv --model alignair4-igh-human`} />
      <ul className="list-disc pl-6 space-y-1.5 my-4">
        <li><code>--input</code> FASTA / FASTQ / CSV / TSV / TXT (<code>.gz</code> ok; <code>-</code> reads stdin). A table's sequence column is found by name (<code>sequence</code>, <code>seq</code>, <code>nucleotide</code>, <code>read</code>); otherwise pass <code>--sequence-column</code> / <code>--id-column</code>.</li>
        <li><code>--model</code> a catalog id (pin with <code>id@version</code>), a local <code>.alignair</code>, or a Hugging Face repo (<code>hf://org/repo</code> or <code>org/repo</code>; <code>--revision</code>, <code>--hf-token</code>). <code>--registry URL</code> (repeatable) and <code>--offline</code> control catalog lookup. <code>alignair4-igh-human</code> is a placeholder: the AlignAIR 4 IGH model's id is not published yet.</li>
        <li><code>--genotype FILE</code> a YAML/JSON genotype keyed <code>V</code>/<code>D</code>/<code>J</code>: the bank the calls come from, a subset of the bound reference.</li>
        <li><code>--reference FASTA</code> germline V, D and J alleles to call from instead of the embedded reference, novel alleles included (see the <DocLink to="model-contract">model contract</DocLink>).</li>
        <li><code>--infer-genotype</code> infer the donor's V genotype from the reads and re-call the reads it settles (below).</li>
        <li><code>--locus LOCUS</code> the locus for records the model cannot attribute (default: the model's one locus, else IGH).</li>
        <li><code>--metadata FILE</code> join a per-read side table by id; <code>--keep-columns a,b,c</code> selects columns.</li>
        <li><code>--columns full|core|minimal|airr</code> output field set, or a comma-separated field list. <code>core</code> is a compact 27-field row (still assembled); <code>minimal</code> is calls + <code>productive</code> only and is the one preset that skips the AIRR assembly.</li>
        <li><code>--chunk-size N</code> stream in chunks (bounded memory; default 20000). <code>--batch-size N</code> reads per batch (default 256 on a GPU, 64 on the CPU, 16 per worker with <code>--threads</code>). <code>--threads N</code> on the CPU runs N single-threaded worker processes.</li>
        <li><code>--device cpu|cuda|mps|auto</code>. Without it, CUDA when available, else the CPU; <code>auto</code> picks CUDA, then MPS, then CPU.</li>
        <li><code>--rejects-out FILE</code> write dropped/invalid input records. <code>--no-run-metadata</code> skips the <code>&lt;out&gt;.run.json</code> sidecar; <code>--quiet</code> the update notice.</li>
      </ul>

      <h3>Input handling</h3>
      <p>Every read is validated before alignment:</p>
      <ul className="list-disc pl-6 space-y-1.5 my-4">
        <li>Sequences are uppercased (alignment is case-insensitive).</li>
        <li>IUPAC ambiguity codes are mapped to <code>N</code>.</li>
        <li>A read that is empty, or more than 20% non-ACGTN, is dropped and recorded in <code>--rejects-out</code>. The reader never silently truncates.</li>
        <li>A read longer than the model window (575 nt: a 576-position input less one reserved slot) is cropped and flagged <code>length_cropped=T</code>.</li>
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

      <h2>Donor genotype: predict --infer-genotype</h2>
      <CodeBlock code={`alignair predict --input reads.fasta --out out.tsv --model alignair4-igh-human --infer-genotype\n# -> out.tsv, out.tsv.genotype.json (AIRR GenotypeSet), out.tsv.run.json (with a genotype_pass record)`} />
      <p>
        The input is read twice, so it must be a file (not <code>-</code>), and all of its reads should come from one
        donor:
      </p>
      <ol className="list-decimal pl-6 space-y-1.5 my-4">
        <li>Pass 1 aligns every read of the file. For each V gene (the name before <code>*</code>), it counts the reads whose <code>v_call</code> is a single allele.</li>
        <li>Each gene keeps its most frequent alleles until they explain <code>--genotype-explain</code> of those reads (default <code>0.875</code>, TIgGER's). A gene with fewer than <code>--genotype-min-reads</code> such reads (default <code>10</code>) has no evidence: it keeps every allele and is left out of the genotype file.</li>
        <li>Pass 2 reads the file again and re-calls only the reads whose call is outside the genotype while their <code>v_call</code> lists an allele it allows, with the genotype's V alleles (and every allele of the open genes) as the bank. D and J are unchanged. Their rows replace pass 1's, and the run's assembly counts are corrected.</li>
      </ol>
      <p>
        The genotype is written to <code>--genotype-out</code> (default <code>&lt;out&gt;.genotype.json</code>). It does not
        genotype D or J, find novel alleles, mark deleted genes, or separate donors, and it never changes a read whose V
        call is a single allele, nor a tie whose listed alleles all lie outside the genotype. It cannot be combined with{" "}
        <code>--genotype</code>.
      </p>
      <h3>alignair genotype</h3>
      <p>The same rule on any tool's AIRR table (a <code>v_call</code> listing several alleles is a tie and counts for none):</p>
      <CodeBlock code={`alignair genotype --airr out.tsv --out genotype.json --locus IGH --explain 0.875 --min-reads 10`} />
      <p>It prints each gene's alleles (or <code>open</code>, with its read count) and writes the AIRR GenotypeSet.</p>

      <h2>alignair benchmark</h2>
      <p>Evaluate call accuracy, coord MAE, and junction match rates against generated GenAIRR ground truth:</p>
      <CodeBlock code={`alignair benchmark --model alignair4-igh-human --n 200 --seed 0 --format json --out benchmark.json`} />
      <ul className="list-disc pl-6 space-y-1.5 my-4">
        <li><code>--model</code> model to evaluate (id or path).</li>
        <li><code>--dataconfig</code> GenAIRR dataconfig(s) to generate from (default: the model card's).</li>
        <li><code>--n</code> (default <code>200</code>) reads to generate per stratum.</li>
        <li><code>--strata</code> comma-separated subset of <code>clean</code>, <code>moderate</code>, <code>high_shm</code>, <code>short_janchor</code>.</li>
        <li><code>--format text|json</code> output layout format (default <code>text</code>).</li>
      </ul>

      <h2>alignair analyze</h2>
      <p>Summarize an AIRR TSV, yielding composition counts, quality checks, and validation flags:</p>
      <CodeBlock code={`alignair analyze out.tsv --format text`} />

      <h2>alignair info</h2>
      <p>Print a model file's card details (params count, creation timestamp, training hyperparameters, embedded references) without loading weights into memory:</p>
      <CodeBlock code={`# 'models info' resolves a catalog id:\nalignair models info alignair4-igh-human\n\n# 'info' reads a model FILE directly (it does not resolve ids):\nalignair info runs/my_igh/bundle/model.alignair\nalignair info runs/my_igh/bundle/model.alignair --json    # the raw metadata`} />
      <p>
        Its <code>sections</code> line tells a checkpoint from a shareable file: a resumable training checkpoint lists{" "}
        <code>dataconfig/0</code> and <code>train_state</code>; a bundle's model file lists only <code>config</code>,{" "}
        <code>weights</code>, <code>reference</code> and <code>reference_json</code>.
      </p>

      <h2>alignair models</h2>
      <p>List, download, verify, or prune cache elements:</p>
      <CodeBlock code={`alignair models list                             # list live models and status\nalignair models get alignair4-igh-human               # download model to cache\nalignair models path alignair4-igh-human              # print local path in cache\nalignair models info alignair4-igh-human              # print cached card\nalignair models verify                           # verify cache SHA-256 hashes\nalignair models prune --keep 1                   # clean old version directories`} />

      <h2>alignair reference &amp; export-reference</h2>
      <p>List built-in dataconfigs or dump reference sequences from a model:</p>
      <CodeBlock code={`alignair reference list --species Human\n\n# 'reference export' takes a model FILE, not a catalog id - resolve the id to a path first:\nalignair models get alignair4-igh-human\nMODEL="$(alignair models path alignair4-igh-human)"\nalignair reference export "$MODEL" --fasta ref.fasta\n\n# export-reference is the same command, standalone:\nalignair export-reference "$MODEL" --fasta ref.fasta`} />
      <p>
        The exported FASTA is the starting point for <code>--reference</code>: add novel alleles to it and bind it at
        predict time. <code>--dataconfig</code> exports the pickled GenAIRR DataConfig, which only a training checkpoint
        holds; a bundle's model file is refused (<code>section 'dataconfig/0' not in model file</code>).
      </p>

      <h2>Other utility commands</h2>
      <CodeBlock code={`alignair demo                                        # offline end-to-end sandbox\nalignair convert lab_chip.alignair model.alignair    # lab-chip-v1 / lab-chipw64-v1 -> chip-v1 only\nalignair validate-airr out.tsv                       # structural check (not the official airr validator)\nalignair compare --a a.tsv --b b.tsv --a-name A --b-name B --out report.md`} />
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
        <code>alignair.genotype</code> helpers and <code>alignair.errors</code> diagnostics.
      </p>

      <h2>Streaming large inputs</h2>
      <p>For repertoire-scale files, stream input and write AIRR incrementally so memory stays bounded:</p>
      <CodeBlock
        lang="python"
        code={`from alignair import Aligner, iter_sequences, AirrWriter\n\naligner = Aligner.from_pretrained("model.alignair")\nwith AirrWriter("out.tsv", locus=aligner.default_locus) as writer:\n    for ids, seqs, n_dropped in iter_sequences("big.fastq.gz", chunk_size=20000):\n        preds = aligner.predict(seqs).to_dicts()\n        writer.write(ids, seqs, preds)`}
      />
      <p>
        <code>AirrWriter.write</code> raises <code>PredictionCountMismatch</code> if the id / sequence / prediction lists
        differ in length, rather than silently dropping reads. <code>aligner.predict_iter(sequences, chunk_size=20000)</code>{" "}
        yields per-read records the same way.
      </p>

      <h2>Genotypes and other references</h2>
      <CodeBlock
        lang="python"
        code={`from alignair.reference import ReferenceSet\n\n# a donor genotype for one call: {segment: allele names of the bound reference}\nresult = aligner.predict(reads, genotype={"V": ["IGHVF1-G3*02", "IGHVF6-G21*07"]})\n\n# another reference (novel alleles included), bound once to a copy\nnovel = aligner.bound_to(ReferenceSet.from_fasta("germline.fasta"))\nresult = novel.predict(reads)\n\n# or at load\naligner = Aligner.from_pretrained("model.alignair", reference=ReferenceSet.from_fasta("germline.fasta"))`}
      />
      <p>
        <code>bound_to</code> annotates the reference against the model's own (gaps and anchors) and returns a new{" "}
        <code>Aligner</code>; the original is unchanged. What a swap accepts and refuses, with each error, is on the{" "}
        <DocLink to="model-contract">model contract</DocLink> page.
      </p>

      <h2>Loading: device, offline, private repos</h2>
      <CodeBlock
        lang="python"
        code={`Aligner.from_pretrained("alignair4-igh-human", device="cuda")        # cpu | cuda | mps | auto (default)\nAligner.from_pretrained("alignair4-igh-human", offline=True)         # cache only\nAligner.from_pretrained("hf://org/private-model", token="hf_...")\nAligner.from_pretrained("hf://org/repo", revision="v1.0")       # a branch, tag or commit\nAligner.from_pretrained("alignair4-igh-human@<version>")             # pin a catalog version`}
      />
      <p>
        <code>alignair4-igh-human</code> is the AlignAIR 4 IGH model's catalog id; it is not published yet.{" "}
        {/* PENDING-V4-PUBLISH: alignair4-igh-human is not published yet. */}
        <code>device="auto"</code> picks CUDA, then Apple MPS, then CPU; an unavailable <code>cuda</code> or{" "}
        <code>mps</code> falls back to the CPU. <code>offline=True</code> uses only what is cached and raises if the
        model is not present locally.
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
      <p>
        Catch <code>AlignAIRError</code> for classified failures, or a domain exception such as
        <code> RegistryError</code> or <code>ModelError</code> when recovery depends on the cause.
        See <DocLink to="errors">Errors and diagnostics API</DocLink> for exception types, structured diagnostics,
        and reporting helpers.
      </p>
      <ul>
        <li><code>ModelError</code> from <code>from_pretrained</code> on a missing path or an unknown id (<code>RegistryError</code> when a registry or the Hub cannot be reached), and on an AlignAIR 3 model.</li>
        <li><code>ModelIntegrityError</code> when the embedded reference does not match the model card's fingerprints.</li>
        <li><code>ConfigurationError</code> / <code>InputError</code> when a reference or genotype cannot be bound (see the <DocLink to="model-contract">model contract</DocLink>).</li>
        <li><code>DuplicateMetadataId</code> when a metadata join table has a duplicate key.</li>
        <li><code>PredictionCountMismatch</code> when writing with mismatched counts.</li>
      </ul>

      <h2>Training from Python</h2>
      <CodeBlock
        lang="python"
        code={`from alignair import train_model, Aligner\nimport GenAIRR.data as gd\nimport os\n\nos.makedirs("runs/igh", exist_ok=True)\npath = train_model([gd.HUMAN_IGH_OGRDB], out_path="runs/igh/model.alignair", steps=50_000, device="cuda")\naligner = Aligner.from_pretrained(path)`}
      />
      <p>
        <code>train_model</code> writes one resumable checkpoint file: it holds pickled training sections, so{" "}
        <code>save_pretrained</code> and <code>push_to_hub</code> refuse it. The typed entry is{" "}
        <code>run_training</code>; neither exports a bundle, so export one for sharing with{" "}
        <code>export_bundle</code> (as <code>alignair train</code> does):
      </p>
      <CodeBlock
        lang="python"
        code={`from alignair import TrainingConfig, run_training\nfrom alignair.genairr import resolve_dataconfigs\nfrom alignair.train.build import export_bundle\n\nrun = run_training(TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", preset="desktop"), output_dir="runs/igh")\naligner = run.best_aligner()          # the best (or last) checkpoint, reloaded\nexport_bundle(run.best_model_path or run.model_path, resolve_dataconfigs(["HUMAN_IGH_OGRDB"]), "runs/igh/bundle")`}
      />

      <h2>API stability</h2>
      <p>
        The stable surface is the top-level <code>alignair</code> namespace, <code>alignair.genotype</code> helpers,
        and the documented <code>alignair.errors</code> API;
        a release line keeps these signatures backward-compatible. Submodules may change in any release.
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
{/* PENDING-V4-PUBLISH: alignair4-igh-human is not published yet. */}
      <CodeBlock code={`alignair predict --input contigs.fasta --out out.tsv --model alignair4-igh-human \\\n  --metadata filtered_contig_annotations.csv --keep-columns barcode,umis,c_gene`} />
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
      <CodeBlock code={`# 1. keep only the fully-assembled records (see the rule above)\npython -c "import pandas as pd; t=pd.read_csv('out.tsv',sep='\\t'); \\\n  t[t.airr_assembly_status=='complete'].to_csv('out.complete.tsv',sep='\\t',index=False)"\n\n# 2. export the exact germline the model used, so germline reconstruction matches its calls\nMODEL="$(alignair models path alignair4-igh-human)"\nalignair reference export "$MODEL" --fasta germline.fasta\n\n# 3. hand off to Change-O\nDefineClones.py -d out.complete.tsv --act set --model ham --norm len --dist 0.16\nCreateGermlines.py -d out.complete_clone-pass.tsv -r germline.fasta`} />

      <h2>IgBLAST (for comparison)</h2>
      <CodeBlock code={`# give IgBLAST the exact germline the model uses (resolve the id to a file first)\nMODEL="$(alignair models path alignair4-igh-human)"\nalignair reference export "$MODEL" --fasta germline.fasta\n\nigblastn -germline_db_V ... -germline_db_D ... -germline_db_J ... \\\n  -outfmt 19 -query reads.fasta > igblast.tsv\nalignair compare --a out.tsv --b igblast.tsv --a-name AlignAIR --b-name IgBLAST --out report.md`} />

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
        code={`process alignair {\n  container 'alignair/alignair:3.0.0'\n  input:  path reads\n  output: path 'out.tsv'\n  script: 'alignair predict --input $reads --out out.tsv --model alignair4-igh-human@<version>'\n}`}
      />
      <p>Pre-cache the model and mount the cache (or run with <code>--offline</code>) so the pipeline never fetches at run time.</p>
    </>
  ),
};

export const guides2Pages: DocPage[] = [cli, pythonApi, integrations];
