// AlignAIR — Nextflow wrapper DRAFT (not yet stable; see ../README.md).
// One process per sample (Nextflow handles parallelism across cores/nodes). The model
// reloads per process — for a single-node cohort, `alignair batch` (one model load) is faster.
nextflow.enable.dsl = 2

params.samplesheet = "${projectDir}/../samplesheet.csv"   // columns: sample_id, input [, genotype]
params.model       = null                                  // bundle dir / .pt / catalog id / HF repo id
params.genotype    = null                                  // optional reference applied to all samples
params.outdir      = "results"
params.device      = "cpu"

process ALIGNAIR_PREDICT {
    tag "${sample_id}"
    container "thomask90/alignair:latest"
    publishDir "${params.outdir}", mode: "copy"
    cpus 4

    input:
    tuple val(sample_id), path(reads)

    output:
    tuple val(sample_id), path("${sample_id}.tsv"),          emit: airr
    path "${sample_id}.tsv.run.json",        optional: true, emit: provenance

    script:
    def geno = params.genotype ? "--genotype ${params.genotype}" : ""
    """
    alignair predict ${reads} -o ${sample_id}.tsv \\
        --model ${params.model} ${geno} --device ${params.device} --quiet
    """
}

workflow {
    if( !params.model )
        error "Set --model (a bundle dir / .pt checkpoint / catalog id / HF repo id)"

    Channel.fromPath(params.samplesheet)
        | splitCsv(header: true)
        | map { row -> tuple(row.sample_id, file(row.input)) }
        | ALIGNAIR_PREDICT

    // Per-sample genotype/metadata staging is a known rough edge — see ../README.md.
}
