# Amino-Acid AlignAIR — Phase 0 Diagnostic Report

- Dataconfig: `HUMAN_IGH_OGRDB`  |  simulated records: 200000  |  seed: 0
- Germline DB hash: `bf12d1517d0f2a29`

## Identifiability (go/no-go gate)
- V: 149/198 protein classes; uniquely identifiable fraction = 0.5808080808080808
- J: 7/7 protein classes; uniquely identifiable fraction = 1.0

## Sequence & residue audit
- AA length p50/p95: 126/133 over 28152 records
- unknown-residue fraction: 0.0  |  stop fraction: 0.004225131105781523

## Boundary & D-segment audit
- V-end boundary cuts a codon in 0.8108127308894573 of reads; J-start in 0.6837169650468883
- mean fully-owned D residues: 5.617362887183859; reads with 0 owned D residues: 0.0028061949417448138
