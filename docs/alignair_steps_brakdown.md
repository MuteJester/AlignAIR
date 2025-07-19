# AlignAIR Pipeline Steps Documentation

## Table of Contents
1. [Configuration Loading](#config-load-step)
2. [File Processing](#file-processing-steps)
   - [File Name Extraction](#file-name-extraction-step)
   - [Sample Counting](#sample-counter-step)
3. [Model Setup](#model-loading-step)
4. [Batch Processing and Prediction](#batch-processing-step)
5. [Post-Processing Steps](#post-processing-steps)
   - [Raw Prediction Cleanup](#clean-and-arrange-step)
   - [Segmentation Correction](#segment-correction-step)
   - [Threshold Application](#allele-threshold-application-steps)
   - [Germline Alignment](#germline-alignment-step)
   - [IMGT Translation](#translation-step)
   - [Results Finalization](#finalization-step)

# AlignAIR v2.0 Pipeline Steps Documentation

## Table of Contents
1. [Configuration Loading](#config-load-step)
2. [File Processing](#file-processing-steps)
   - [File Name Extraction](#file-name-extraction-step)
   - [Sample Counting](#sample-counter-step)
3. [Model Setup](#model-loading-step)
4. [Batch Processing and Prediction](#batch-processing-step)
5. [Post-Processing Steps](#post-processing-steps)
   - [Raw Prediction Cleanup](#clean-and-arrange-step)
   - [Segmentation Correction](#segment-correction-step)
   - [Threshold Application](#allele-threshold-application-steps)
   - [Germline Alignment](#germline-alignment-step)
   - [IMGT Translation](#translation-step)
   - [Results Finalization](#finalization-step)

## Architecture Overview

AlignAIR v2.0 introduces a unified pipeline that automatically detects single-chain vs. multi-chain scenarios:

- **Single-Chain Mode**: Uses `SingleChainAlignAIR` with `SingleChainDataset`
- **Multi-Chain Mode**: Uses `MultiChainAlignAIR` with `MultiChainDataset` and `MultiDataConfigContainer`
- **Dynamic Detection**: Based on the number of GenAIRR dataconfigs provided

## Pipeline Steps Details
---

### [Config Load Step](#config-load-step)
**Purpose**: Loads GenAIRR DataConfig(s) and determines single vs. multi-chain mode

**Class**: `ConfigLoadStep`

**Input**: GenAIRR dataconfig identifier(s) or path(s) to custom dataconfig pickle files

**Output**: 
- Single-chain: Loaded DataConfig object
- Multi-chain: MultiDataConfigContainer with multiple DataConfigs

**Key Features**:
- Supports built-in GenAIRR dataconfigs: `HUMAN_IGH_OGRDB`, `HUMAN_IGK_OGRDB`, `HUMAN_IGL_OGRDB`, `HUMAN_TCRB_IMGT`
- Handles custom dataconfig pickle files
- Multi-chain support with comma-separated configs
- Automatic chain type detection and D-gene presence validation

**Dependencies**:
- GenAIRR utilities for loading built-in dataconfigs
- MultiDataConfigContainer for multi-chain scenarios
---

### [File Processing Steps](#file-processing-steps)

#### [File Name Extraction Step](#file-name-extraction-step)
**Purpose**: Extracts file name and suffix from the input file path

**Class**: `FileNameExtractionStep`

**Input**: File path from script arguments

**Output**: 
- File name
- File suffix

**Key Features**:
- Inherits from base Step class
- Uses `get_filename` utility function
- Updates PredictObject with extracted information
---

#### [Sample Counter Step](#sample-counter-step)
**Purpose**: Counts total number of sequences/samples in the input file

**Class**: `FileSampleCounterStep`

**Input**: File path and type from PredictObject

**Output**: Total sample count

**Key Features**:
- Supports multiple file formats through `FILE_ROW_COUNTERS` dictionary
- Automatically selects appropriate counting method based on file suffix
- Logs progress and final sample count
- Updates PredictObject with sample count information

**Example Flow**:

1. File name extraction:
    - Input: "data/sequences.fasta"
    - Output: name="sequences", suffix=".fasta"


2. Sample counting:
    - Uses format-specific counter (FASTA/CSV/TSV)
    - Counts total sequences/rows
    - Logs count for monitoring

**Note**: These steps are crucial for:
- File handling setup
- Progress tracking
- Resource allocation
- Batch processing preparation
---

### [Model Loading Step](#model-loading-step)
**Purpose**: Loads and initializes the unified AlignAIR model architecture and related components

**Class**: `ModelLoadingStep`

**Input**: 
- GenAIRR dataconfig(s) (single or multiple)
- Model checkpoint path
- Maximum sequence size
- Configuration data

**Key Components**:

1. **Architecture Detection and Model Loading**
    - Detects single-chain vs. multi-chain mode based on dataconfig count
    - Single-chain: Creates SingleChainDataset + SingleChainAlignAIR
    - Multi-chain: Creates MultiChainDataset + MultiChainAlignAIR with MultiDataConfigContainer
    - Automatically determines model parameters from dataconfig(s)
    - Builds model with appropriate input shape
    - Loads pre-trained weights from checkpoint

2. **Universal Dataset Creation**
    - SingleChainDataset: For single receptor type analysis
    - MultiChainDataset: For mixed receptor analysis with batch partitioning
    - Automatic chain type encoding and batch type management
    - Dynamic allele mapping from GenAIRR dataconfigs

3. **Model Architecture Selection**
    - SingleChainAlignAIR: V/D/J segmentation + allele calling + productivity
    - MultiChainAlignAIR: Above + chain type classification + multi-chain optimization
    - Dynamic D-gene support based on dataconfig metadata
    - Adaptive latent dimensions and output heads

4. **Orientation Pipeline** (Optional)
    - Loads DNA orientation classifier
    - Supports both built-in and custom orientation models
    - Enabled via `fix_orientation` flag

5. **Kmer Density Model**
    - Initializes FastKmerDensityExtractor
    - Processes reference alleles from all dataconfigs
    - Unified handling of V, D, J alleles across chain types
    - Fits extractor with combined reference sequences

**Key Features**:
    - Unified architecture supporting any GenAIRR dataconfig combination
    - Automatic single vs. multi-chain detection
    - Dynamic model parameter inference
    - Built-in chain type management
    - Extensible to custom receptor types

**Output**: PredictObject updated with:
    - Loaded unified model (SingleChainAlignAIR or MultiChainAlignAIR)
    - Appropriate dataset (SingleChainDataset or MultiChainDataset)
    - Orientation pipeline (if enabled)
    - Fitted kmer density extractor
    - Model architecture metadata

**Dependencies**:
    - SingleChainAlignAIR/MultiChainAlignAIR models
    - SingleChainDataset/MultiChainDataset classes
    - MultiDataConfigContainer for multi-chain scenarios
    - FastKmerDensityExtractor
    - GenAIRR utilities for dataconfig loading
---

### [Batch Processing Step](#batch-processing-step)
**Purpose**: Processes sequences in batches through a multi-process pipeline for efficient tokenization and prediction

**Class**: `BatchProcessingStep`

**Key Components**:

1. **Tokenizer Process**
    - Initializes multiprocessing queue for data transfer
    - Creates worker process for sequence reading and tokenization
    - Supports multiple file formats (CSV, TSV, FASTA)
    - Handles DNA sequence tokenization with dictionary mapping:
    - A → 1, T → 2, G → 3, C → 4, N → 5, P → 0 (padding)

2. **Batch Processing Pipeline**
    - Queue-based producer-consumer architecture
    - Maximum queue size of 64 for memory management
    - Processes sequences in configurable batch sizes
    - Applies orientation fixing if enabled

3. **Prediction Processing**
    - Batch-wise model predictions
    - Tracks processing statistics:
        - Batch processing times
        - Queue sizes
        - Estimated remaining time
        - Total processing duration

**Key Features**:
    - Parallel processing with multiprocessing
    - Progress monitoring and logging
    - Memory-efficient batch processing
    - Error handling with proper process cleanup
    - Dynamic batch size configuration
    - Real-time progress updates

**Output**: PredictObject updated with:
    - Model predictions array
    - Processed sequences
    - Processing statistics

**Performance Metrics**:
    - Tracks individual batch times
    - Calculates average processing speed
    - Estimates remaining processing time
    - Reports total processing duration

**Dependencies**:
    - Multiprocessing
    - NumPy
    - Custom worker types for different file formats
---


#### [Raw Prediction Cleanup: Clean and Arrange Step](#clean-and-arrange-step)
**Purpose**: Organizes and structures raw model predictions into a clean, standardized format

**Class**: `CleanAndArrangeStep`

**Key Components**:

1. **Data Collection**
Processes predictions for:
    - Mutation rates
    - V/D/J allele predictions
    - Start/end positions for each segment
    - Indel counts
    - Productivity status
    - Chain type information

2. **Data Processing**
    - Stacks individual predictions using numpy
    - Converts arrays to appropriate formats:
        - Productive status to boolean (>0.5 threshold)
        - Squeezes single-dimensional arrays
        - Maintains multi-dimensional arrays for allele predictions

3. **Chain-Specific Processing**
    - Heavy Chain:
        - Includes D segment information (start, end, allele)
    - Light Chain:
         - Includes chain type information
        - Excludes D segment processing

**Output Structure**:
```python
{
    'v_allele': np.array,      # V gene predictions
    'd_allele': np.array,      # D gene predictions (heavy chain only)
    'j_allele': np.array,      # J gene predictions
    'v_start': np.array,       # V segment start positions
    'v_end': np.array,         # V segment end positions
    'd_start': np.array,       # D segment start positions (heavy chain only)
    'd_end': np.array,         # D segment end positions (heavy chain only)
    'j_start': np.array,       # J segment start positions
    'j_end': np.array,         # J segment end positions
    'mutation_rate': np.array, # Mutation rates
    'indel_count': np.array,   # Indel counts
    'productive': np.array,    # Productivity boolean
    'type_': np.array         # Chain type (light chain only)
}
```
---

#### [Segmentation Correction Step](#segment-correction-step)
**Purpose**: Adjusts segment positions by accounting for padding added during sequence processing

**Class**: `SegmentCorrectionStep`

**Key Components**:

1. **Padding Size Calculation**
    - Calculates padding added to sequences to reach max_length
    - Determines padding size for both sides of sequence
    - Default max length: 576 nucleotides
    - Returns padding size for start of sequence
```python
def calculate_pad_size(self, sequence, max_length=576):
```




2. **Segment Position Correction**
Corrects positions by:
    - Computing padding for each sequence
    - Adjusting start/end positions by subtracting padding
    - Processing segments:
         - V segment (start/end)
         - J segment (start/end)
         - D segment (start/end) for heavy chains only
    - Rounds positions to integers

```python
def correct_segments_for_paddings(self, sequences, chain_type, v_start, v_end, d_start, d_end, j_start, j_end):
```


**Key Features**:
- Handles both heavy and light chain sequences
- Processes batches using NumPy operations
- Maintains segment order and relationships
- Converts predictions to actual sequence positions

**Output**: Dictionary containing corrected positions:
```python
{
    'v_start': int[],  # V segment start positions
    'v_end': int[],    # V segment end positions
    'd_start': int[],  # D segment start positions (heavy chain)
    'd_end': int[],    # D segment end positions (heavy chain)
    'j_start': int[],  # J segment start positions
    'j_end': int[]     # J segment end positions
}
```

**Dependencies**:
- NumPy for array operations
- Base Step class for pipeline integration

---
#### [Allele Threshold Application Steps](#allele-threshold-application-steps)
**Purpose**: Applies thresholds to select the most likely allele predictions using two different methods

**Classes**: 
- `ConfidenceMethodThresholdApplicationStep`
- `MaxLikelihoodPercentageThresholdApplicationStep`

**Key Components**:

1. **Confidence Method Threshold**
    - Uses `CappedDynamicConfidenceThreshold` for selection
    - Applies confidence-based thresholds to predictions
    - Handles both heavy and light chain configurations
    - Returns selected alleles based on confidence scores


2. **Max Likelihood Percentage Threshold**
    - Uses `MaxLikelihoodPercentageThreshold` for selection
    - Selects alleles based on likelihood percentages
    - Applies caps to limit number of predictions
    - Processes V, D (heavy chain only), and J genes

**Common Features**:
- Processes alleles by gene type (V, D, J)
- Applies chain-specific thresholds:

```python
  thresholds = {
      'v': v_allele_threshold,
      'd': d_allele_threshold,
      'j': j_allele_threshold
  }
```
---
#### [Germline Alignment Step](#germline-alignment-step)
**Purpose**: Aligns predicted segments with germline reference sequences to validate and refine predictions

**Class**: `AlleleAlignmentStep`

**Key Components**:

1. **Alignment Process**
    - Takes segment positions, threshold objects, predicted alleles, and sequences
    - Creates alignments for each gene type (V, D, J)
    - Uses `HeuristicReferenceMatcher` for sequence alignment
    - Special handling for D segments with "Short-D" reference

```python
def align_with_germline(self, segments, threshold_objects, predicted_alleles, sequences):
```

2. **Segment Processing**
- Organizes segments by gene type:
    - Uses corrected segment positions from previous steps
    - Maintains chain-type specific processing (heavy vs light)

```python
  segments = {
      'v': [v_start, v_end],
      'j': [j_start, j_end],
      'd': [d_start, d_end]  # heavy chain only
  }
```

3. **Reference Mapping**
- Retrieves reference alleles from threshold objects
- Maps predicted sequences to reference sequences
- Generates alignment information for each segment

**Output**: Dictionary containing:
```python
{
    'v': mappings,  # V segment alignments
    'd': mappings,  # D segment alignments (heavy chain)
    'j': mappings   # J segment alignments
}
```

**Dependencies**:
- `HeuristicReferenceMatcher` for sequence alignment
- Base Step class
- Previous step results (corrected segments, allele info)

---
#### [IMGT Translation Step](#translation-step)
**Purpose**: Translates allele names between different naming conventions (ASC to IMGT), this assumes that by defualt the AlignAIR model
was trained on ASC allele naming scheme and not on IMGT naming scheme and that the DatConfig instance contains the ASC mapping dataframe

**Class**: `TranslationStep`

**Key Components**:

1. **Translation Control**
    - Checks `translate_to_asc` flag to determine if translation is needed
    - Only proceeds if translation is not set to ASC format

2. **Translation Process**
    - Creates `TranslateToIMGT` instance with data configuration
    - Processes V gene allele names
    - Performs nested translation:
         - Outer list: groups of alleles
         - Inner list: individual allele names
         - Applies translation to each allele name


**Key Features**:
    - Maintains allele grouping structure
    - Uses data configuration for translation reference
    - Logs translation progress
    - Preserves original structure of predictions

**Output**: Updates `predict_object` with:
```python
results['allele_info'][0]['v'] = [
    [translated_name1, translated_name2, ...],
    [translated_name1, translated_name2, ...],
    ...
]
```

**Dependencies**:
- 

TranslateToIMGT

 class for name conversion
- Base Step class
- Data configuration for reference mappings
---
#### [Results Finalization Step](#finalization-step)
**Purpose**: Compiles all results into a final DataFrame and saves to CSV

**Class**: `FinalizationStep`

**Key Components**:

1. **Result Compilation**
    - Aggregates results from previous steps:
    - Predicted alleles
    - Germline alignments
    - Allele likelihoods
    - Mutation rates
    - Productivity status
    - Indel counts
    - Handles chain-type specific data (heavy/light)

2. **DataFrame Creation**
    Creates pandas DataFrame with columns:
```python
    base_columns = {
        'sequence': sequences,
        'v_call': joined_v_alleles,
        'j_call': joined_j_alleles,
        'v_sequence_start/end': alignment_positions,
        'j_sequence_start/end': alignment_positions,
        'v/j_germline_start/end': reference_positions,
        'v/j_likelihoods': likelihood_scores,
        'mutation_rate': mutation_rates,
        'ar_indels': indel_counts,
        'ar_productive': productivity
    }
```

3. **Chain-Specific Processing**
    - Heavy Chain adds:
        - D segment information
         - Type = 'heavy'
    - Light Chain adds:
         - Type = 'kappa'/'lambda'
         - No D segment information

4. **File Output**

    - Generates output filename: `{save_path}{file_name}_alignairr_results.csv`
    - Saves DataFrame to CSV
    - Logs success message with file path
    
5. **Dependencies**:

    - pandas for DataFrame operations
    - Base Step class
    - Previous step results

