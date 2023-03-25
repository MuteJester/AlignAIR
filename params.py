#
max_seq_length = 512
batch_size = 64
epochs = 2

# Data
data_dir = r"/home/eng/eisenbr2/vdj_nlp/airrship_data/val/airrship_01_val.tsv"
# data_dir = r"/home/eng/eisenbr2/vdj_nlp/airrship_data/train/airrship_01_train.tsv"
data_cloumns = [
    "sequence",
    "v_call",
    "d_call",
    "j_call",
    "v_sequence_end",
    "d_sequence_start",
    "j_sequence_start",
    "j_sequence_end",
    "d_sequence_end",
    "v_sequence_start",
]

compile_metrics_dict = {
    "v_start": "mae",
    "v_end": "mae",
    "d_start": "mae",
    "d_end": "mae",
    "j_start": "mae",
    "j_end": "mae",
    "v_family": "categorical_accuracy",
    "v_gene": "categorical_accuracy",
    "v_allele": "categorical_accuracy",
    "d_family": "categorical_accuracy",
    "d_gene": "categorical_accuracy",
    "d_allele": "categorical_accuracy",
    "j_gene": "categorical_accuracy",
    "j_allele": "categorical_accuracy",
}
