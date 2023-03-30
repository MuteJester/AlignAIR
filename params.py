# General
max_seq_length = 512
batch_size = 64
epochs = 1
debug_forward = True

# Data
# Debug Mode
# train_data_dir = r"/home/eng/eisenbr2/vdj_nlp/airrship_data/val/airrship_01_val.tsv"
# val_data_dir = r"/home/eng/eisenbr2/vdj_nlp/airrship_data/val/airrship_01_val.tsv"

# Non Debug Mode
train_data_dir = (
    r"/home/bcrlab/eisenbr2/vdj_nlp/airrship_data/train/airrship_01_train.tsv"
)
val_data_dir = r"/home/eng/eisenbr2/vdj_nlp/airrship_data/val/airrship_01_val.tsv"

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
corrupt_beginning = True
new_allels = False
new_allels_df_dir = r"/home/eng/eisenbr2/vdj_nlp/data/alleles_db.csv"

# Model
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
