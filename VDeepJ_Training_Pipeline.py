import pandas as pd
import tensorflow as tf
from tqdm.keras import TqdmCallback
from tqdm.auto import tqdm

tqdm.pandas()
from VDeepJDataPrepper import VDeepJDataPrepper
from VDeepJDataPrepper import VdjClassHolder
from VDeepJModel import VDeepJAllign
import params

# To-Do: Convert all the code below into a class that will handle the training given the dataset, similar to what transformer "Trainer" does.
# Step 1: Create Data Prepper
prepper = VDeepJDataPrepper(params.max_seq_length)

print("*" * 40)
print(" Loading Data")
print("*" * 40)

# Step 2: Load Data
train_data = pd.read_table(params.train_data_dir, usecols=params.data_cloumns)
val_data = pd.read_table(
    params.val_data_dir,
    usecols=params.data_cloumns,
)

print("*" * 40)
print(" Adding classes to data table")
print("*" * 40)
# Step 3: Derive a mapping between calls and their respective family/gene/allele
# Step 4: Add seperated family/gene/allele for each call type back into our train dataset
(
    train_data,
    v_dict,
    d_dict,
    j_dict,
    sub_cllases,
) = prepper.modify_data_table_with_family_gene_allele(train_data, params)
val_data, _, _, _, _, = prepper.modify_data_table_with_family_gene_allele(
    val_data, params, v_dict, d_dict, j_dict
)

print("*" * 40)
print(" Creating data objects")
print("*" * 40)
# Step 5: Get one-hot encoded matrices for each call type and level + the reverse mapping dictionary
# Step 6: Get the counts of each call type level in order to know the classification heads dimension for each call level
vdj_class_holder_train = VdjClassHolder(train_data, sub_cllases)
vdj_class_holder_val = VdjClassHolder(val_data, sub_cllases)

# Step 7: Get Train/Test Datasets
train_dataset = prepper.get_train_dataset(
    train_data, vdj_class_holder_train, params, train=True
)
val_dataset = prepper.get_train_dataset(
    val_data, vdj_class_holder_val, params, train=True
)

# Step 8: Create a VDeepJ instance
vdeepj = VDeepJAllign(params.max_seq_length, vdj_class_holder_train)

# Step 9: Compile the model instance
vdeepj.compile(
    optimizer=tf.optimizers.Adam(),  #'adam',
    loss=None,
    metrics=params.compile_metrics_dict,
)

if params.debug_forward:
    x, y = next(iter(train_dataset))
    # vdeepj.call(x)
    vdeepj.train_step([x, y])

# Step 10: Fit Model
steps_per_epoch_train = len(train_data) // params.batch_size

#
history = vdeepj.fit(
    train_dataset,
    epochs=params.epochs,
    steps_per_epoch=steps_per_epoch_train,
    verbose=0,
    callbacks=[TqdmCallback(1)],
)

# Step 11: Evauate the model
steps_per_epoch_val = len(val_data) // params.batch_size

score = vdeepj.evaluate(
    val_dataset, verbose=0, steps=steps_per_epoch_val, return_dict=True
)
print(score)
