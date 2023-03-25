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

# Step 2: Load Data
print("*" * 40)
print(" Loading Data")
print("*" * 40)

data = pd.read_table(
    params.data_dir,
    usecols=params.data_cloumns,
)

# Step 3: Derive a mapping between calls and their respective family/gene/allele
# Step 4: Add seperated family/gene/allele for each call type back into our train dataset
print("*" * 40)
print(" Adding classes to data table")
print("*" * 40)
data = prepper.modify_data_table_with_family_gene_allele(data)

# Step 5: Get one-hot encoded matrices for each call type and level + the reverse mapping dictionary
# Step 6: Get the counts of each call type level in order to know the classification heads dimension for each call level
vdj_class_holder = VdjClassHolder(data)

# Step 7: Get Train Dataset
train_dataset = prepper.get_train_dataset(
    data, vdj_class_holder, params.batch_size, train=True, corrupt_beginning=True
)

# Step 8: Create a VDeepJ instance
vdeepj = VDeepJAllign(params.max_seq_length, vdj_class_holder)

# Step 9: Compile the model instance
vdeepj.compile(
    optimizer=tf.optimizers.Adam(),  #'adam',
    loss=None,
    metrics=params.compile_metrics_dict,
)

# Step 10: Fit Model
steps_per_epoch = len(data) // params.batch_size

vdeepj.call(train_dataset)
#
history = vdeepj.fit(
    train_dataset,
    epochs=params.epochs,
    steps_per_epoch=steps_per_epoch,
    verbose=0,
    callbacks=[TqdmCallback(1)],
)
