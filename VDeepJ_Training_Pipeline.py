import pandas as pd
import tensorflow as tf
from tqdm.keras import TqdmCallback
from tqdm.auto import tqdm
tqdm.pandas()
from VDeepJDataPrepper import VDeepJDataPrepper
from VDeepJModel import VDeepJAllign

# To-Do: Convert all the code below into a class that will handle the training given the dataset, similar to what transformer "Trainer" does.
# Step 1: Create Data Prepper
max_seq_length = 512
prepper = VDeepJDataPrepper(max_seq_length)

# Step 2: Load Data
data = pd.read_table(r'E:\igor_generated_5M\shm_rep_airship_0.8_flatvdj_gene.tsv',usecols=['sequence','v_call','d_call','j_call','v_sequence_end','d_sequence_start','j_sequence_start',
                                                                'j_sequence_end','d_sequence_end','v_sequence_start'],nrows = 150000)


# Step 3: Derive a mapping between calls and their respective family/gene/allele
v_dict,d_dict,j_dict = prepper.get_family_gene_allele_map(data.v_call.unique(),data.d_call.unique(),data.j_call.unique())

# Step 4: Add seperated family/gene/allele for each call type back into our train dataset
data['v_family'] = data.v_call.progress_apply(lambda x: v_dict[x]['family'])
data['d_family'] = data.d_call.progress_apply(lambda x: d_dict[x]['family'])

data['v_gene'] = data.v_call.progress_apply(lambda x: v_dict[x]['gene'])
data['d_gene'] = data.d_call.progress_apply(lambda x: d_dict[x]['gene'])
data['j_gene'] = data.j_call.progress_apply(lambda x: j_dict[x]['gene'])

data['v_allele'] = data.v_call.progress_apply(lambda x: v_dict[x]['allele'])
data['d_allele'] = data.d_call.progress_apply(lambda x: d_dict[x]['allele'])
data['j_allele'] = data.j_call.progress_apply(lambda x: j_dict[x]['allele'])

# Step 5: Get one-hot encoded matrices for each call type and level + the reverse mapping dictionary
v_family_call_ohe,v_family_call_ohe_np,v_gene_call_ohe,v_gene_call_ohe_np,v_allele_call_ohe,v_allele_call_ohe_np = prepper.convert_calls_to_one_hot(data['v_gene'],data['v_allele'],data['v_family'])
d_family_call_ohe,d_family_call_ohe_np,d_gene_call_ohe,d_gene_call_ohe_np,d_allele_call_ohe,d_allele_call_ohe_np = prepper.convert_calls_to_one_hot(data['d_gene'],data['d_allele'],data['d_family'])
j_gene_call_ohe,j_gene_call_ohe_np,j_allele_call_ohe,j_allele_call_ohe_np = prepper.convert_calls_to_one_hot(data['j_gene'],data['j_allele'])

# Step 6: Get the counts of each call type level in order to know the classification heads dimension for each call level
v_family_count = len(v_family_call_ohe)
v_gene_count   = len(v_gene_call_ohe)
v_allele_count = len(v_allele_call_ohe)

d_family_count = len(d_family_call_ohe)
d_gene_count   = len(d_gene_call_ohe)
d_allele_count = len(d_allele_call_ohe)

j_gene_count   = len(j_gene_call_ohe)
j_allele_count = len(j_allele_call_ohe)

# Step 7: Get Train Dataset
batch_size = 64
train_dataset = prepper.get_train_dataset(data,v_family_call_ohe_np,v_gene_call_ohe_np,v_allele_call_ohe_np,
                                          d_family_call_ohe_np,d_gene_call_ohe_np,d_allele_call_ohe_np,
                                          j_gene_call_ohe_np,j_allele_call_ohe_np,batch_size,
                                          train=True,corrupt_beginning=True)

# Step 8: Create a VDeepJ instance
vdeepj = VDeepJAllign(max_seq_length,
                      v_family_count,
                      v_gene_count,
                      v_allele_count,
                      d_family_count,
                      d_gene_count,
                      d_allele_count,
                      j_gene_count,
                      j_allele_count)

# Step 9: Compile the model instance
vdeepj.compile(optimizer=tf.optimizers.Adam(),#'adam',
                       loss=None,
                       metrics={'v_start':'mae','v_end':'mae','d_start':'mae','d_end':'mae','j_start':'mae','j_end':'mae',
                                'v_family':'categorical_accuracy',
                                'v_gene':'categorical_accuracy',
                                'v_allele':'categorical_accuracy',
                                'd_family':'categorical_accuracy',
                                'd_gene':'categorical_accuracy',
                                'd_allele':'categorical_accuracy',
                                'j_gene':'categorical_accuracy',
                                'j_allele':'categorical_accuracy',

                               })






# Step 10: Fit Model
steps_per_epoch = len(data) // batch_size
epochs = 2

history = vdeepj.fit(
    train_dataset,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    verbose=0,
    callbacks=[TqdmCallback(1)]

)


