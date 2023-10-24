import time
from VDeepJUnbondedDataset import VDeepJUnbondedDatasetSingleBeam
import numpy as np

def worker_UnboundedDataset(queue, input_size=None, corrupt_beginning=None,
                            corrupt_proba=None, nucleotide_add_coef=None, nucleotide_remove_coef=None,
                            batch_size=None, randomize_rate=None, airrship_mutation_rate=None, N_proportion=None,
                            random_sequence_add_proba=None, single_base_stream_proba=None, duplicate_leading_proba=None,
                            random_allele_proba=None):
        # Generate and preprocess data here
        my_id = np.random.randint(0,10000,1).item()
        train_dataset = VDeepJUnbondedDatasetSingleBeam(max_sequence_length=input_size,
                                                        corrupt_beginning=corrupt_beginning,
                                                        corrupt_proba=corrupt_proba,
                                                        nucleotide_add_coef=nucleotide_add_coef,
                                                        nucleotide_remove_coef=nucleotide_remove_coef,
                                                        batch_size=batch_size,
                                                        randomize_rate=randomize_rate,
                                                        mutation_rate=airrship_mutation_rate,
                                                        N_proportion=N_proportion,
                                                        random_sequence_add_proba=random_sequence_add_proba,
                                                        single_base_stream_proba=single_base_stream_proba,
                                                        duplicate_leading_proba=duplicate_leading_proba,
                                                        random_allele_proba=random_allele_proba
                                                    )

        train_ds =train_dataset.get_train_dataset()
        train_ds = iter(train_ds)
        print(f'Worker:{my_id} Just Created A Dataset !')

        while True:  # Infinite loop to continuously generate data
            for x in train_ds:
                if not queue.full():
                    queue.put(x)
                    #print(f'Worker:{my_id} Just Add a Batch to the Queue !')
                else:
                    #print(f'Worker:{my_id} Queue is full, waiting...')
                    time.sleep(0.5)  # Wait for a short duration before checking again

