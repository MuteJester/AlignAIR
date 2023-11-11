import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count, Manager
from airrship.create_repertoire import generate_sequence, load_data, global_genotype
import threading
import time
import requests
from AlingAIRR_Sequence_Simulator import SequenceSimulator,SequenceSimulatorArguments
class SlackCallback:
    def __init__(self, webhook_url, interval=60):
        self.webhook_url = webhook_url
        self.interval = interval
        self.last_time = time.time()
        
    def send_message(self, text):
        data = {
            'text': text
        }
        response = requests.post(self.webhook_url, json=data)
        return response.status_code == 200

    def update(self, message):
        current_time = time.time()
        if current_time - self.last_time > self.interval:
            self.send_message(message)
            self.last_time = current_time

#path_to_data = '/home/bcrlab/thomas/anaconda3/lib/python3.9/site-packages/airrship/data/'
mutate = True
data_dict = load_data()
locus = global_genotype()
save_path = '/localdata/alignairr_data/AlignAIRR_Large_Train_Dataset/'
BATCH_SIZE = 1000
#slack_callback = SlackCallback("https://hooks.slack.com/services/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", interval=60)
num_samples = 30_000_000


def generate_samples(n, queue):
    args = SequenceSimulatorArguments()
    simulator = SequenceSimulator(args)
    for _ in range(n):
        simulation = simulator.get_sequence()
        queue.put(simulation)


def writer_thread(queue):
    buffer = []
    written_count = 0  # Keep track of the number of sequences written to the file
    while True:
        item = queue.get()
        if item == "STOP":
            if buffer:
                df = pd.DataFrame(buffer)
                df.to_csv(save_path, mode='a', header=False, index=False)
                written_count += len(buffer)
            #print(f"Total sequences written to file: {written_count}")
            if written_count % 100_000:
                ppp = np.round(written_count/num_samples,3)
                message = f'Hey Thomas!\n{written_count} Sequences Are Ready For Use, This Is {1-ppp}% Left to Be Generated \n'
            break
        buffer.append(item)
        if len(buffer) >= BATCH_SIZE:
            df = pd.DataFrame(buffer)
            df.to_csv(save_path, mode='a', header=False, index=False)
            written_count += BATCH_SIZE
            #print(f"Total sequences written to file: {written_count}")

            buffer = []

if __name__ == '__main__':
    num_cores = cpu_count()
    samples_per_core = num_samples // num_cores

    # Create a shared queue
    manager = Manager()
    queue = manager.Queue()

    # Start the writer thread
    writer = threading.Thread(target=writer_thread, args=(queue,))
    writer.start()

    # Create initial CSV with headers
    args = SequenceSimulatorArguments()
    simulator = SequenceSimulator(args)
    df = pd.DataFrame(columns=simulator.columns)
    df.to_csv(save_path, index=False)

    # Start the processes
    with Pool(num_cores) as pool:
        pool.starmap(generate_samples, [(samples_per_core, queue) for _ in range(num_cores)])

    # Signal the writer thread to stop
    queue.put("STOP")
    writer.join()



















# import tensorflow as tf
# from tensorflow.python.distribute.multi_process_lib import multiprocessing
# from SingleBeamUnboundedTrainerMPWorker import worker_UnboundedDataset
# from UnboundedTrainer import SingleBeamUnboundedTrainerMP
# from VDeepJModelExperimental import VDeepJAllignExperimentalSingleBeamRG
#
#
# def data_generator(queue):
#     while True:
#         batch_data, batch_labels = queue.get(True)
#         yield batch_data, batch_labels
#
#
# if __name__ == '__main__':
#
#     input_size = 512
#     corrupt_beginning = True
#     corrupt_proba = 0.7
#     nucleotide_add_coef = 210
#     nucleotide_remove_coef = 330
#     batch_size = 32
#     airrship_mutation_rate = 0.25
#     N_proportion = 0.02
#     random_sequence_add_proba = 0.45
#     single_base_stream_proba = 0.05
#     duplicate_leading_proba = 0.25
#     random_allele_proba = 0.25
#     randomize_rate = 1
#
#     num_processes = 3
#     processes = []
#     queue = multiprocessing.Queue(maxsize=35)
#     for p in range(num_processes):
#         process = multiprocessing.Process(target=worker_UnboundedDataset, args=(queue, input_size,
#                                                                                 corrupt_beginning,
#                                                                                 corrupt_proba,
#                                                                                 nucleotide_add_coef,
#                                                                                 nucleotide_remove_coef,
#                                                                                 batch_size,
#                                                                                 randomize_rate,
#                                                                                 airrship_mutation_rate,
#                                                                                 N_proportion,
#                                                                                 random_sequence_add_proba,
#                                                                                 single_base_stream_proba,
#                                                                                 duplicate_leading_proba,
#                                                                                 random_allele_proba))
#         process.daemon = True
#         processes.append(process)
#         print(f'Launching Process: {p}')
#         process.start()
#
#     data_gen = data_generator(queue)
#
#     trainer = SingleBeamUnboundedTrainerMP(
#         VDeepJAllignExperimentalSingleBeamRG,
#         epochs=2,
#         batch_size=64,
#         steps_per_epoch=150_000,
#         verbose=1,
#         data_gen=data_gen,
#         corrupt_beginning=corrupt_beginning,
#         classification_head_metric=[tf.keras.metrics.AUC(),tf.keras.metrics.AUC(),tf.keras.metrics.AUC()],
#         interval_head_metric=tf.keras.losses.mae,
#         corrupt_proba=corrupt_proba,
#         airrship_mutation_rate=airrship_mutation_rate,
#         nucleotide_add_coef=nucleotide_add_coef,
#         nucleotide_remove_coef=nucleotide_remove_coef,
#         random_sequence_add_proba=random_sequence_add_proba,
#         single_base_stream_proba=single_base_stream_proba,
#         duplicate_leading_proba=duplicate_leading_proba,
#         random_allele_proba=random_allele_proba,
#         num_parallel_calls=6,
#         optimizers_params={"clipnorm": 1},
#     )
#
#     # i = 0
#     # for x in data_gen:
#     #     print('B',i,'---',len(x))
#     #     i+=1
#     #     if i == 50:
#     #         break
#
#     trainer.train()