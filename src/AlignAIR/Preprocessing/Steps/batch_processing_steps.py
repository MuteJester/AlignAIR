import multiprocessing
import time
from multiprocessing import Process

import numpy as np

from AlignAIR.Step.Step import Step
from AlignAIR.Utilities.consumer_producer import READER_WORKER_TYPES


class BatchProcessingStep(Step):

    def start_tokenizer_process(self,file_path, max_seq_length, logger, orientation_pipeline, batch_size=256):
        tokenizer_dictionary = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 5, "P": 0}  # pad token
        queue = multiprocessing.Queue(maxsize=64)  # Control the prefetching size
        file_type = file_path.split('.')[-1]  # get the file type i.e .csv,.tsv or .fasta
        worker_reading_type = READER_WORKER_TYPES[file_type]
        process = Process(target=worker_reading_type,
                          args=(file_path, queue, max_seq_length, tokenizer_dictionary, batch_size, logger,
                                orientation_pipeline))
        process.start()
        self.log('Producer Process Started!')
        return queue, process


    def execute(self, predict_object):
        self.log("Starting batch processing...")
        queue, process = self.start_tokenizer_process(
            predict_object.script_arguments.sequences,
            predict_object.script_arguments.max_input_size,
            predict_object.orientation_pipeline,
            predict_object.script_arguments.batch_size,
        )


        predictions = []
        sequences = []
        batch_number = 0
        batch_times = []
        start_time = time.time()
        total_batches = int(np.ceil(predict_object.number_of_samples / predict_object.script_arguments.batch_size))

        try:
            while True:
                batch = queue.get()
                if batch is None:
                    break
                tokenized_batch, orientation_fixed_sequences = batch
                sequences.extend(orientation_fixed_sequences)

                # Predict
                batch_start_time = time.time()
                predictions.append(predict_object.model.predict({'tokenized_sequence': tokenized_batch}, verbose=0,
                                                                batch_size=predict_object.script_arguments.batch_size) )
                batch_times.append(time.time() - batch_start_time)

                # Logging
                batch_number += 1
                avg_batch_time = sum(batch_times) / len(batch_times)
                estimated_time_remaining = avg_batch_time * (total_batches - batch_number)
                self.log(f"Processed Batch {batch_number}/{total_batches}. Queue Size {queue.qsize()}  > Estimated Time Remaining: {estimated_time_remaining:.2f} seconds.")
        finally:
            total_duration = time.time() - start_time
            self.log(f"All batches processed in {total_duration:.2f} seconds.")
            process.join()

        predict_object.results['predictions'] = predictions
        predict_object.sequences = sequences
        return predict_object

