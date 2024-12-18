import multiprocessing
import time
from multiprocessing import Process

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import torch

from AlignAIR.PredictObject.PredictObject import PredictObject
from AlignAIR.Pytorch.Dataset import CSVReaderDataset
from AlignAIR.Pytorch.InputPreProcessors import SequenceTokenizer
from AlignAIR.Step.Step import Step
from AlignAIR.Utilities.consumer_producer import READER_WORKER_TYPES


class BatchProcessingStep(Step):
    def __init__(self, name):
        super().__init__(name)

    def start_tokenizer_process(self, file_path, max_seq_length, logger, orientation_pipeline,
                                candidate_sequence_extractor, batch_size=256):
        tokenizer_dictionary = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 5, "P": 0}  # pad token
        queue = multiprocessing.Queue(maxsize=64)  # Control the prefetching size
        file_type = file_path.split('.')[-1]  # get the file type i.e .csv,.tsv or .fasta
        worker_reading_type = READER_WORKER_TYPES[file_type]
        process = Process(target=worker_reading_type,
                          args=(file_path, queue, max_seq_length, tokenizer_dictionary, batch_size, logger,
                                orientation_pipeline, candidate_sequence_extractor))
        process.start()
        self.log('Producer Process Started!')
        return queue, process

    def execute(self, predict_object: PredictObject):
        self.log("Starting batch processing...")
        queue, process = self.start_tokenizer_process(
            predict_object.file_info.path,
            predict_object.script_arguments.max_input_size,
            predict_object.orientation_pipeline,
            predict_object.candidate_sequence_extractor,
            predict_object.script_arguments.batch_size,
        )

        predictions = []
        sequences = []
        batch_number = 0
        batch_times = []
        start_time = time.time()
        total_batches = int(np.ceil(len(predict_object.file_info) / predict_object.script_arguments.batch_size))

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
                                                                batch_size=predict_object.script_arguments.batch_size))
                batch_times.append(time.time() - batch_start_time)

                # Logging
                batch_number += 1
                avg_batch_time = sum(batch_times) / len(batch_times)
                estimated_time_remaining = avg_batch_time * (total_batches - batch_number)
                self.log(
                    f"Processed Batch {batch_number}/{total_batches}. Queue Size {queue.qsize()}  > Estimated Time Remaining: {estimated_time_remaining:.2f} seconds.")
        finally:
            total_duration = time.time() - start_time
            self.log(f"All batches processed in {total_duration:.2f} seconds.")
            process.join()

        predict_object.raw_predictions = predictions
        predict_object.sequences = sequences

        return predict_object


def detach_and_move_to_cpu(output_dict):
    return {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in output_dict.items()}


def concatenate_predictions(predictions):
    """
    Concatenate a list of dictionaries with tensor or numpy array values
    into a single dictionary.

    Args:
        predictions (list): List of dictionaries containing predictions.

    Returns:
        dict: A dictionary with concatenated arrays for each key.
    """
    concatenated = {}

    for batch in predictions:
        for key, value in batch.items():
            if key not in concatenated:
                # Initialize with an empty list
                concatenated[key] = []
            if isinstance(value, torch.Tensor):
                concatenated[key].append(value.cpu().numpy())
            elif isinstance(value, np.ndarray):
                concatenated[key].append(value)
            else:
                raise ValueError(f"Unsupported value type for key '{key}': {type(value)}")

    # Concatenate all lists into arrays
    for key in concatenated:
        concatenated[key] = np.concatenate(concatenated[key], axis=0)

    return concatenated


class PytorchBatchProcessingStep(Step):

    # def start_tokenizer_process(self,file_path, max_seq_length, logger, orientation_pipeline,candidate_sequence_extractor, batch_size=256):
    #     tokenizer_dictionary = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 5, "P": 0}  # pad token
    #     queue = multiprocessing.Queue(maxsize=64)  # Control the prefetching size
    #     file_type = file_path.split('.')[-1]  # get the file type i.e .csv,.tsv or .fasta
    #     worker_reading_type = READER_WORKER_TYPES[file_type]
    #     # process = Process(target=worker_reading_type,
    #     #                   args=(file_path, queue, max_seq_length, tokenizer_dictionary, batch_size, logger,
    #     #                         orientation_pipeline,candidate_sequence_extractor))
    #     # process.start()
    #     # self.log('Producer Process Started!')
    #     return queue, process

    def execute(self, predict_object):
        self.log("Starting batch processing...")
        # queue, process = self.start_tokenizer_process(
        #     predict_object.script_arguments.sequences,
        #     predict_object.script_arguments.max_input_size,
        #     predict_object.orientation_pipeline,
        #     predict_object.candidate_sequence_extractor,
        #     predict_object.script_arguments.batch_size,
        # )

        predictions = []
        sequences = []
        batch_number = 0
        batch_times = []
        start_time = time.time()
        total_batches = int(np.ceil(predict_object.number_of_samples / predict_object.script_arguments.batch_size))

        tokenizer = SequenceTokenizer(predict_object.data_config, return_original_sequence=True)
        dataset = CSVReaderDataset(
            csv_file=predict_object.script_arguments.sequences,
            preprocessor=tokenizer,  # Custom preprocessing logic
            batch_size=64,
            separator=',')
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        predictions = []
        for i in tqdm(dataloader):
            batch = i['x'].to('cuda:0')
            original_sequences = i['x_original']
            with torch.no_grad():  # Disable gradient computation for inference
                output = predict_object.model(batch)
                detached_output = detach_and_move_to_cpu(output)  # Detach and move to CPU
            predictions.append(detached_output)
            sequences.extend(original_sequences)
        #predictions = concatenate_predictions(predictions)
        predict_object.results['predictions'] = predictions
        predict_object.sequences = sequences
        return predict_object
