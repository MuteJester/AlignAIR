"""BatchInferenceStage — tokenize and predict in batches using multiprocess producer."""
from __future__ import annotations

import logging
import multiprocessing
import time
from multiprocessing import Process
from typing import Any, Dict, List

import numpy as np

from AlignAIR.Pipeline.Stage.protocol import Stage, StageContext

logger = logging.getLogger("AlignAIR.Pipeline")


class BatchInferenceStage(Stage):
    """Tokenize sequences and run model inference in batches.

    Reuses the existing multiprocess producer-consumer pattern for tokenization
    to maintain identical behavior with the baseline pipeline.
    """

    reads = frozenset({"config", "model", "sequences"})
    writes = frozenset({"raw_predictions"})

    def run(self, context: StageContext) -> Dict[str, Any]:
        config = context.config
        model = context["model"]
        sequences = context["sequences"]

        file_info = context.get("file_info")
        file_path = config.sequences_path
        batch_size = config.memory.batch_size
        max_seq_length = model.max_seq_length

        # Use the existing multiprocess tokenization pipeline
        from AlignAIR.Utilities.consumer_producer import READER_WORKER_TYPES

        tokenizer_dictionary = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 5, "P": 0}
        file_type = file_path.rsplit('.', 1)[-1]
        worker_type = READER_WORKER_TYPES[file_type]

        queue = multiprocessing.Queue(maxsize=64)
        process = Process(
            target=worker_type,
            args=(
                file_path,
                queue,
                max_seq_length,
                tokenizer_dictionary,
                None,  # logger (handled at pipeline level)
                model.orientation_pipeline,
                model.candidate_extractor,
                batch_size,
            ),
        )
        process.start()
        logger.info("Producer process started")

        predictions: List[dict] = []
        all_sequences: List[str] = []
        batch_number = 0
        batch_times: List[float] = []
        start_time = time.time()
        total_batches = int(np.ceil(len(sequences) / batch_size))

        try:
            while True:
                batch = queue.get()
                if batch is None:
                    break
                tokenized_batch, orientation_fixed_sequences = batch
                all_sequences.extend(orientation_fixed_sequences)

                batch_start = time.time()
                pred = model.inference_wrapper.predict(
                    {'tokenized_sequence': tokenized_batch},
                    verbose=0,
                    batch_size=batch_size,
                )
                batch_times.append(time.time() - batch_start)

                predictions.append(pred)

                batch_number += 1
                avg_time = sum(batch_times) / len(batch_times)
                eta = avg_time * (total_batches - batch_number)
                logger.info(
                    "Processed batch %d/%d. Queue size %d. ETA: %.2fs",
                    batch_number, total_batches, queue.qsize(), eta,
                )
        finally:
            total_duration = time.time() - start_time
            logger.info("All batches processed in %.2fs", total_duration)
            process.join()

        # Return raw predictions list and updated sequences
        # (orientation-fixed sequences replace the original ones)
        return {"raw_predictions": predictions, "sequences": all_sequences}

    # Note: raw_predictions is a list of dicts (one per batch) at this point.
    # CleanAndExtractStage will merge them into arrays.
