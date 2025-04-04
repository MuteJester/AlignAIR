import logging
from typing import List, Dict, Optional, Any
import pickle

from AlignAIR.Utilities.GenotypeYamlParser import GenotypeYamlParser
from AlignAIR.Utilities.step_utilities import DataConfigLibrary, FileInfo

class PredictObject:
    """
    A class to encapsulate all data and variables used during the prediction process.

    Attributes:
        sequences (Optional[List[str]]): List or array of sequences to be processed.
        groundtruth_table (Optional[Any]): Ground truth data table.
        model (Optional[Any]): The model that will be used for predictions.
        logger (Optional[logging.Logger]): Logger instance for logging.
        additional_data (Dict[str, Any]): Any additional data that might be needed during processing.
        script_arguments (Any): Arguments passed to the script.
        data_config_library (DataConfigLibrary): Configuration library for data.
        file_info (FileInfo): Information about the file being processed.
        orientation_pipeline (Optional[Any]): Pipeline for orientation processing.
        final_results (Optional[Any]): Final results of the prediction process.
        candidate_sequence_extractor (Optional[Any]): Extractor for candidate sequences.
        raw_predictions (Optional[Any]): Raw predictions from the model.
        processed_predictions (Optional[Any]): Processed predictions after post-processing.
        selected_allele_calls (Optional[Any]): Selected allele calls after thresholding.
        likelihoods_of_selected_alleles (Optional[Any]): Likelihoods of the selected alleles.
        threshold_extractor_instances (Optional[Any]): Instances of threshold extractors.
        germline_alignments (Optional[Any]): Germline alignment results.
    """

    def __init__(self, args: Any, sequences: Optional[List[str]] = None,
                 model: Optional[Any] = None,
                 logger: Optional[logging.Logger] = None,
                 additional_data: Optional[Dict[str, Any]] = None):
        """
        Initialize the PredictObject with default values.

        Args:
            sequences (Optional[List[str]]): List or array of sequences to be processed.
            model (Optional[Any]): The model that will be used for predictions.
            logger (Optional[logging.Logger]): Logger instance for logging.
            additional_data (Optional[Dict[str, Any]]): Any additional data that might be needed during processing.
        """
        self.sequences: Optional[List[str]] = sequences
        self.groundtruth_table: Optional[Any] = None
        self.model: Optional[Any] = model
        self.logger: Optional[logging.Logger] = logger
        self.additional_data: Dict[str, Any] = additional_data or {}
        self.script_arguments: Any = args
        self.data_config_library: DataConfigLibrary
        self.file_info: FileInfo
        self.orientation_pipeline: Optional[Any] = None
        self.final_results: Optional[Any] = None
        self.candidate_sequence_extractor: Optional[Any] = None
        self.genotype = None
        # results
        self.raw_predictions: Optional[Any] = None
        self.processed_predictions: Optional[Any] = None

        # allele likelihood thresholding results
        self.selected_allele_calls: Optional[Any] = None
        self.likelihoods_of_selected_alleles: Optional[Any] = None
        self.threshold_extractor_instances: Optional[Any] = None

        # germline alignment results
        self.germline_alignments: Optional[Any] = None

        # required processing methods


    def mount_genotype_list(self):
        """ Mount genotype from yaml file if provided and validate it matches the data config """
        if self.script_arguments.custom_genotype is not None:
            # parse the genotype yaml file
            genotype = GenotypeYamlParser(self.script_arguments.custom_genotype)
            # test if the genotype alleles intersect with the data config alleles, the genotype should be a subset of the data config used to train the model
            genotype.test_intersection_with_data_config(self.data_config_library)
            self.genotype = genotype

    def log(self, message: str):
        """
        Method to log a message using the object's logger, if it exists.

        Args:
            message (str): Message to log.
        """
        if self.logger:
            self.logger.info(message)

    def save(self, path: str):
        """
        Save the PredictObject instance as a pickle file, excluding the model.

        Args:
            path (str): Path to save the pickle file.
        """
        self.model = None
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'PredictObject':
        """
        Load a PredictObject instance from a pickle file.

        Args:
            path (str): Path to the pickle file.

        Returns:
            PredictObject: The loaded PredictObject instance.
        """
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj