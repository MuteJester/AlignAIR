from AlignAIR.PredictObject.PredictObject import PredictObject
from AlignAIR.Step.Step import Step
from AlignAIR.Utilities.file_processing import count_rows, tabular_sequence_generator, FILE_SEQUENCE_GENERATOR, \
    FILE_ROW_COUNTERS
from AlignAIR.Utilities.step_utilities import FileInfo


class FileNameExtractionStep(Step):
    def __init__(self, name):
        super().__init__(name)

    def process(self, predict_object: PredictObject):
        """
        Loads File Name and Suffix from file Path.

        Args:
            predict_object (PredictObject)

        Returns:
            PredictObject: Updated with loaded configuration.
        """

        self.log(f"Extracting File Name and Suffix")
        predict_object.file_info = FileInfo(predict_object.script_arguments.sequences)
        self.log("Extracted File Name: {} and the File Type is:  {}".format(predict_object.file_info.file_name,
                                                                            predict_object.file_info.file_type))

        return predict_object


class FileSampleCounterStep(Step):
    def __init__(self, name):
        super().__init__(name)

    def process(self, predict_object: PredictObject):
        """
        Count the number of samples in file.

        Args:
            predict_object (PredictObject)

        Returns:
            PredictObject: Updated with loaded configuration.
        """

        self.log(f"Starting to Count Sample in Input File")
        row_counter = FILE_ROW_COUNTERS[predict_object.file_info.file_type]
        number_of_samples = row_counter(predict_object.script_arguments.sequences)
        predict_object.file_info.set_sample_count(number_of_samples)
        self.log("Finished Counting Sample in Input File")
        self.log(f'There are : {number_of_samples} Samples for the AlignAIR to Process')

        return predict_object
