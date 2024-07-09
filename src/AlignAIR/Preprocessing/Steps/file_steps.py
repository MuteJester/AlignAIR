from AlignAIR.Step.Step import Step
from AlignAIR.Utilities.predict_script_utilities import get_filename
from AlignAIR.Utilities.file_processing import count_rows, tabular_sequence_generator, FILE_SEQUENCE_GENERATOR, \
    FILE_ROW_COUNTERS

class FileNameExtractionStep(Step):
    def __init__(self, name, logger=None):
        super().__init__(name, logger)

    def process(self, predict_object):
        """
        Loads File Name and Suffix from file Path.

        Args:
            predict_object (PredictObject)

        Returns:
            PredictObject: Updated with loaded configuration.
        """

        self.log(f"Extracting File Name and Suffix")
        file_name, file_type = get_filename(predict_object.script_arguments.sequences)


        self.log("Data Config loaded successfully")
        predict_object.file_name = file_name
        predict_object.file_suffix = file_type
        return predict_object


class FileSampleCounterStep(Step):
    def __init__(self, name, logger=None):
        super().__init__(name, logger)

    def process(self, predict_object):
        """
        Count the number of samples in file.

        Args:
            predict_object (PredictObject)

        Returns:
            PredictObject: Updated with loaded configuration.
        """

        self.log(f"Starting to Count Sample in Input File")
        row_counter = FILE_ROW_COUNTERS[predict_object.file_suffix.replace('.','')]
        number_of_samples = row_counter(predict_object.script_arguments.sequences)
        predict_object.number_of_samples = number_of_samples
        self.log("Finished Counting Sample in Input File")
        self.log(f'There are : {number_of_samples} Samples for the AlignAIR to Process')

        return predict_object
