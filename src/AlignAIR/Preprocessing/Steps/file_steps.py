from AlignAIR.PredictObject.PredictObject import PredictObject
from AlignAIR.Step.Step import Step
from AlignAIR.Utilities.file_processing import count_rows, tabular_sequence_generator, FILE_SEQUENCE_GENERATOR, \
    FILE_ROW_COUNTERS
from AlignAIR.Utilities.step_utilities import FileInfo, MultiFileInfoContainer


class FileNameExtractionStep(Step):
    def __init__(self, name):
        super().__init__(name)

    def process(self, predict_object: PredictObject):
        """
        Loads File Name and Suffix from file Path(s).
        Supports both single files and comma-separated multiple files.

        Args:
            predict_object (PredictObject)

        Returns:
            PredictObject: Updated with loaded configuration.
        """

        self.log(f"Extracting File Name and Suffix")
        
        # Check if we have comma-separated paths (multiple files)
        if ',' in predict_object.script_arguments.sequences:
            predict_object.file_info = MultiFileInfoContainer(predict_object.script_arguments.sequences)
            self.log("Extracted Multiple Files: {} with types: {}".format(
                predict_object.file_info.file_names(), 
                predict_object.file_info.file_types()
            ))
        else:
            # Single file - keep backward compatibility
            predict_object.file_info = FileInfo(predict_object.script_arguments.sequences)
            self.log("Extracted File Name: {} and the File Type is:  {}".format(
                predict_object.file_info.file_name,
                predict_object.file_info.file_type
            ))

        return predict_object


class FileSampleCounterStep(Step):
    def __init__(self, name):
        super().__init__(name)

    def process(self, predict_object: PredictObject):
        """
        Count the number of samples in file(s).
        Supports both single files and multiple files.

        Args:
            predict_object (PredictObject)

        Returns:
            PredictObject: Updated with loaded configuration.
        """

        self.log(f"Starting to Count Sample in Input File(s)")
        
        # Check if we have multiple files
        if hasattr(predict_object.file_info, 'file_infos'):
            # Multiple files
            for file_info in predict_object.file_info:
                row_counter = FILE_ROW_COUNTERS[file_info.file_type]
                number_of_samples = row_counter(file_info.path)
                file_info.set_sample_count(number_of_samples)
            
            total_samples = predict_object.file_info.total_sample_count()
            self.log(f"Finished Counting Samples: {total_samples} total across {len(predict_object.file_info)} files")
        else:
            # Single file
            row_counter = FILE_ROW_COUNTERS[predict_object.file_info.file_type]
            number_of_samples = row_counter(predict_object.script_arguments.sequences)
            predict_object.file_info.set_sample_count(number_of_samples)
            self.log(f"Finished Counting Samples: {number_of_samples} in single file")
        
        return predict_object
        self.log(f'There are : {number_of_samples} Samples for the AlignAIR to Process')

        return predict_object
