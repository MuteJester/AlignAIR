class PredictObject:
    """
    A class to encapsulate all data and variables used during the prediction process.
    """

    def __init__(self,args, sequences=None, config=None, model=None, results=None, logger=None, additional_data=None):
        """
        Initialize the PredictObject with default values.

        Args:
            sequences (list, optional): List or array of sequences to be processed.
            config (dict, optional): Configuration data for various steps.
            model (object, optional): The model that will be used for predictions.
            results (dict, optional): Dictionary to store results of predictions and post-processings.
            logger (logging.Logger, optional): Logger instance for logging.
            additional_data (dict, optional): Any additional data that might be needed during processing.
        """
        self.sequences = sequences
        self.groundtruth_table = None
        self.config = config
        self.model = model
        self.results = results or {}
        self.logger = logger
        self.additional_data = additional_data or {}
        self.script_arguments = args
        self.chain_type = args.chain_type
        self.data_config = None
        self.file_name = None
        self.file_suffix = None
        self.number_of_samples = 0
        self.orientation_pipeline=None
        self.model = None
        self.final_results = None
        self.candidate_sequence_extractor = None

    def log(self, message):
        """
        Method to log a message using the object's logger, if it exists.

        Args:
            message (str): Message to log.
        """
        if self.logger:
            self.logger.info(message)
