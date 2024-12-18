from abc import ABC, abstractmethod
import logging

class Step(ABC):
    """
    Abstract base class for processing steps, with optional logging.
    """
    logger = None
    @classmethod
    def set_logger(cls, logger):
        cls.logger = logger
    def __init__(self, name):
        """
        Initialize the Step with a name and an optional logger.

        Args:
            name (str): The name of the step.
            logger (logging.Logger, optional): Logger instance for logging. Defaults to None.
        """
        self.name = name

    def log(self, message):
        """
        Log a message if a logger is provided.

        Args:
            message (str): Message to log.
        """
        if self.logger:
            self.logger.info(message)

    def execute(self, data):
        """
        Execute the step's processing action. Logs the start and end if a logger is provided.

        Args:
            data (any): Input data to process.

        Returns:
            any: Processed data.
        """
        self.log(f"Starting {self.name}...")
        result = self.process(data)
        self.log(f"Completed {self.name}.")
        return result

    def process(self, data):
        """
        Process the input data. To be implemented by subclasses.

        Args:
            data (any): Data to process.

        Returns:
            any: Processed data.
        """
        pass
