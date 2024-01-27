from src.const import DEBUG
from enum import Enum

# Enum that reserves names for predefined training sequences
class BuiltInTraining(Enum):
    DEFAULT = "default"
    MINI = "mini"
    UNINITIALIZED = "__uninitialized__"

# Enum defining optimizer types
class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"
    RMS_PROP = "rms_prop"

class TrainingSettings:
    '''
    Class to store training parameters.

    Parameters:
    - optimizer (OptimizerType): Optimizer used in the network. Options: 'SGD' (momentum gradient descent), 'RMSprop' (decaying learning rate), 'adam' (both). 'adam' is generally a good choice.
    - learning_rate (float): Learning rate for the optimizer. Higher values make learning easier initially but may hinder long-term accuracy. Value between 0.0 - 0.01.
    - epochs (int): Number of epochs to perform during training.
    - batch_size (int): Number of steps in a single epoch. A higher value may increase training time but helps with overfitting. data_count/batch_size determines the number of steps. If batch_size = 1, it is called stochastic gradient descent.
    - validation_split (float): Percentage of data used for validation.
    - print_summary (bool): Toggle for verbose output.
    - verbose (bool): Print network summary upon creation.
    '''

    def __init__(self,
                 optimizer: OptimizerType = OptimizerType.ADAM,
                 learning_rate: float = 0.001,
                 epochs: int = 30,
                 batch_size: int = 2048,
                 validation_split: float = 0.1,
                 print_summary: bool = True,
                 verbose: bool = True):
        '''
        Initialize TrainingSettings with specified parameters.

        Parameters:
        - dropout_rate: float, dropout rate for regularization
        - optimizer: OptimizerType, optimization algorithm for training the model
        - learning_rate: float, learning rate for the optimizer
        - epochs: int, number of training epochs
        - batch_size: int, size of training batches
        - validation_split: float, percentage of data to be used for validation
        - print_summary: bool, whether to print a model summary during training
        - verbose: bool, whether to display verbose training information
        '''

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.print_summary = print_summary
        self.verbose = verbose

        # Debug mode overrides
        if DEBUG:
            self.epochs = 1
            self.verbose = True

    def __str__(self) -> str:
        optimizer_dict = {OptimizerType.ADAM: 'adam', OptimizerType.SGD: 'SGD', OptimizerType.RMS_PROP: 'RMSProp'}

        return ";".join([
            f"{optimizer_dict[self.optimizer]}",
            f"{self.learning_rate}",
            f"{self.epochs}",
            f"{self.batch_size}",
            f"{self.validation_split}"
        ])

    @staticmethod
    def get_header() -> str:
        '''
        Return a list of semicolon separated column names.

        Returns:
        - str, formatted string representation of the header
        '''

        return ";".join([
            "Optimizer",
            "Learning rate",
            "Epochs",
            "Batch size",
            "Validation split"
        ])
