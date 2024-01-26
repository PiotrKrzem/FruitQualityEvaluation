from enum import Enum
from src.const import DEBUG

# Enum that reserves names for pretrained models
class BuiltInModel(Enum):
    ALEXNET = "AlexNet"
    RESNET_PRETRAINED = "ResNet_pretrained"
    MINI = "Mini"
    UNINITIALIZED = "__uninitialized__"

# Enum defining activation types
class ActivationType(Enum):
    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"
    SIGMOID = "sigmoid"

# Enum defining middle layer types
class MiddleLayerType(Enum):
    FLATTEN = "flatten"
    MAX_POOL = "max_pool"
    AVERAGE_POOL = "average_pool"

class ModelSettings:
    '''
    Class to store model parameters.

    Parameters:
    - input_size (int): Size of the input data. If the images have not been processed to this size, a preprocessor sequence will execute before training. Input size is a single integer that is later expanded to [input_size, input_size, 3].
    - convolution_layers (list): List of integers or tuples of integers specifying convolutional layer sizes. One layer is defined as a set of Conv2D operations followed by the MaxPool2D operation. The length of this array determines the number of these layers in the network. MaxPool2D is applied after each layer excluding the last one. Provide a single integer for a single Conv2D operation or an array of integers to chain multiple Conv2D operations before MaxPool2D. Using power-of-2 numbers is recommended. Default is [(32, 32), (64, 64), (128, 128)].
    - convolution_activation (ActivationType): Activation function used in the convolutional filters. Options: 'relu', 'tanh'. 'relu' is faster but may cause the loss function to explode upwards. Default is ActivationType.RELU.
    - convolution_sizes (list): List of integers specifying sizes for convolutional layers. The length of this array must match convolution_layers'. Only integers are allowed. Default is [3, 3, 3].
    - middle_layer (MiddleLayerType): Type of middle layer in the model connecting Convolutional layers with Dense layers. Options: 'global_avg', 'maxpool', 'flatten'. 'global_avg' is faster but might lose some information. 'maxpool' is a versatile option. 'flatten' is the best but slower. Default is MiddleLayerType.FLATTEN.
    - dense_layers (list): List of integers specifying sizes for dense layers. The length of this array determines the number of layers in the network. Dropout is applied after each layer excluding the last one. Using power-of-2 numbers is recommended. Default is [512, 512, 512].
    - dense_activation (ActivationType): Activation function used in the neurons of dense layers. Options: 'relu', 'tanh'. 'relu' is faster but the loss function might explode upwards. Default is ActivationType.RELU.
    - dropout_rate (float): Probability of deactivating the output of any neuron. A number between 0 and 1, helps prevent overfitting.
    - model_name (str): Name of the model. Default is 'model'.
    - builtin (bool): Whether it is a builtin model with predefined architecture. Default is False.
    - pretrained (bool): Whether it is a pretrained model (load model) or a new model. Default is False.
    '''

    def __init__(self,
                 input_size: int = 32,
                 convolution_layers: list = [(32, 32), (64, 64), (128, 128)],
                 convolution_activation: ActivationType = ActivationType.RELU,
                 convolution_sizes: list = [3, 3, 3],
                 middle_layer: MiddleLayerType = MiddleLayerType.FLATTEN,
                 dense_layers: list = [512, 512, 512],
                 dense_activation: ActivationType = ActivationType.RELU,
                 output_activation: ActivationType = ActivationType.SIGMOID,
                 dropout_rate: float = 0.2,
                 model_name: str = 'model',
                 builtin = False,
                 pretrained = False):
        '''
        Initialize ModelSettings with specified parameters.

        Parameters:
        - input_size (int): Size of the input data, automatically converted to [input_size, input_size, 3].
        - convolution_layers (list): List of integers or tuples of integers specifying convolutional layer sizes.
        - convolution_activation (ActivationType): Activation function for convolutional layers.
        - convolution_sizes (list): List of integers specifying sizes for convolutional layers.
        - middle_layer (MiddleLayerType): Type of middle layer in the model.
        - dense_layers (list): List of integers specifying sizes for dense layers.
        - dense_activation (ActivationType): Activation function for dense layers.
        - output_activation (ActivationType): Activation function for output layer.
        - model_name (str): Name of the model.
        - pretrained (bool): Whether the model is pretrained.
        '''

        self.input_size = input_size
        self.convolution_layers = convolution_layers
        self.convolution_activation = convolution_activation
        self.convolution_sizes = convolution_sizes
        self.middle_layer = middle_layer
        self.dense_layers = dense_layers
        self.dense_activation = dense_activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.model_name = model_name
        self.pretrained = pretrained
        self.builtin = builtin

        # Asserting the consistency of convolution layers and sizes
        assert len(self.convolution_layers) == len(self.convolution_sizes)

        # Asserting model_name does not collide with any name of the BuiltIn models
        reserved_names = [name.value for name in list(BuiltInModel)]
        reserved_names.remove(BuiltInModel.UNINITIALIZED.value) # Valid value for BuiltIn models
        assert self.model_name not in reserved_names

    def __str__(self) -> str:
        """
        Return a formatted string representation of the ModelSettings.

        Returns:
        - str, formatted string representation of the model settings
        """

        # Dictionary mappings for string representations
        activation_dict = {ActivationType.RELU: 'relu', ActivationType.TANH: 'tanh'}
        middle_layer_dict = {MiddleLayerType.FLATTEN: 'flatten', MiddleLayerType.MAX_POOL: 'max_pool', MiddleLayerType.AVERAGE_POOL: 'avg_pool'}

        # Returning a formatted string representation of the model settings
        return ";".join([
            f"{self.model_name}",
            f"{self.pretrained}",
            f"{self.input_size}",
            f"{self.convolution_layers}",
            f"{activation_dict[self.convolution_activation]}",
            f"{self.convolution_sizes}",
            f"{middle_layer_dict[self.middle_layer]}",
            f"{self.dense_layers}",
            f"{activation_dict[self.dense_activation]}",
            f"{self.dropout_rate}",
        ])

    @staticmethod
    def get_header() -> str:
        """
        Return a list of semicolon separated column names.

        Returns:
        - str, formatted string representation of the header
        """

        return ";".join([
            "Model name",
            "Pretrained",
            "Input size",
            "Convolution layers",
            "Convolution activation",
            "Convolution sizes",
            "Middle layer",
            "Dense layers",
            "Dense activation",
            "Dropout Rate"
        ])
