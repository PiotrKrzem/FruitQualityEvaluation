from src.settings.model import *

def get_builtin_model_settings(model: BuiltInModel):
    '''
    Method returns builtin settings of the model based on the indicated model type.
    '''
    if model == BuiltInModel.ALEXNET:
        output =  _get_alexnet()
    elif model == BuiltInModel.RESNET_PRETRAINED:
        output = _get_resnet_pretrained()
    elif model == BuiltInModel.MINI:
        output = _get_mini()
    elif model == BuiltInModel.UNINITIALIZED:
        raise Exception("Builtin model UNINITIALIZED should not be used")
    else:
        raise Exception(f"Unknown builtin model {model.name}")

    # Model name set after the settings constructor to avoid name check
    output.model_name = model.value
    return output

def _get_alexnet():
    return ModelSettings(
        input_size=32,
        convolution_layers=[
            [16, 16],
            [32, 32],
            [64, 64]
        ],
        convolution_activation=ActivationType.RELU,
        convolution_sizes=[
            3,
            3,
            3
        ],
        middle_layer=MiddleLayerType.MAX_POOL,
        dense_layers=[
            1024,
            1024
        ],
        dense_activation=ActivationType.RELU,
        output_activation=ActivationType.SIGMOID,
        dropout_rate=0.2,
        model_name=BuiltInModel.UNINITIALIZED,
        builtin=True,
        pretrained=False    
    )

def _get_mini():
    return ModelSettings(
        input_size=32,
        convolution_layers=[
            [16, 16],
            [32, 32],

        ],
        convolution_activation=ActivationType.RELU,
        convolution_sizes=[
            3,
            3
        ],
        middle_layer=MiddleLayerType.MAX_POOL,
        dense_layers=[
            1024,
            1024,
        ],
        dense_activation=ActivationType.RELU,
        output_activation=ActivationType.SIGMOID,
        dropout_rate=0.1,
        model_name=BuiltInModel.UNINITIALIZED,
        builtin=True,
        pretrained=False    
    )

def _get_resnet_pretrained():
    return ModelSettings(
        input_size=256,
        model_name=BuiltInModel.UNINITIALIZED,
        builtin=True,
        pretrained=True
    )