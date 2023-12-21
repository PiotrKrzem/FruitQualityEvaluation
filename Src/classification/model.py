import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from time import time
from typing import Tuple
from tensorflow.keras import layers

from src.const import *
from src.settings.model import *
from src.settings.training import *

def create_model(model_settings: ModelSettings) -> tf.keras.models.Sequential:
    """Creates a custom model based n input arguments.
    Model's structure: Input layer -> Conv2Ds -> Middle layer -> Dense layers -> Output layer.
    Output layer is a Dense layer with softmax activation.
    """

    if not model_settings.pretrained: # F, [T, F]
        return construct_model(model_settings)
    elif model_settings.pretrained and not model_settings.builtin: # T, F
        return load_model(model_settings)
    else:   # T, T
        return get_builtin_model(model_settings)

def construct_model(model_settings: ModelSettings) -> tf.keras.models.Sequential:
    """Constructs a custom model based n input arguments.
    Model's structure: Input layer -> Conv2Ds -> Middle layer -> Dense layers -> Output layer.
    Output layer is a Dense layer with softmax activation.
    """

    model = tf.keras.Sequential()

    # Input layer
    model.add(layers.Input(shape=[model_settings.input_size, model_settings.input_size, 3]))

    # Convolutional layers builder - Conv2D layer(s) followed by MaxPool2D. 
    for idx,layer in enumerate(model_settings.convolution_layers):
        s = model_settings.convolution_sizes[idx]

        if not (isinstance(layer, list) or isinstance(layer, tuple)):
            layer = [layer]

        for l in layer:
            model.add(layers.Conv2D(
                filters = l,
                kernel_size = (s, s), 
                padding = 'same',
                activation = model_settings.convolution_activation.value
            ))

        # Adds MaxPool2D after requested Conv2D layers
        # Does not add after the last layer, as some networks 
        # use different layer as a connector
        if idx < len(model_settings.convolution_layers) - 1:
            model.add(layers.MaxPool2D(
                pool_size = (2,2), 
                strides = (2,2)
            ))
        
    # Middle layer - connection between Conv2Ds and Dense
    # Flattens the shape from 4D to 2D
    if model_settings.middle_layer == MiddleLayerType.AVERAGE_POOL:
        model.add(layers.GlobalAveragePooling2D(
            data_format='channels_last'
        ))
    elif model_settings.middle_layer == MiddleLayerType.MAX_POOL:
        model.add(layers.MaxPool2D(
            pool_size = (2,2), 
            strides = (2,2)
        ))
        model.add(layers.Flatten(
            data_format='channels_last'
        ))
    elif model_settings.middle_layer == MiddleLayerType.FLATTEN:
        model.add(layers.Flatten(
            data_format='channels_last'
        ))

    # Dense layers builder - Dense layer(s) followed by Dropout.
    for idx,layer in enumerate(model_settings.dense_layers):
        if not (isinstance(layer, list) or isinstance(layer, tuple)):
            layer = [layer]

        for l in layer:
            model.add(layers.Dense(
                units = l,
                activation = model_settings.dense_activation.value
            ))

        # Adds dropout after adding requested dense layers
        model.add(layers.Dropout(
            rate = model_settings.dropout_rate
        ))
    
        
    # Output layer - dense layer that will output a value with quality prediction
    model.add(layers.Dense(
        1,
        activation='softmax'
    ))
    
    return model

def get_builtin_model(model_settings: ModelSettings) -> tf.keras.models.Sequential:
    if model_settings.model_name == BuiltInModel.ALEXNET.value:
        return construct_model(model_settings)

    elif model_settings.model_name == BuiltInModel.RESNET_PRETRAINED.value:
        inputs = layers.Input(shape=(model_settings.input_size, model_settings.input_size, 3))

        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
        ])(inputs)

        normalisation = tf.keras.applications.resnet.preprocess_input(data_augmentation)

        resnet = tf.keras.applications.resnet.ResNet152(
            include_top=False,
            weights='imagenet',
            input_shape=(model_settings.input_size, model_settings.input_size, 3))
        resnet.trainable = False

        convolutions = resnet(normalisation, training = False)

        flatten = layers.GlobalAveragePooling2D()(convolutions)
        outputs = layers.Dense(1, activation = 'sigmoid')(flatten)

        return tf.keras.Model(inputs, outputs)

def save_model(model:tf.keras.Sequential, model_name: str) -> None:
    '''
    Saves the model into the MODELS folder.
    '''

    model.save(model_name)
    print("[INFO] Saved model at {}".format(model_name))


def load_model(model_settings: ModelSettings):
    '''
    Loads the model based on the name.
    '''
    for path in os.listdir(MODELS_PATH):
        if path.startswith(model_settings.model_name):
            model = tf.keras.models.load_model(path)
            model.summary()
            return model
    raise Exception(f"Model named {model_settings.model_name} not found in models folder")

def make_a_guess(model:tf.keras.Sequential, images:np.array):
    '''
    By using a given trained model, predicts the output(s) for the input image(s).

    Output is a 2D array with shape: [images_count, 1]
    '''
    if len(images.shape) == 3:
        images = np.reshape(images, (1, -1, -1, 3))

    return model(images, training = False)
