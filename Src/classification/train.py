import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from time import time
from typing import Tuple
from tensorflow.keras import layers

from src.const import *
from src.settings.model import *
from src.settings.training import *


def compile_model(model:tf.keras.Sequential, model_settings: ModelSettings, training_settings: TrainingSettings) -> tf.keras.models.Sequential:
    '''
    Compiles the model with selected optimizer (adam/SGD/RMSProp) and learning rate.\n
    Uses 'accuracy' as a measure statistic, Categorical Cross Entropy loss function. 
    '''

    if(training_settings.optimizer==OptimizerType.ADAM):
        opt = tf.keras.optimizers.Adam(learning_rate=training_settings.learning_rate)
    elif(training_settings.optimizer==OptimizerType.SGD):
        opt = tf.keras.optimizers.SGD(learning_rate=training_settings.learning_rate)
    elif(training_settings.optimizer==OptimizerType.RMS_PROP):
        opt = tf.keras.optimizers.RMSprop(learning_rate=training_settings.learning_rate)

    model.compile(
        optimizer = opt,
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics = ['accuracy']
    )

    # Prints summary of the model, mostly for debug
    if training_settings.print_summary: 
        model.summary()

    return model

def train_model(model:tf.keras.Sequential, 
                train_dataset: tf.data.Dataset,
                validation_dataset: tf.data.Dataset,
                model_settings: ModelSettings,
                training_settings: TrainingSettings):
    '''
    Trains the model using provided images and labels.
    '''

    if(training_settings.verbose == 0): 
        print("[INFO] Training in progress...")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=training_settings.epochs,
        batch_size=training_settings.batch_size,
        verbose=training_settings.verbose
    )

    return (model, history)

def save_statistics(history: tf.keras.callbacks.History, model_settings: ModelSettings, training_settings: TrainingSettings):
    '''
    Loads training statistics from the model, and saves them to STATS_FILE.
    '''
    # Extract interesting statistics
    rounded_acc = round(history.history['val_accuracy'][-1],3)
    rounded_loss = round(history.history['val_loss'][-1],3)

    id = str(time())
    # Generate names for the model and object
    model_name = MODELS_PATH + f"{model_settings.model_name}_{model_settings.input_size}_id{id}.h5"
    graph_name = GRAPHS_PATH + f"graph_{model_settings.model_name}_{model_settings.input_size}_id{id}.png"

    # Write statistics to file
    with open(STATS_FILE, 'a') as file:
        file.write(";".join([
            f"{rounded_acc:.3f}",
            f"{rounded_loss:.3f}",
            str(model_settings),
            str(training_settings),
            "\n"
        ]))

    # Returns generated names
    return (model_path, graph_path)

def save_graph(history: tf.keras.callbacks.History, graph_name: str, model_settings: ModelSettings, training_settings: TrainingSettings) -> None:
    '''
    Generates graphs of accuracy and loss for a given training session,
    and saves them to GRAPHS folder.
    '''

    # Extract interesting statistics
    acc  = history.history['accuracy']
    loss = history.history['loss']
    val_acc  = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    epochs_range = range(training_settings.epochs)

    
    plt.figure(figsize=(8, 8))

    # Creates 2 plots on one graph
    # First one represents the changing accuracy over epochs
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.title('Accuracy')

    # Second one represents the changing loss over epochs
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.title('Loss')

    # Saves the figure to GRAPHS folder
    plt.savefig(graph_name)
