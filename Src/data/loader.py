import os
import tensorflow as tf

from src.const import *
from src.settings.model import *
from src.settings.training import *

def create_required_files_and_folders_if_missing() -> None:
    '''
    Checks if all the required folders and files are created
    and if not, makes them and fills with starting data
    '''

    # Creates folders for graphs and models
    if(not os.path.exists(GRAPHS_PATH)):
        os.mkdir(GRAPHS_PATH)
    if(not os.path.exists(MODELS_PATH)):
        os.mkdir(MODELS_PATH)
    if(not os.path.exists(STATS_PATH)):
        os.mkdir(STATS_PATH)

    # Creates the statistics file that will hold training results
    if(not os.path.exists(STATS_FILE)):
        with open(STATS_FILE, 'w') as file:
            file.write(";".join([
                "Accuracy",
                "Loss",
                ModelSettings.get_header(),
                TrainingSettings.get_header(),
                "\n"
            ]))
    
    if(not os.path.exists(TEST_FILE)):
        with open(TEST_FILE, 'w') as file:
            file.write(";".join([
                "Accuracy",
                "Loss",
                ModelSettings.get_header(),
                "\n"
            ]))
    
def get_train_validation_test_data(model_settings: ModelSettings, training_settings: TrainingSettings):
    '''
    Method split the dataset into train, validation and testing.
    '''
    train, test = tf.keras.utils.image_dataset_from_directory(
        FRUITS_PATH,
        validation_split=training_settings.validation_split,
        subset="both",
        seed=123,
        image_size=(model_settings.input_size, model_settings.input_size),
        batch_size=training_settings.batch_size
    )

    num_batches = tf.data.experimental.cardinality(train).numpy()
    batches_to_take = int(training_settings.validation_split * num_batches)

    validation = train.take(batches_to_take)
    train = train.skip(batches_to_take)

    AUTOTUNE = tf.data.AUTOTUNE

    train = train.prefetch(buffer_size=AUTOTUNE)
    validation = validation.prefetch(buffer_size=AUTOTUNE)
    test = test.prefetch(buffer_size=AUTOTUNE)

    return train, validation, test
