import os
import cv2
import pandas
import numpy as np
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

    # Creates the statistics file that will hold trainig results
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


def verify_preprocessing_complete() -> None:
    '''
    Checks if the preprocessing was completed.\n
    Throws an error and stop the program execution if not.
    '''
    print("[INFO] Veryfying if preprocessing completed...")
    # s = str(IMAGE_SIZE)

    # # Tests for folders genenerated by preprocessor and csv files
    # conditions_satisfied = (os.path.exists(IMAGES_PATH) and
    #     os.path.exists(ANNONS_PATH) and
    #     os.path.exists(OUTPUTS_PATH+s) and
    #     os.path.exists(PREPRC_PATH+s+".csv") and 
    #     os.path.getsize(PREPRC_PATH+s+".csv")>0 and
    #     len(os.listdir(OUTPUTS_PATH+s))>=len(os.listdir(IMAGES_PATH)))

    # if not conditions_satisfied:
    #     print("[ERR] Preprocessing was not yet completed for {}x{} image size. Please run preprocessor.py before launching this code".format(s,s))
    #     exit(-1) 

    # # Checks if image shape match the ones requested by the CONST file
    # test_image_shape = cv2.imread(OUTPUTS_PATH+s+'/'+os.listdir(OUTPUTS_PATH+s)[0]).shape
    # if(test_image_shape != (IMAGE_SIZE, IMAGE_SIZE,3)):
    #     print("[ERR] Image size in const.py ({}) does not match preprocessed images' (folder: {}, test image: {}). Re-leaunch preprocesor to fix this".format(IMAGE_SIZE,s, test_image_shape[0]))
    #     exit(-1)

    print("[INFO] Success!")
    
def get_train_validation_test_data(model_settings: ModelSettings, training_settings: TrainingSettings):
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
