from src.data.loader import *

from src.settings.model import *
from src.settings.training import *
from src.settings.builtin_models import *
from src.settings.builtin_training import *

from src.classification.model import *
from src.classification.train import *
from src.classification.test import *


def main():
    model_settings = get_builtin_model_settings(BuiltInModel.MINI)
    training_settings = get_builtin_training_settings(BuiltInTraining.MINI)

    # Creates graphs, models folders if missing
    create_required_files_and_folders_if_missing()

    # Load features and labels datasets into memory with selected features
    train, validation, test = get_train_validation_test_data(model_settings, training_settings)

    # Creates the model with specified structure and properties
    model = create_model(model_settings)

    # Compiles the model and prints its summary
    compiled_model = compile_model(model, training_settings)

    # Trains the model over given amount of epochs, tests data on validation dataset
    compiled_model, history = train_model(compiled_model, train, validation, training_settings)

    # Saves the accuracy & loss graphs, saves statistics to the CSV file
    model_path, graph_path = save_statistics(history, model_settings, training_settings)

    # Saves graphs that display accuracy and loss
    save_graph(history, graph_path, model_settings, training_settings)

    # Saves model into the folder with name specified below (adds unique id to the file name)
    save_model(compiled_model, model_path)

    # Testing trained model on previously prepared data
    test_model(compiled_model, test)
    

if __name__ == "__main__":
    main()
