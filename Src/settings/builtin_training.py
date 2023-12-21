from src.settings.training import *

def get_builtin_training_settings(training: BuiltInTraining):
    if training == BuiltInTraining.DEFAULT:
        output =  _get_default()
    elif training == BuiltInTraining.MINI:
        output =  _get_mini()
    elif training == BuiltInTraining.UNINITIALIZED:
        raise Exception("Builtin training UNINITIALIZED should not be used")
    else:
        raise Exception(f"Unknown builtin training {training.name}")

    return output

def _get_default():
    return TrainingSettings(
        optimizer=OptimizerType.ADAM,
        learning_rate=0.001,
        epochs=20,
        batch_size=256,
        validation_split=0.2,
        verbose=True,           
        print_summary=True             
    )

def _get_mini():
    return TrainingSettings(
        optimizer=OptimizerType.ADAM,
        learning_rate=0.001,
        epochs=10,
        batch_size=256,
        validation_split=0.2,
        verbose=True,           
        print_summary=False             
    )