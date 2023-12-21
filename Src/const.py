from os import environ

# Constants for data and file paths
FRUITS_PATH = './fruits/'              # Directory containing fruit data
GRAPHS_PATH = './graphs/'              # Directory for storing generated graphs
MODELS_PATH = './models/'              # Directory for storing trained models
STATS_PATH  = './statistics/'          # Directory for storing training statistics
PREPRC_PATH = './fruits_processed/'    # Directory containing preprocessed fruit images

# File paths for statistics
STATS_FILE = './statistics/models_statistics.csv'    # File storing detailed statistics for each training session
TEST_FILE  = './statistics/test_statistics.csv'      # File storing statistics for test sessions on selected models

# Debug mode
DEBUG = False   # Debug mode: Processes a subset of data for quicker testing

# Set TensorFlow log level based on debug mode
if not DEBUG:
    # Silence less relevant TensorFlow messages in non-debug mode
    environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
else:
    # Allow more detailed TensorFlow messages in debug mode
    environ['TF_CPP_MIN_LOG_LEVEL'] = "0"