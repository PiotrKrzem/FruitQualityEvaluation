# DoglassifierAI - Convolutional Neural Network to classify 200 dog breeds

For demonstrations, this solution comes with AlexGG, my final version of the classifier with accuracy 50% higher than most online sources!

This project is perfect for people who need to test a lot of different CNN models without delving into the code - the main method allows to customize the network with ease!

## Project description

Use dogs of different breeds to create a CNN classifier capable of determining a given dog's breed.

![Final Result 1](demos/example_output1.png)

* [X] Train the first model
* [X] Add a method of testing different CNN networks quickly
* [X] Train multiple online models and test their ability to categorize breeds
* [X] Obtain better results
* [X] Implement demonstration methods

## Before Running

1. Download the following dataset:
   [Stanford Dogs Dataset](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset)
2. Extract the folders: *Images* from *images* directory and *Annotation* from *annotations*
3. Place extracted folders to *data* folder
4. In the command line, navigate to the folder directory and run (preferably on a fresh python environment):
   Windows:
   `python -m pip install -r requirements.txt`
   Linux/Mac:
   `python3 -m pip install -r requirements.txt`

## Data Preprocessing and Augmentation, Data Split

In order to reduce the chance of overfitting and improve the accuracy on the test set, the solution comes with a preprocessing script. The code is designed to reduce the images' size to reduce memory usage, augment the data using the flip, blur, x- and y- offsets, and generate a test set (20% of the data)

**Data split for this project: 70% training, 10% validation, 20% testing**

To perform preprocessing (**MANDATORY TO LAUNCH DEMO**)

1. Run preprocessor.py
   `python preprocessor.py`
2. When prompted for image size, leave empty for default (64). Alternatively, you can use any other - remeber to change IMAGE_SIZE constant in *Src/const.py*

## Tested Models

Base models of this project come from the suggestions from this online source:

[Dog breed classification using Deep Learning concepts](https://towardsdatascience.com/dog-breed-classification-using-deep-learning-concepts-23213d67936chttps://)

From these sources I have decided to use the VGG19 (slightly size-reduced), AlexNet and another suggested Custom Network from Udemy course. Here are the models represented graphically:

**Udemy:**

![](demos/udemy.png)

**AlexNet:**

![](demos/alexnet.png)

**VGG19:**

![](demos/vgg19.png)

The best results obtained from the first run for these networks was AlexNet, which scored 22% accuracy that plateued after 20 epochs:

![](demos/best_results.png)

## Solution Description

There are two scripts used to train and test the network.

*main.py* contains a list of settings that propagate throughtout the program to construct and train the network. Below are the settings used to construct the AlexGG, a mix between VGG19 and AlexNet - the final best performing network. In the script there are multiple comments that describe the usage of every value shown below.

```
settings = ModelSettings(
    convolution_layers = [  
                [16,16],  
                [32,32],
                [64,64]
    ],
    convolution_activation = ActivationType.relu,
    convolution_sizes = [   
                3,
                3,
                3
    ],  
    middle_layer = MiddleLayerType.max_pool,
    dense_layers = [  
                1024,
                1024
    ],
    dense_activation = ActivationType.relu,  
    dropout_rate = 0.2,
    optimizer = OptimizerType.adam,   
    learning_rate = 0.001,  

    epochs = 20, 
    batch_size = 2048,  
  

    model_name = 'alexGG',  
    validation_split = 0.1,   
    verbose = 1,   
    print_summary = True   
)
```

After every model finished training, the following occurs:

- Graph of Accuracy/Loss is saved to *Graphs* folder
- Model is saved to the *Models*  folder
- *models_statistics.csv* is updated to contain a new entry, which contains whole architescture of the model, as well as final statistics as a csv string separated by a ';' symbol. The csv file contains headers that describe the values for each column. It is safe to delete this file - it will be generated with headers automatically.

  **Headers:**
  Model path;Graph path;Accuracy;Loss;Image dim;Convolution layers;Convolution activation;Convolution sizes;Middle layer;Dense layers;Dense activation;Dropout rate;Optimizer;Learning rate;Epochs;Batch size;

  **Example saved CSV string with data:**
  ./Models/alexy-64x64-acc 0.299-id 1643408372.9734535.h5;./Graphs/graph-acc 0.299-1643408372.9734535.png;0.299;2.928;64;[[16, 16], [32, 32], [64, 64]];relu;[3, 3, 3];max_pool;[1024, 1024];relu;0.2;adam;0.001;20;2048;

**In order to test/demonstrate the results of training**, *demonstrator.py* can be used. This script alllows to perform one of the two operations:

1. Input any image and predict the dog breed
   The console will output the dog breed alongside the score, and a new file *output.png* will be created that contains the image alongside some examples of images of predicted breed (See Results Section)
2. Test the network on the test dataset
   After the test completes, a new CSV file *test_set_statistics.csv* is created that contains the results. For every breed, there is information about the correct predictions, incorrect predictions, accuracy, most commonly associated incorrect breed, and count of associations to that incorrect breed.

   **Headers:**
   Doggo;Correct guesses;Incorrect guesses;Accuracy;Most commonly mistaken as;Count of mistakes for this breed;

   **Example saved CSV string with data:**
   sealyham terrier;25;15;0.625;samoyed;4

## Results - Demonstration for AlexGG

The best result achieved an average accuracy of 30% on the test set, which is around 50% better than the popular solutions available online, such as the one mentioned in the *Tested Models*  section:

[Dog breed classification using Deep Learning concepts](https://towardsdatascience.com/dog-breed-classification-using-deep-learning-concepts-23213d67936chttps://)

After 20 epochs, the accuracy plateued at 30.9% with a low loss:

![](demos/alexgg.png)

Example demonstrations:

![Final Result 3](demos/example_output3.png)

![Final Result 2](demos/example_output2.png)

Result of testing of AlexGG network is available in the *test_set_statistics.csv*
