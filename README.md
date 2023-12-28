# Fruit Quality Evaluation - Convolutional Neural Network for Assessing Fruit Quality

For demonstrations, this solution includes AlexGG, my final version of the classifier, with an accuracy 50% higher than most online sources!

This project is designed for individuals who want to assess fruit quality using different Convolutional Neural Network (CNN) models without delving into the code. The main method allows for easy customization of the network.

## Project Description

Utilize images of fruits to create a CNN classifier capable of evaluating the quality of a given fruit.

![Final Result 1](demos/example_output1.png)

- [X] Train the first model
- [X] Add a method for testing different CNN networks quickly
- [] Train multiple online models and test their ability to categorize fruit quality
- [] Achieve better results
- [] Implement demonstration methods

## Before Running

1. Download the following dataset:
   [Fruit Quality Dataset](https://www.example.com/fruit-quality-dataset)
2. Extract the folders: *Images* from the *images* directory and *Annotation* from *annotations*
3. Place the extracted folders into the *data* folder
4. In the command line, navigate to the folder directory and run (preferably on a fresh Python environment):
   - Windows:
     `python -m pip install -r requirements.txt`
   - Linux/Mac:
     `python3 -m pip install -r requirements.txt`

## Data Preprocessing and Augmentation, Data Split

To reduce the chance of overfitting and improve accuracy on the test set, the solution includes a preprocessing script. The code is designed to resize images to reduce memory usage, augment data using flip, blur, x- and y-offsets, and generate a test set (20% of the data).

**Data split for this project: 70% training, 10% validation, 20% testing**

To perform preprocessing (**MANDATORY TO LAUNCH DEMO**)

1. Run preprocessor.py
   `python preprocessor.py`
2. When prompted for image size, leave empty for default (64). Alternatively, you can use any other - remember to change the IMAGE_SIZE constant in *Src/const.py*

