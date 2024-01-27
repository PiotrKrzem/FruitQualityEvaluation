# Fruit Quality Evaluation - Convolutional Neural Network for Assessing Fruit Quality

This project is designed for individuals who want to assess fruit quality using different Convolutional Neural Network (CNN) models without delving into the code. The main method allows for easy customization of the network.

## Project Description

Utilize images of fruits to create a CNN classifier capable of evaluating the quality of a given fruit.

## Before Running

1. Download the following dataset: 
   [Fruit Quality Dataset](https://drive.google.com/drive/folders/12BZO95o4ZY7YpTlH9GuZd1S5D1mq7HUw?usp=sharing)
   Important! Please download entire *fruits* folder (not only *good_quality* or *bad_quality* folders)!
3. Extract downloaded *zip* and place its content explicitly in the source directory of the project under the name *fruits*
   It means that, the directory tree, after these operations should have the following directories:
   |--**fruits** <-- directory with pictures of fruits
   |  |--bad_quality
   |  |--good_quality
   |--src

4. In the command line, navigate to the project directory and run (preferably on a fresh Python environment):
   - Windows:
     `python -m pip install -r requirements.txt`
   - Linux/Mac:
     `python3 -m pip install -r requirements.txt`
   These commands will install all necessary dependencies


## Running the main code

In order to run the base code, which allows for constructing and testing CNNs, navigate to project directory and run the following command:
`python main.py`

By default, the command will:

1. Read the image files from the *./fruits* directory
2. Split the data into training, testing and validation data sets
3. Construct the custom CNN network with its default architecture
4. Train the network and saved it in the *./models* directory
5. Test the network on the test data set

In order to construct different network, the user have to change the main script by selecting different *BuiltInModel*
in lines 14 and 15. Available types include:
- **ALEXNET**
- **RESNET_PRETRAINED**
- **MINI** (custom CNN)

It is also possible to play with different configurations of network models and training parameters.
All configurations are placed within *./settings/builtin_models.py* and *./settings/builtin_training.py* files and can be changed from there.

However! It should be noted that instructions described above apply to the user which want to construct and train networks from scratch.
In order to simply test how pre-trained networks work on different images, please refer to the next section.

## Testing pre-trained models

In order to test pre-trained AlexNet, ResNet and custom CNN networks, navigate to the *./testing* directory.
Here, all of the pre-trained models of the networks have been placed.

In directory *./training/fruits_random_images* and *./training/fruits_original_images sample images have been placed, so that they can be used in testing.

In order to run tests of all networks, simply executed the *runner.ipynb* Jupyter Notebook. 
It will run and print the results of tests of all three networks.
