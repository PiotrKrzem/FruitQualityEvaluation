# Fruit Quality Evaluation - Convolutional Neural Network for Assessing Fruit Quality

This project is designed for individuals who want to assess fruit quality using different Convolutional Neural Network (CNN) models without delving into the code. The main method allows for easy customization of the network.

## Project Description

Utilize images of fruits to create a CNN classifier capable of evaluating the quality of a given fruit.

## Before Running

1. The dataset and pre-trained models can be found under the following directory: 
   [Fruit Quality Dataset and Models](https://drive.google.com/drive/folders/1lD_cLQZnzv_IjkSNbMOLxdK1mQa86tEC?usp=sharing)
   If you would like to download dataset + pre-trained then download the entire directory
   However, if you need only pre-trained models (testing data is already included within the project), then simply download files *alexnet.h5*, *mini.h5* and *resnet_based.h5*.
2. Extract downloaded *zip*
3. To add input data, place the directory *fruits* explicitly in the source directory of the project.
   It means that, the directory tree, after these operations should have the following structure:
   |--**fruits** <-- directory with pictures of fruits
   |  |--bad_quality
   |  |--good_quality
   |--src
   |--testing
4. The files *alexnet.h5*, *mini.h5* and *resnet_based.h5* are pre-trained models used in testing, therefore they should be placed within *./testing* directory.
   It means that, afterwards *./testing* directory should look as follows:
   |--fruits_original_dataset
   |--fruits_random_dataset
   |--alexnet.h5
   |--mini.h5
   |--resnet_based.h5
   |--runner.ipynb

   Important! All 3 models must be placed in *./testing* directory in order for the tests to work! 
5. In the command line, navigate to the project directory and run (preferably on a fresh Python environment):
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
in line 14. Available types include:
- **ALEXNET**
- **RESNET_PRETRAINED**
- **MINI** (custom CNN)

It is also possible to specify different training setting, by using different *BuiltInTraining* in line 15.
Available types include:
- **DEFAULT**
- **MINI**
- **RESNET**

It is also possible to play with different configurations of network models and training parameters.
All configurations are placed within *./settings/builtin_models.py* and *./settings/builtin_training.py* files and can be changed from there.

However! It should be noted that instructions described above apply to the user which want to construct and train networks from scratch.
In order to simply test how pre-trained networks work on different images, please refer to the next section.

## Testing pre-trained models

In order to test pre-trained AlexNet, ResNet and custom CNN networks, navigate to the *./testing* directory.
Here, all of the pre-trained models of the networks should been placed.

In directory *./training/fruits_random_images* and *./training/fruits_original_images sample images have been placed, so that they can be used in testing.

In order to run tests of all networks, simply executed the *runner.ipynb* Jupyter Notebook. 
It will run and print the results of tests of all three networks.
