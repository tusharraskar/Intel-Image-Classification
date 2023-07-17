# Image Classification 

This project implements an image classifier using a pre-trained EfficientNetB4 model. The model has been fine-tuned for classification on a new dataset containing images of different landscapes. The aim of the project is to correctly classify images into one of the six categories: "buildings", "forest", "glacier", "mountain", "sea", "street".

## Table of Contents
- [Project Setup](#project-setup)
- [Model Architecture](#model-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Training and Evaluation](#training-and-evaluation)
- [Fine-tuning](#fine-tuning)
- [Running the Code](#running-the-code)

## Project Setup
To set up the project, follow the steps below:

1. Clone the repository: `$ git clone <repo-url>`
2. Navigate into the project directory: `$ cd Image-Classification`
3. Install the required dependencies: `$ pip install -r requirements.txt`

## Steps to run the project
- Clone the repository to your local machine.

- Ensure that you have all the required libraries installed. You can install them using the following command:


`pip install -r requirements.txt`
- Open the image_classification.ipynb file in Jupyter Notebook.

- Update the TRAIN_PATH and TEST_PATH in the Jupyter notebook to point to your local directories containing the training and testing images respectively.

- Run all the cells in the notebook.


## Model Architecture
The model architecture is based on the EfficientNetB4 architecture, a powerful pre-trained model available in the tf.keras.applications module. The model was used without the top (classification) layers, which were replaced with a GlobalAveragePooling2D layer and a custom Dense layer to perform the classification task.

The final architecture is as follows:

- Input Layer: Receives the input images with dimensions 224 x 224 x 3.
- Data Augmentation Layer: Randomly transforms the input images to augment the training data.
- Base Model (EfficientNetB4): Extracts features from the input images. The base model is non-trainable during the feature extraction phase but is fine-tuned during the fine-tuning phase.
- Global Average Pooling Layer: Reduces the spatial dimensions of the input.
- Output Layer: A Dense layer with 6 units (one for each class) and a softmax activation function to output the class probabilities.

## Data Preprocessing
The data preprocessing steps include resizing the images to the required dimensions (224 x 224), normalizing the pixel values to [0, 1], and one-hot encoding the labels for the multi-class classification task. The image_dataset_from_directory function provided by TensorFlow was used to load the images from the directories.

## Training and Evaluation
The model was trained for 50 epochs using the Adam optimizer with a learning rate of 0.0001 and a batch size of 32. The performance of the model was evaluated on a separate test dataset. Various metrics, including accuracy, precision, recall, and F1-score, were computed to assess the model's performance. Additionally, a confusion matrix was plotted to visualize the model's performance across the different classes.

## Fine-tuning
After the initial training phase, the model was fine-tuned by unfreezing the base model and training it again with a lower learning rate. This fine-tuning phase allowed the model to adjust its pre-trained weights to better fit the new data.

## Running the Code
To run the code:

1. Open a terminal and navigate to the project directory.
2. Run the Python script: `$ python image_classification.py`

The script will load the data, train the model, evaluate its performance, fine-tune it, and finally display the model's performance metrics and confusion matrix.

## Dependencies
The following Python packages are required to run this code:

- pandas==1.3.4
- numpy==1.21.4
- matplotlib==3.5.1
- seaborn==0.11.2
- tensorflow==2.8.0
- tensorflow-addons==0.15.0
- tqdm==4.62.3
- warnings==1.2.0
- sklearn==0.24.2
- joblib==1.1.0
- mlxtend==0.18.0

## Future Improvements
The performance of the model can potentially be improved by using a larger dataset, applying additional data augmentation techniques, or using a more complex model architecture
