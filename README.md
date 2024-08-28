# Coding-Raja-Technologies-Internship-Task2

# Food Image Classification with PyTorch

This project implements a food image classification model using a pre-trained ResNet50 neural network. The model is fine-tuned on the **Food-101** dataset to classify images into 101 different categories of food.

## Features

- **Download Dataset**: Automatically download and extract the Food-101 dataset.
- **Train Model**: Fine-tune a pre-trained ResNet50 model on the Food-101 dataset.
- **Evaluate Model**: Test the model's accuracy on unseen data.
- **Predict Image Class**: Predict the class of a new food image using the trained model.
- **Visualize Predictions**: Display the predicted class along with the input image.

## Project Structure

- **main.py**: Main script to download the dataset, train the model, and evaluate its performance.
- **dataset.py**: Contains functions to download the dataset and create PyTorch data loaders.
- **model.py**: Defines the ResNet50 model architecture and modifies it for food classification.
- **train.py**: Script for training the model.
- **evaluate.py**: Functions for evaluating the model's accuracy and visualizing predictions.
- **predict.py**: Script to load the trained model and predict the class of a new food image.

## Prerequisites

- Python 3.x
- PyTorch
- Torchvision
- PIL (Python Imaging Library)
- Matplotlib

Install the required packages using pip:

```bash
pip install torch torchvision pillow matplotlib
```

## Dataset

The model is trained on the **Food-101** dataset. The dataset will be downloaded and extracted automatically by running the `main.py` script.

## How to Use

### 1. Download Dataset and Train the Model

To download the dataset, train the model, and evaluate its performance, run the following command:

```bash
python main.py
```

This script will:
- Download and extract the Food-101 dataset.
- Create training and testing data loaders.
- Fine-tune a pre-trained ResNet50 model on the Food-101 dataset.
- Save the trained model (`food101_model.pth`).

### 2. Predict the Class of a New Image

To predict the class of a new food image, run:

```bash
python predict.py
```

Edit the `image_path` variable in `predict.py` to point to the image you want to classify. The script will load the trained model and output the predicted class.

Example output:

```
The predicted class is: Apple Pie
```

### 3. Visualize Predictions

To visualize the predictions on a batch of test images, modify and use the `visualize_predictions` function in `evaluate.py`. This function will display the images along with the predicted labels.

### 4. Model Architecture

The `create_model` function in `model.py` fine-tunes a pre-trained ResNet50 model:
- **Freezing Layers**: All layers are frozen except for the final fully connected layer.
- **Custom Classifier**: The final layer is replaced with a custom classifier to output probabilities for 101 food categories.

### 5. Evaluation

After training, the model is evaluated on the test data, and the accuracy is displayed:

```
Accuracy: 85.76%
```

## Example Output

Sample training output:

```
Epoch 1/10, Loss: 0.5684
Epoch 2/10, Loss: 0.4532
...
```

Sample prediction output:

```
The predicted class is: Sushi
```

## Author

[Mayank Wadhwa](https://github.com/MmayankK21)

