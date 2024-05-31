# MNIST Logistic Regression

This repository contains an implementation of a Logistic Regression model to classify handwritten digits from the MNIST dataset. The model is built using Python with libraries such as NumPy, Pandas, Scikit-learn, and Matplotlib.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model](#model)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The MNIST dataset is a benchmark dataset in machine learning, containing 70,000 images of handwritten digits (0-9). This project implements a Logistic Regression model to classify these digits with high accuracy. Logistic Regression is a simple yet powerful algorithm for binary and multi-class classification tasks.

## Dataset

The MNIST dataset consists of:
- **60,000 training images**
- **10,000 test images**

Each image is a 28x28 pixel grayscale image, flattened into a 784-dimensional vector. The dataset is widely used for training and testing in the field of machine learning.

## Model

Logistic Regression is a linear model for binary classification. In this project, we extend it for multi-class classification using the one-vs-rest (OvR) strategy. The model predicts the probability of each class and assigns the class with the highest probability.

### Key Features:
- **Simple and efficient**: Suitable for large datasets
- **Interpretable**: Coefficients can be interpreted as the impact of features on the log-odds of the outcome
- **Scalable**: Can be extended to multi-class classification using strategies like one-vs-rest (OvR) or softmax regression

## Dependencies

The following dependencies are required to run the project:

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mnist-logistic-regression.git
   cd mnist-logistic-regression

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Usage

1. Run the script to train the model and evaluate its performance:
   ```bash
   python mnist_logistic_regression.py

2. The script will:

    - Load the MNIST dataset
    - Preprocess the data (flatten images and normalize pixel values)
    - Split the data into training and testing sets
    - Train the Logistic Regression model using the training data
    - Evaluate the model performance on the test data
    - Output evaluation metrics and visualizations

## Results

The model achieves high accuracy in classifying handwritten digits. Key results include:

  - Accuracy: Approximately 92%
  - Confusion Matrix: Visual representation of classification performance
  - Example Visualizations: Correctly and incorrectly classified digits


## Contributing
Contributions are welcome! Please fork this repository, make your changes, and submit a pull request with your improvements. Here are some areas where you could contribute:

  - Improving model accuracy
  - Adding more detailed visualizations
  - Implementing different machine learning algorithms
  - Enhancing data preprocessing steps

## License
This project is licensed under the MIT License. See the LICENSE file for details.
