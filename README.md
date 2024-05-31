### MNIST Logistic Regression
This repository contains an implementation of a Logistic Regression model to classify handwritten digits from the MNIST dataset. The model is built using Python with libraries such as NumPy, Pandas, Scikit-learn, and Matplotlib.

<p>Table of Contents<br>
Introduction<br>
Dataset<br>
Model<br>
Dependencies<br>
Installation<br>
Usage<br>
Results<br>
Contributing</p>


#### Introduction<br>
The MNIST dataset is a benchmark dataset in machine learning, containing 70,000 images of handwritten digits (0-9). This project implements a Logistic Regression model to classify these digits with high accuracy.

Dataset<br>
The MNIST dataset consists of 60,000 training images and 10,000 test images. Each image is a 28x28 pixel grayscale image, flattened into a 784-dimensional vector.

Model<br>
Logistic Regression is a linear model for binary classification. In this project, we extend it for multi-class classification using the one-vs-rest (OvR) strategy. The model predicts the probability of each class and assigns the class with the highest probability.

#### Dependencies<br>

Python 3.x
NumPy
Pandas
Scikit-learn
Matplotlib


#### Installation<br>

Clone the repository:<br>

>bash
>git clone https://github.com/your-username/mnist-logistic-regression.git
>cd mnist-logistic-regression

Create and activate a virtual environment (optional but recommended):<br>

>bash
>python -m venv env
>source env/bin/activate  # On Windows: env\Scripts\activate

Install the required dependencies:<br>

>bash
>pip install -r requirements.txt

#### Usage<br>
Run the script to train the model and evaluate its performance:

>bash
>python mnist_logistic_regression.py

The script will load the MNIST dataset, train the Logistic Regression model, and output evaluation metrics and visualizations.

#### Results<br>
The model achieves high accuracy in classifying handwritten digits. Key results include:

#### Accuracy: Approximately 96%<br>
Confusion Matrix: Visual representation of classification performance.
Example visualization of correctly and incorrectly classified digits.

#### Contributing<br>
Contributions are welcome! Please fork this repository and submit a pull request with your improvements.
