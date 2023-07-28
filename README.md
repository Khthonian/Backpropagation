# Backpropagation

This Python script implements a neural network for training and validating a set of input data. It aims to optimise the network's weights through backpropagation to minimise the squared error between the predicted outputs and the actual target outputs.

## Setup

Make sure you have Python installed on your system to run this script successfully. Additionally, the script requires the `matplotlib` module, which can be installed using the following command:

`pip install matplotlib`


## Description

The script consists of several functions to perform the training and validation of the neural network. Here is an overview of each function's role:

### 1. `loadData(fileName)`

This function reads data from the specified file and prepares it for training. It returns a list of tuples, where each tuple contains the input features and target outputs for a particular example.

### 2. Neural Network Initialisation

Before training, the script sets up the neural network with the following components:

- Learning rate (`learningRate`): A hyperparameter used to adjust the weights during training.
- Bias (`bias`): A fixed value added to the net calculation for each neuron.
- Weights for each neuron: Initial weights for five neurons (4, 5, 6, 7, and 8) in the network.

### 3. Sigmoid and Softmax Functions

The `sigmoid(x)` and `softmax(x, y)` functions compute the sigmoid and softmax activations, respectively, for the neural network.

### 4. Net Calculation Functions

The `netCalc(weights, inputData)` and `hiddenNetCalc(weights)` functions calculate the net input for neurons in the output and hidden layers, respectively.

### 5. Output Calculation

The `outputCalc(net)` function computes the output for a neuron using the sigmoid activation function.

### 6. Weight Backpropagation Functions

The `weightReset(error, weights, inputData)` and `hiddenWeightReset(error, weights, inputData)` functions adjust the weights of the neurons based on the calculated errors during backpropagation.

### 7. Training and Validation

The script performs training for a specified number of epochs (`epochMax`). It updates the network weights using the training samples and calculates the training loss. The validation loss is computed using a separate set of samples. The neural network's learning curve is plotted to visualise the loss reduction over epochs.

### 8. Testing

Finally, the neural network is tested using a set of test input data (`testInputData`). The output and softmax probabilities for the test data are displayed.

## Usage

To use the script, follow these steps:

1. Ensure the required modules are installed using `pip install matplotlib`.
2. Prepare your input data file in the following format: each line contains space-separated input features and target outputs (e.g., `0.1 0.2 0.3 0.4 0.5 0.6`).
3. Adjust hyperparameters such as the learning rate and initial weights, if necessary.
4. Run the script and provide the number of epochs for training when prompted.
5. The script will train the neural network and display the learning curve.
6. After training, the neural network will be tested on the provided test data, and the outputs and softmax probabilities will be shown.

Feel free to modify the script to suit your specific data and neural network architecture.
