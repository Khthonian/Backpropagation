# Neural Network

This Python script implements a neural network for training and validating a set of input data. It aims to optimise the network's weights through backpropagation to minimise the squared error between the predicted outputs and the actual target outputs.

## Description

This neural network implementation uses a single hidden layer and output layer. The number of neurons in the hidden layer is defined by the user; the default value of neurons is 3. Each neuron is defined by the `Neuron` class, which allows internal calculation of outputs.

The `Network` class is used to define the entire structure of the neural network and the neurons within. This class stores various information, including lists of the neurons in both the hidden and output layers. The methods within this class are responsible for driving the neural network tasks, such as the forward pass, the error calculation, and the backpropagation.

Data can either be given to the network as a list or loaded into the network from a text file, also as a list.

## Usage

To use the script, follow these steps:

1. Prepare your input data file in the following format: each line contains space-separated input features and target outputs. An example can be seen below, from `data-CMP2020M-item1-train.txt`:
```
0.50 1.00 0.75	1 0
1.00 0.50 0.75	1 0
1.00 1.00 1.00	1 0
-0.01 0.50 0.25	0 1
0.50 -0.25 0.13	0 1
0.01 0.02 0.05	0 1
```
2. Adjust hyperparameters such as the learning rate and initial weights, if necessary.
3. Run the script and provide the number of epochs for training when prompted.
4. The script will train the neural network.
5. After training, the neural network will be tested on the provided test data, and the outputs and softmax probabilities will be shown. This is done using the `forward` method from the `Network` class.

The script is designed such that the user can adjust the parameters to best fit their data and preferred architecture.

## Miscellaneous

This implementation is an improvement upon a prior university assignment. To see the original code, `BackpropAlgo.py`, submitted in the assignment, click [here](https://github.com/Khthonian/Neural-Network/releases/tag/v1.0).