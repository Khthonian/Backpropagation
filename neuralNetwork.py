import matplotlib as plt
import math
import random

class Neuron:
    def __init__(self, nodeNumber, weights, input, output, bias=1):
        self.nodeNumber = nodeNumber
        self.weights = weights
        self.input = input
        self.output = output
        self.error = 0
        self.bias = bias

    def outputCalculation(self):
        net = self.weights[0] * self.bias
        for x in range(1, len(self.weights)):
            net += self.weights[x] * self.input[x - 1]
        self.output = 1 / (1 + math.exp(-net))
        return self.output
    
class Network:
    def __init__(self, input, truth, hiddenSize, learningRate=0.1):
        self.input = input
        self.truth = truth
        self.nodeCount = 0
        self.hiddenSize = hiddenSize
        self.hiddenLayer = self.populateNodes(hiddenSize, len(input[0]))
        # Output layer size should match truth[0] size
        self.outputLayer = self.populateNodes(len(truth[0]), hiddenSize)
        self.learningRate = learningRate

    def populateNodes(self, layerSize, numInputs):
        nodes = []
        for i in range(layerSize):
            # +1 for the bias weight
            weights = [random.uniform(-1, 1) for _ in range(numInputs + 1)]
            # Initialise input with zeros
            node = Neuron(f"Node {self.nodeCount}", weights, [0] * numInputs, 0)
            nodes.append(node)
            self.nodeCount += 1
        return nodes

    def forward(self, inputData):
        # Set input data to the hidden layer of the network
        for neuron in self.hiddenLayer:
            neuron.input = inputData

        # Forward propagation through the hidden layer
        hiddenOutput = [neuron.outputCalculation() for neuron in self.hiddenLayer]

        # Set hidden layer outputs as input to the output layer
        for neuron in self.outputLayer:
            neuron.input = hiddenOutput

        # Forward propagation through the output layer
        output = [neuron.outputCalculation() for neuron in self.outputLayer]
        return output

    def errorCalculation(self, truth, output):
        # Calculate error for the output layer
        for i in range(len(self.outputLayer)):
            neuron = self.outputLayer[i]
            neuron.output = output[i]  # Set the output value obtained from the forward step
            error = output[i] * (1 - output[i]) * (truth[i] - output[i])
            neuron.error = error

        # Calculate error for the hidden layer
        for i in range(len(self.hiddenLayer)):
            neuron = self.hiddenLayer[i]
            output = neuron.output
            error = output * (1 - output) * sum([self.outputLayer[j].weights[i + 1] * self.outputLayer[j].error for j in range(len(self.outputLayer))])
            neuron.error = error

    def backpropagate(self):
        # Update weights for the output layer
        for i in range(len(self.outputLayer)):
            neuron = self.outputLayer[i]
            deltaWeights = [self.learningRate * neuron.error * inputValue for inputValue in [1] + neuron.input]
            neuron.weights = [weight + delta for weight, delta in zip(neuron.weights, deltaWeights)]

        # Update weights for the hidden layer
        for i in range(len(self.hiddenLayer)):
            neuron = self.hiddenLayer[i]
            deltaWeights = [self.learningRate * neuron.error * inputValue for inputValue in [1] + neuron.input]
            neuron.weights = [weight + delta for weight, delta in zip(neuron.weights, deltaWeights)]

    def train(self, epochs):
        for epoch in range(epochs):
            for i in range(len(self.input)):
                inputData = self.input[i]
                output = self.forward(inputData)
                self.errorCalculation(self.truth[i], output)
                self.backpropagate()

            # After each epoch, print the weights
            print(f"Epoch {epoch + 1}")
            for layer in [self.hiddenLayer, self.outputLayer]:
                for neuron in layer:
                    print(f"{neuron.nodeNumber} Weights: {neuron.weights}")
            print("\n")
        
# Load data from file
def loadData(fileName):
    data = []
    with open(fileName) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            example = line.split('\t')
            if len(example) != 2:
                raise ValueError("Each line must contain exactly two tab-separated values.")
            features = [float(i) for i in example[0].strip().split(' ')]
            targets = [float(i) for i in example[1].strip().split(' ')]
            data.append((features, targets))
    return data

# Softmax function
def softmax(x, y):
    a = math.exp(x) / (math.exp(x) + math.exp(y))
    b = math.exp(y) / (math.exp(x) + math.exp(y))
    return {a, b}

# Example usage:
data = loadData("data-CMP2020M-item1-train.txt")

features = [x[0] for x in data]
truth = [x[1] for x in data]
test = [0.3, 0.7, 0.9]
hiddenLayerSize = 3
learningRate = 0.1

network = Network(features, truth, hiddenLayerSize, learningRate)

# Epoch Counter
epochs = int(input("How many epochs should I train for?: "))
network.train(epochs)

# Perform forward step for the first input
output = network.forward(test)
print("Output:", output[0])
softmaxOne, softmaxTwo = softmax(output[0], output[1])
print("Softmax: " + str(softmaxOne) + ", " + str(softmaxTwo))
