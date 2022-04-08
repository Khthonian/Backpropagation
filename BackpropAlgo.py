import matplotlib.pyplot as plt
import math

# Load Data Function
def loadData(fileName):
    data = []
    with open(fileName) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            example = line.split('\t')
            features = [float(i) for i in example[0].strip().split(' ')]
            targets = [float(i) for i in example[1].strip().split(' ')]
            data.append((features, targets))
    return data

# Epoch Counter
epochs = 0
epochMax = int(input("How many epochs should I train for?: "))

# Graph Coordinate
graphPlot = []

# Learning Rate
learningRate = 0.1

# Bias
bias = 1

# Weights
neuron4Weights = [0.9, 0.74, 0.8, 0.35]
neuron5Weights = [0.45, 0.13, 0.4, 0.97]
neuron6Weights = [0.36, 0.68, 0.1, 0.96]
neuron7Weights = [0.98, 0.35, 0.5, 0.9]
neuron8Weights = [0.92, 0.8, 0.13, 0.8]

# Input and Output
inputData = loadData("data-CMP2020M-item1-train.txt")
testInputData = [0.3, 0.7, 0.9]

# List for hidden layer inputs
hiddenInputData = [1, 0, 0, 0]

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Softmax Function
def softmax(x, y):
    return math.exp(x) / (math.exp(x) + math.exp(y))

# Net Calculation
def netCalc(weights, inputData):
    net = 0
    net += weights[0] * bias
    for x in range(1, len(weights) - 1):
        net += weights[x] * inputData[x]
    return net

# Hidden Layer Net Calculation
def hiddenNetCalc(weights):
    net = 0
    net += weights[0] * bias
    for x in range(1, len(weights) - 1):
        net += weights[x] * hiddenInputData[x]
    return net

# Output Calculation
def outputCalc(net):
    output = sigmoid(net)
    return output

# Weight Backpropragation
def weightReset(error, weights, inputData):
    deltaWeights = []
    deltaWeights.append(learningRate * error * bias)
    for x in range(1, len(weights) - 1):
        deltaWeights.append(learningRate * error * float(inputData[x]))
    weights[0] += deltaWeights[0]
    for x in range(1, len(weights) - 1):
        weights[x] += deltaWeights[x]

# Hidden Weight Backpropragation
def hiddenWeightReset(error, weights, inputData):
    deltaWeights = []
    deltaWeights.append(learningRate * error * bias)
    for x in range(1, len(weights) - 1):
        deltaWeights.append(learningRate * error * float(hiddenInputData[x]))
    weights[0] += deltaWeights[0]
    for x in range(1, len(weights) - 1):
        weights[x] += deltaWeights[x]

# Output the weights
def printWeights(interationWeights):
    neuronLabels = [4, 5, 6, 7, 8]
    for x in range(0, 5):
        print("Weights for neuron " + str(neuronLabels[x]) + ":")
        print(str(iterationWeights[x]))
        print("")


# Train the neural network
while epochs < epochMax:
    trainSamples = [0, 1, 2]
    for x in trainSamples:
        # Determine expected outputs
        expOutputN7 = inputData[x][1][0]
        expOutputN8 = inputData[x][1][1]

        # Determine inputs
        activeInputData = inputData[x][0]
        
        # Neuron 4
        neuron4Net = netCalc(neuron4Weights, activeInputData)
        neuron4Output = outputCalc(neuron4Net)

        # Neuron 5
        neuron5Net = netCalc(neuron5Weights, activeInputData)
        neuron5Output = outputCalc(neuron5Net)

        # Neuron 6
        neuron6Net = netCalc(neuron6Weights, activeInputData)
        neuron6Output = outputCalc(neuron6Net)

        # Update inputs from the hidden layer
        hiddenInputData[1] = neuron4Output
        hiddenInputData[2] = neuron5Output
        hiddenInputData[3] = neuron6Output

        # Neuron 7
        neuron7Net = netCalc(neuron7Weights, activeInputData)
        neuron7Output = outputCalc(neuron7Net)

        # Neuron 8
        neuron8Net = netCalc(neuron8Weights, activeInputData)
        neuron8Output = outputCalc(neuron8Net)

        # Calculate errors for neurons 7 & 8
        neuron7Error = expOutputN7 - neuron7Output
        neuron8Error = expOutputN8 - neuron8Output

        # Calculate errors for neurons 4, 5 & 6
        neuron4Error = neuron4Output * (1 - neuron4Output) * ((neuron7Weights[1] * neuron7Error) + (neuron8Weights[1] * neuron8Error))
        neuron5Error = neuron5Output * (1 - neuron5Output) * ((neuron7Weights[2] * neuron7Error) + (neuron8Weights[2] * neuron8Error))
        neuron6Error = neuron6Output * (1 - neuron6Output) * ((neuron7Weights[3] * neuron7Error) + (neuron8Weights[3] * neuron8Error))

        # Backpropagate errors
        weightReset(neuron4Error, neuron4Weights, activeInputData)
        weightReset(neuron5Error, neuron5Weights, activeInputData)
        weightReset(neuron6Error, neuron6Weights, activeInputData)
        hiddenWeightReset(neuron7Error, neuron7Weights, activeInputData)
        hiddenWeightReset(neuron8Error, neuron8Weights, activeInputData)

    trainingLoss = ((1/3) * ((neuron4Error**2) + (neuron5Error**2) + (neuron6Error**2) + (neuron7Error**2) + (neuron8Error**2)))
    iterationWeights = [neuron4Weights, neuron5Weights, neuron6Weights, neuron7Weights, neuron8Weights]

    validateSamples = [4, 5, 6]
    for x in trainSamples:
        # Determine expected outputs
        expOutputN7 = inputData[x][1][0]
        expOutputN8 = inputData[x][1][1]

        # Determine inputs
        activeInputData = inputData[x][0]
        
        # Neuron 4
        neuron4Net = netCalc(neuron4Weights, activeInputData)
        neuron4Output = outputCalc(neuron4Net)

        # Neuron 5
        neuron5Net = netCalc(neuron5Weights, activeInputData)
        neuron5Output = outputCalc(neuron5Net)

        # Neuron 6
        neuron6Net = netCalc(neuron6Weights, activeInputData)
        neuron6Output = outputCalc(neuron6Net)

        # Update inputs from the hidden layer
        hiddenInputData[1] = neuron4Output
        hiddenInputData[2] = neuron5Output
        hiddenInputData[3] = neuron6Output

        # Neuron 7
        neuron7Net = netCalc(neuron7Weights, activeInputData)
        neuron7Output = outputCalc(neuron7Net)

        # Neuron 8
        neuron8Net = netCalc(neuron8Weights, activeInputData)
        neuron8Output = outputCalc(neuron8Net)

        # Calculate errors for neurons 7 & 8
        neuron7Error = expOutputN7 - neuron7Output
        neuron8Error = expOutputN8 - neuron8Output

        # Calculate errors for neurons 4, 5 & 6
        neuron4Error = neuron4Output * (1 - neuron4Output) * ((neuron7Weights[1] * neuron7Error) + (neuron8Weights[1] * neuron8Error))
        neuron5Error = neuron5Output * (1 - neuron5Output) * ((neuron7Weights[2] * neuron7Error) + (neuron8Weights[2] * neuron8Error))
        neuron6Error = neuron6Output * (1 - neuron6Output) * ((neuron7Weights[3] * neuron7Error) + (neuron8Weights[3] * neuron8Error))

    validateLoss = ((1/3) * ((neuron4Error**2) + (neuron5Error**2) + (neuron6Error**2) + (neuron7Error**2) + (neuron8Error**2)))
    print("Epoch " + str(epochs + 1))
    printWeights(iterationWeights)
    epochs += 1
    graphPlot.append((epochs, trainingLoss, validateLoss))

# Plot a graph for the learning curve
x_data = []
y_data = []
z_data = []
x_data.extend([graphPlot[i][0] for i in range(0, len(graphPlot))])
y_data.extend([graphPlot[i][1] for i in range(0, len(graphPlot))])
z_data.extend([graphPlot[i][2] for i in range(0, len(graphPlot))])
fig, ax = plt.subplots()
fig.suptitle("Learning Curve")
ax.set(xlabel="Epoch", ylabel= "Squared Error")
ax.plot(x_data, y_data, color="g")
ax.plot(x_data, z_data, color='r')
plt.show()

# Test the neural network
activeInputData = testInputData

# Neuron 4
neuron4Net = netCalc(neuron4Weights, activeInputData)
neuron4Output = outputCalc(neuron4Net)

# Neuron 5
neuron5Net = netCalc(neuron5Weights, activeInputData)
neuron5Output = outputCalc(neuron5Net)

# Neuron 6
neuron6Net = netCalc(neuron6Weights, activeInputData)
neuron6Output = outputCalc(neuron6Net)

# Update inputs from the hidden layer
hiddenInputData[1] = neuron4Output
hiddenInputData[2] = neuron5Output
hiddenInputData[3] = neuron6Output

# Neuron 7
neuron7Net = netCalc(neuron7Weights, activeInputData)
neuron7Output = outputCalc(neuron7Net)

# Neuron 8
neuron8Net = netCalc(neuron8Weights, activeInputData)
neuron8Output = outputCalc(neuron8Net)

print("Output: " + str(neuron7Output))
print("Softmax: " + str(softmax(neuron7Output, neuron8Output)) + ", " + str(softmax(neuron8Output, neuron7Output)))