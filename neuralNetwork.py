import matplotlib as plt
import math

class Neuron:
    def __init__(self, nodeNumber, weights, input, output, bias):
        self.nodeNumber = nodeNumber
        self.weights = weights
        self.input = input
        self.output = output
        self.error = 0
        self.bias = bias

    def output(self):
        net = 0
        net += self.weights[0] * self.bias
        for x in range(1, len(self.weights) - 1):
            net += self.weights[x] * self.input[x]
        return 1 / (1 + math.exp(-net))
    


# Main
def main():
    # Ask the user how many epochs to train for
    epochs = 0
    epochMax = int(input("How many epochs should I train for?: "))
    

if __name__ == "__main__":
    main()
