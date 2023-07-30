import matplotlib as plt
import math

# A function for the user to dynamically set the neural network structure
def getStructure():
    userInput = input("Enter the structure of the neural network (comma-separated integers): ")
    structure = [int(neurons) for neurons in userInput.split(",")]
    return structure