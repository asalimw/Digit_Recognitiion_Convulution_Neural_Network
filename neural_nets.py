import numpy as np
import math

"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""


def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    # TODO
    # print(x)
    return np.maximum(0, x)

def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    # TODO
    if x <= 0:
        x = 0
    else:
        x = 1
    return x

def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1

class NeuralNetwork():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):

        # DO NOT CHANGE PARAMETERS (Initialized to floats instead of ints)
        self.input_to_hidden_weights = np.matrix('1. 1.; 1. 1.; 1. 1.')
        self.hidden_to_output_weights = np.matrix('1. 1. 1.')
        self.biases = np.matrix('0.; 0.; 0.')
        self.learning_rate = .001
        self.epochs_to_train = 10
        self.training_points = [((2,1), 10), ((3,3), 21), ((4,5), 32), ((6, 6), 42)]
        self.testing_points = [(1,1), (2,2), (3,3), (5,5), (10,10)]

    def train(self, x1, x2, y):

        # By jre04d - write down the array (matrix or vector) versions of all the variables and parameters:
        weight1 = np.array(self.input_to_hidden_weights) #The weight between input to hidden layer(first)
        bias = np.array(self.biases) # The (input-to-hidden) biases as bias
        relu = np.vectorize(rectified_linear_unit) #The hidden-layer-activation using Relu
        weight2 = np.array(self.hidden_to_output_weights)  # The weight between hidden to output layer (last)
        relu_prime = np.vectorize(rectified_linear_unit_derivative) #Relu derivative for backprop
        output_relu = output_layer_activation # The Relu for output
        output_relu_prime = output_layer_activation_derivative #The Relu derivative for output during backdrop


        # print(x1, x2, y)

        ### Forward propagation ###
        # https://ml-cheatsheet.readthedocs.io/en/latest/forwardpropagation.html
        input_values = np.matrix([[x1],[x2]]) # 2 by 1 given by the project
        x = np.array(input_values) # The input value as x in an array

        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input = weight1 @ x + bias # TODO (3 by 1 matrix)
        hidden_layer_activation = relu(hidden_layer_weighted_input)  # TODO (3 by 1 matrix)

        # Calculate the Output Layer
        output = weight2 @ hidden_layer_activation  # TODO Output before Relu
        activated_output = output_relu(output) # TODO ReLU Output Layer

        ### Backpropagation ###
        # https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html


        # Compute gradients

        output_layer_error = activated_output - y # TODO
        hidden_layer_error = np.multiply((output_layer_error * output_relu_prime(output)), np.multiply(relu_prime(hidden_layer_weighted_input), weight2.T)) # TODO (3 by 1 matrix)
        # print(hidden_layer_error.shape)


        # https://datascience.stackexchange.com/questions/20139/gradients-for-bias-terms-in-backpropagation
        bias_gradients = hidden_layer_error  # TODO
        hidden_to_output_weight_gradients =  np.multiply(output_layer_error, hidden_layer_activation) # TODO
        input_to_hidden_weight_gradients = np.multiply(bias_gradients, x.T) # TODO

        # print(bias_gradients.shape)
        # print(hidden_to_output_weight_gradients.shape)
        # print(input_to_hidden_weight_gradients.shape)


        # Use gradients to adjust weights and biases using gradient descent
        self.biases = bias - self.learning_rate * bias_gradients # TODO
        self.input_to_hidden_weights = weight1 - self.learning_rate * input_to_hidden_weight_gradients # TODO
        self.hidden_to_output_weights = weight2 - self.learning_rate * hidden_to_output_weight_gradients.T # TODO
        # print(self.biases.shape)
        # print(self.input_to_hidden_weights.shape)
        # print(self.hidden_to_output_weights.shape)


    def predict(self, x1, x2):

        bias = np.array(self.biases)  # The (input-to-hidden) biases as bias
        weight1 = np.array(self.input_to_hidden_weights)  # The weight between input to hidden layer(first)
        weight2 = np.array(self.hidden_to_output_weights)  # The weight between hidden to output layer (last)
        relu = np.vectorize(rectified_linear_unit)  # The hidden-layer-activation using Relu
        output_relu = output_layer_activation  # The Relu for output

        input_values = np.matrix([[x1],[x2]])
        x = np.array(input_values)  # The input value as x in an array

        # Compute output for a single input(should be same as the forward propagation in training)
        hidden_layer_weighted_input = weight1 @ x + bias # TODO
        hidden_layer_activation = relu(hidden_layer_weighted_input) # TODO
        output = weight2 @ hidden_layer_activation # TODO
        activated_output = output_relu(output) # TODO

        return activated_output.item()

    # Run this to train your neural network once you complete the train method
    def train_neural_network(self):

        for epoch in range(self.epochs_to_train):
            for x,y in self.training_points:
                self.train(x[0], x[1], y)

    # Run this to test your neural network implementation for correctness after it is trained
    def test_neural_network(self):

        for point in self.testing_points:
            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            if abs(self.predict(point[0], point[1]) - 7*point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0], point[1], " failed to be predicted correctly.")
                return

x = NeuralNetwork()

x.train_neural_network()

# UNCOMMENT THE LINE BELOW TO TEST YOUR NEURAL NETWORK
x.test_neural_network()
