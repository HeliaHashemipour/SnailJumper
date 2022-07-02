import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        """
            Neural Network initialization.
            Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
            :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
            3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
            """
        # TODO (Implement FCNNs architecture here)
        input_layer = self.layer_sizes[0]
        hidden_layer = self.layer_sizes[1]
        output_layer = self.layer_sizes[2] # input_layer = 8, hidden_layer = 32, output_layer
        # = 2
        self.weight_1 = np.random.randn(input_layer * hidden_layer).reshape(hidden_layer,
                                                                            input_layer)  # weight_1 = 32x8
        self.bias_1 = np.zeros((hidden_layer, 1))  # bias_1 = 32x1
        self.weight_2 = np.random.randn(output_layer * hidden_layer).reshape(output_layer,
                                                                             hidden_layer)  # weight_2 = 2x32
        self.bias_2 = np.zeros((output_layer, 1))  # bias_2 = 2x1

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param function:
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)

        return np.exp(x) / np.exp(x).sum()
        # return 1 / (1 + np.exp(-x))

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # TODO (Implement forward function here)
        x = np.array(x)
        x = x.reshape(self.layer_sizes[0], 1)  # x = 8x1
        z1 = self.weight_1 @ x + self.bias_1  # z1 = 32x1
        a1 = self.activation(z1)  # a1 = 32x1
        z2 = self.weight_2 @ a1 + self.bias_2  # z2 = 2x1
        a2 = self.activation(z2)  # a2 = 2x1
        return a2  # return 2x1 (z2)
