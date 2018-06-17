# Libraries
# Standard

import random
import json
import sys

# 3rd Party Libraries

import numpy as np


def sigmoid(z):

    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):

    return sigmoid(z) * (1 - sigmoid(z))


class Neural_Network(object):

    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes

        self.randomize()

    def randomize(self):

        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]

        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedForward(self, a):

        for b, w in zip(self.biases, self.weights):

            a = sigmoid(np.dot(w, a) + b)

        return a

    def save(self, filename):

        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases]
        }

        f = open(filename, "w")

        json.dump(data, f)

        f.close()


def load(filename):

    f = open(filename, "r")

    data = json.load(f)

    f.close()

    smarty_pants = Neural_Network(data["sizes"])
    smarty_pants.weights = [np.array(w) for w in data["weights"]]
    smarty_pants.biases = [np.array(b) for b in data["biases"]]

    return smarty_pants


def main():

    smarty_pants = Neural_Network([784, 15, 10])

    smarty_pants.save("smarty_pants.json")


if __name__ == '__main__':

    main()
