# Smarty-Pants

Digit recognition Neural Network made in C.

## What it is

The goal in this project was to build a simple feedforward neural network from scratch which would, hopefully, be able to take in pixel values of handwritten digit images as input and correctly classify them as the output.

## Neural Network

In this section i'll briefly try to explain a few sections of my network.

### Data

For this project, I decided to use the mnist dataset of handwritten digits. It contains 70,000 images of handwritten digits.

The data was split into 2 groups: 60,000 images for training and 10,000 images for testing.

Each image is 28x28 pixels in size, and contains a handwritten digit.

### Structure

I chose a network with:

- 784 neurons as input (one for each pixel of each image).
- 30 neurons on the first hidden layer.
- no extra hidden layers.
- and 10 neurons as output (one for each digit).

### Neurons

I chose the sigmoid function as the activation function for my neurons.

### Cost function

I chose the quadratic loss function as my cost function.

### Learning algorithm

I used Stochastic Gradient Descent as my learning algorithm.
