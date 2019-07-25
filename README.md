# MNIST-NeuralNetwork

Digit recognition Neural Network made in C.

This project was my "Hello world!" into machine learning. I built everything from scratch based only on the math in order to gain a deeper understanding and insight on machine learning.

## What is in this repository

This project contains C code for: Building, Training and Testing a Standard Neural Network.

It also contains the 70,000 images used to Train/Test, along side various metadata information.

## Data

Here I'll talk a bit about and try to explain the data used in this project.

### Mnist

For this project, I decided to use the mnist dataset of handwritten digits. It contains 70,000 images of handwritten digits and their associated corresponding labels.

Each image is 28x28 pixels in size, and contains a black handwritten digit on white background.

Each label is a number from 0 to 9 that identifies the digit shown in the image.

### MNIST-NeuralNetwork

The data was split into 2 groups: 60,000 images for training and 10,000 images for testing.

The neural network learns only from the infomation gathered from the training data and, later, after training is complete, tests itself on the testing data leftover that it has never seen before.

## Neural Network

In this section i'll briefly try to explain how my network works.

### Structure

I built a network with:

- 784 input neurons (one for each pixel of each image).
- 30  hidden neurons.
- 10  output neurons (one for each digit).

784-30-10

#### Neurons

I chose the Sigmoid function as the activation function for my neurons.

#### Cost function

I chose the Quadratic Loss (aka: Mean Squared Error) function as my cost function.

#### Learning algorithm

I used Stochastic Gradient Descent as my optimization (learning) algorithm.

### How neural networks work

A Neural Network is just a sequence of fully connected layers of neurons where, in each layer, the neuron activations within depend on the activations of the neurons in the previous layer (with the exception of the input neurons).

When a network makes a prediction, it simply feeds forward the activations of neurons from one layer to the next according to their weights and biases, starting in the input, going layer by layer, until it reaches the final output layer.

### MNIST-NeuralNetwor

By default the network builts itself, randomizes its weights and biases and then proceeds to train.

#### Building

Consists of creating a Neural Network where any neuron of any layer is always fully connected to every other neuron in the previous layer, with the exception of the input neurons.

#### Randomizing

Consists of setting the initial weights and biases to sensible, random values. These values will later be adjusted to improve the network performance.

#### Training

Consists of using an optimization algorithm such as Stochastic Gradient Decent (SGD) in order to minimize the cost function by adjusting the weights and biases of the network.

##### Cost function

The cost function is a function that rates how good (or bad) the network is doing. for each given training example, it compares the output produced by the network to the output that should have been produced.

This Cost function gives high outputs when the network is classifying digits poorly and gives lower outputs when the network is classifying digits better.

So, if we could figure out how to minimize this cost function, we could figure out how to make the network improve its classification.

##### Stochastic Gradient Decent

Thats where SGD comes in. It gives us a method for efficiently computing gradients (vector containing partial derivatives of the cost function in respect to the weights and biases) and use the negative gradient of the cost function to take a small steps in the oposite direction, lowering the future outputs of the cost function until a minimum is reached.

SGD depends on an algoritm called backpropagation to feed backwards an error through the network which in turn allows us the calculate the partial derivatives of the gradient.

## What can I do with this code?

There are a couple things you can try using my code

#### Use net.bin (trained model) to make predictions on test images

1. compile the code:

```
$ gcc neural_network.c -o neural_network.exe -lm
```

2. Open the folder: data/mnist-test-images/tif

3. Choose images you'd like to see the network classify.

4. execute the code with the image numbers as parameters:

```
Ex: $ ./neural_network.exe net.bin 1 2 3 4 5 6 7 8 9 10
```

#### See net.bin (trained model) precision over all the 10,000 images

1. compile the code:

```
$ gcc neural_network.c -o neural_network.exe -lm
```

2. execute the code:

```
$ ./neural_network.exe net.bin
```

#### Train your own neural network and test it

1. compile the code:

```
$ gcc neural_network.c -o neural_network.exe -lm
```

2. train: execute the code (this will take a long time: ~10min):

```
$ ./neural_network.exe
```

3. By default the new network will be saved as custom.bin.
