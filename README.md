# Smarty-Pants

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

### Smarty-Pants

The data was split into 2 groups: 60,000 images for training and 10,000 images for testing.

The neural network learns only from the infomation gathered from the training data and, later, after training is complete, tests itself on the testing data leftover that it has never seen before.

## Neural Network

In this section i'll briefly try to explain how my network works.

### Structure

I built a network with:

- 784 input neurons. (one for each pixel of each image).
- 30  hidden neurons.
- 10  output neurons. (one for each digit).

784-30-10

### Neurons

I chose the Sigmoid function as the activation function for my neurons.

### Cost function

I chose the Quadratic Loss (aka: Mean Squared Error) function as my cost function.

### Learning algorithm

I used Stochastic Gradient Descent as my optimization (learning) algorithm.

## What can I do with this code?

There are a couple things you can try using my code

### Use smarty_pants.bin (trained model) to make predictions on test images

1. compile the code: 

```
$ gcc neural_network.c -o neural_network.exe -lm
```

2. Open the folder: data/mnist-test-images/tif

3. Choose images you'd like to see the network classify.

4. execute the code with the image numbers as parameters: 

```
Ex: $ ./neural_network.exe smarty_pants.bin 1 2 3 4 5 6 7 8 9 10
```

### See smarty_pants.bin (trained model) precision over all the 10,000 images

1. compile the code: 

```
$ gcc neural_network.c -o neural_network.exe -lm
```

2. execute the code: 

```
$ ./neural_network.exe smarty_pants.bin
```

### Train your own neural network and test it

1. compile the code: 

```
$ gcc neural_network.c -o neural_network.exe -lm
```

2. train: execute the code (this will take a long time: ~10min): 

```
$ ./neural_network.exe
```

3. By default the new network will be saved as custom.bin.

4. test: execute the code:

```
$ ./neural_network.exe custom.bin
```
