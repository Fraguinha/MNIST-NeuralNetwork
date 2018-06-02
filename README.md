# Smarty-Pants

Digit recognition Neural Network made in C.

## What im trying to do

The goal is to build a simple neural network from scratch which will, hopefully, be able to take handwritten digit images as input and classify which digit was shown as the output.

## Neural Network Sketch

![Network-Sketch](https://i.imgur.com/wCMvXsC.jpg)

At the time of this writing, This network consists of 784 sensors (one for each of the 28x28 pixels) all conected to every neuron on the first layer of 16 neurons (called sensor neurons). Those first neurons are in turn all connected to the neurons in the first hidden layer of neurons, containing 16 neurons. Finally the last hidden layer of neurons is connected to each of the 10 output neurons (one for each of the 10 digits).

### Sensor & Neuron Sketch

![Sensor/Neuron-Sketch](https://i.imgur.com/ZkVIwDJ.jpg)

The Sensor only knows the grayscale value of the pixel it corresponds to.
The Neuron knows each of the activations from the previous layer (inputs), what importance it gives each one (weights), its own activation bias (bias) and its own computed activation value (weighted_sum).
