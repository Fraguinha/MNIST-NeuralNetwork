/* Libraries */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/* Definitions */

#define inputs 784 // the inputs available to make a prediction
#define outputs 10 // the outputs available as a prediction

#define total_weights (inputs * layer_size + layers * (layer_size * layer_size) + layer_size * outputs)
#define total_biases (layer_size + layers * layer_size + outputs)

/* Tweaking */
#define layers 1      // number of layers
#define layer_size 16 // number of neurons for each layer

#define min_weight -10 // minimum value for the weights
#define max_weight 10  // maximum value for the weights

#define min_bias -10 // maximum value for the bias
#define max_bias 10  // maximum value for the bias

/* Structures */

/* Sensor */
typedef struct
{
    float activation;

} sensor;

/* Neuron */
typedef struct
{
    float bias;
    float weights[inputs];

    // activation
    float weighted_sum;

} sensor_neuron;

/* Neuron */
typedef struct
{
    float bias;
    float weights[layer_size];

    // activation
    float weighted_sum;

} neuron_neuron;

/* Neural Network */
typedef struct
{
    /* Structure */
    sensor array_sensors[inputs];               // sensors
    sensor_neuron array_sn[layer_size];         // sensor neurons
    neuron_neuron array_nn[layers][layer_size]; // layers of neuron neurons
    neuron_neuron array_outputs[outputs];       // output neurons

    /* Information */
    float score;
    int label;

} Neural_Network;

/* Function Declarations */

float sigmoid(float x);

float randomizedFloat(float minimum, float maximum);

void randomize(Neural_Network *net);

void save(Neural_Network *net, const char *filename);
void load(Neural_Network *net, const char *filename);

void setActivation(Neural_Network *net, const char *filename, const char *label, int i);
void showInputs(Neural_Network *net);

void calculateSum(Neural_Network *net);

void showPrediction(Neural_Network *net);
void showOutputs(Neural_Network *net);
void showLabel(Neural_Network *net);

void calculateScore(Neural_Network *net);
void showScore(Neural_Network *net);

void backpropagate(Neural_Network *net);

int main(int argc, char const *argv[])
{
    // Create the Neural Network
    Neural_Network smarty_pants;

    // Randomize Neurons
    randomize(&smarty_pants);

    // Save Network
    save(&smarty_pants, "smarty_pants.bin");

    // Load Network
    load(&smarty_pants, "smarty_pants.bin");

    // Set Activation of input neurons
    setActivation(&smarty_pants, "data/mnist-train-images/txt/00001.tif.txt", "data/mnist-train-labels.txt", 1);

    // Show inputs
    showInputs(&smarty_pants);

    // Calculate activations
    calculateSum(&smarty_pants);

    // Show outputs
    showOutputs(&smarty_pants);

    // Show correct label
    showLabel(&smarty_pants);

    // Show prediction
    showPrediction(&smarty_pants);

    // Calculate score
    calculateScore(&smarty_pants);

    // Show score
    showScore(&smarty_pants);

    // Optimize
    backpropagate(&smarty_pants);

    return 0;
}

/*
 * The sigmoid function
 * 
 * @param float x
 * @return float sigmoid
 */
float sigmoid(float x)
{
    float ex = expf(-x);

    float sigmoid = 1 / (1 + ex);

    return sigmoid;
}

/*
 *  This function generates a random float
 *
 *  @param float minimum, float maximum
 *  @return float random
 */
float randomizedFloat(float minimum, float maximum)
{
    float random;
    random = (((float)rand()) / (float)(RAND_MAX)) * (maximum - minimum) + minimum;
    return random;
}

/*
 *  This functions randomizes all the weights and biases of the neurons in the network
 *
 *  @param Neural_Network *net
 */
void randomize(Neural_Network *net)
{
    // Make randomizer
    srand((unsigned int)time(NULL));

    // Randomize sensor neurons weights and biases
    for (int i = 0; i < layer_size; i++)
    {
        net->array_sn[i].bias = randomizedFloat(min_bias, max_bias);

        for (int j = 0; j < inputs; j++)
        {
            net->array_sn[i].weights[j] = randomizedFloat(min_weight, max_weight);
        }
    }

    // Randomize hidden layer neurons weights and biases
    for (int i = 0; i < layers; i++)
    {
        for (int j = 0; j < layer_size; j++)
        {
            net->array_nn[i][j].bias = randomizedFloat(min_bias, max_bias);

            for (int k = 0; k < layer_size; k++)
            {
                net->array_nn[i][j].weights[k] = randomizedFloat(min_weight, max_weight);
            }
        }
    }

    // Randomize output neurons weights and biases
    for (int i = 0; i < outputs; i++)
    {
        net->array_outputs[i].bias = randomizedFloat(min_bias, max_bias);

        for (int j = 0; j < layer_size; j++)
        {
            net->array_outputs[i].weights[j] = randomizedFloat(min_weight, max_weight);
        }
    }
}

/*
 *  This functions saves the network to a binary file
 *
 *  @param Neural_Network *net, const char *filename
 */
void save(Neural_Network *net, const char *filename)
{
    FILE *fp = fopen(filename, "wb");

    if (fp != NULL)
    {
        fwrite(net, sizeof(Neural_Network), 1, fp);
    }

    fclose(fp);
}

/*
 *  This functions loads the network from a binary file
 *
 *  @param Neural_Network *net, const char *filename
 */
void load(Neural_Network *net, const char *filename)
{
    FILE *fp = fopen(filename, "rb");

    if (fp != NULL)
    {
        fread(net, sizeof(Neural_Network), 1, fp);
    }

    fclose(fp);
}

/*
 *  This functions sets the activations of all the sensors in the network, and the label
 *
 *  @param Neural_Network *net, const char *filename, const char *label, int i
 */
void setActivation(Neural_Network *net, const char *filename, const char *label, int i)
{

    FILE *image = fopen(filename, "r");

    for (int i = 0; i < inputs; i++)
    {
        fscanf(image, "%f", &net->array_sensors[i].activation);
    }

    fclose(image);

    FILE *solution = fopen(label, "r");

    int trash;

    for (int j = 0; j < i - 1; j++)
    {
        fscanf(solution, "%d", &trash);
    }

    fscanf(solution, "%d", &net->label);

    fclose(solution);
}

/*
 *  This functions shows the input activations
 *
 *  @param Neural_Network *net
 */
void showInputs(Neural_Network *net)
{
    for (int i = 0; i < inputs; i++)
    {
        printf("Activation of input [%3d]: %.2f\n", i, net->array_sensors[i].activation);
    }
}

/*
 *  This functions calculates the weighted sum of all the neurons in the network
 *  (weighted sum = sum(weights * activations) - bias)
 *
 *  @param Neural_Network *net
 */
void calculateSum(Neural_Network *net)
{
    float sum;

    // Calculate sensor neurons weighted sum
    for (int i = 0; i < layer_size; i++)
    {
        sum = 0;

        for (int j = 0; j < inputs; j++)
        {
            sum += net->array_sensors[j].activation * net->array_sn[i].weights[j];
        }

        sum += net->array_sn[i].bias;

        sum = sigmoid(sum);

        net->array_sn[i].weighted_sum = sum;
    }

    // Calculate neuron neurons weighted sum
    for (int i = 0; i < layers; i++)
    {
        for (int j = 0; j < layer_size; j++)
        {
            sum = 0;

            for (int k = 0; k < layer_size; k++)
            {
                sum += net->array_sn[k].weighted_sum * net->array_nn[i][j].weights[k];
            }

            sum += net->array_nn[i][j].bias;

            sum = sigmoid(sum);

            net->array_nn[i][j].weighted_sum = sum;
        }
    }

    // Calculate output neurons weighted sum
    for (int i = 0; i < outputs; i++)
    {
        sum = 0;

        for (int j = 0; j < layer_size; j++)
        {
            sum += net->array_nn[layers - 1][j].weighted_sum * net->array_outputs[i].weights[j];
        }

        sum += net->array_outputs[i].bias;

        sum = sigmoid(sum);

        net->array_outputs[i].weighted_sum = sum;
    }
}

/*
 *  This function runs through the output neurons and displays the one with the highest activation
 *
 *  @param Neural_Network *net
 */
void showPrediction(Neural_Network *net)
{

    float max = net->array_outputs[0].weighted_sum;
    int prediction = 0;

    for (int i = 0; i < outputs; i++)
    {
        if (net->array_outputs[i].weighted_sum > max)
        {
            max = net->array_outputs[i].weighted_sum;
            prediction = i;
        }
    }

    printf("Neural network prediction: %d\n", prediction);
}

/*
 *  This functions shows the outputs activations
 *
 *  @param Neural_Network *net
 */
void showOutputs(Neural_Network *net)
{
    for (int i = 0; i < outputs; i++)
    {
        printf("Activation of output [%d]: %.2f\n", i, net->array_outputs[i].weighted_sum);
    }
}

/*
 *  This functions shows the correct label
 *
 *  @param Neural_Network *net
 */
void showLabel(Neural_Network *net)
{
    printf("Neural network correct label: %d\n", net->label);
}

/*
 *  This functions calculates the score (low values = good ~ high values = bad) of the network
 *  (score = sum(pow(output_neuron.weighted_sum - expected_weighted_sum, 2)))
 *
 *  @param Neural_Network *net
 */
void calculateScore(Neural_Network *net)
{
    float sum = 0;

    for (int i = 0; i < outputs; i++)
    {
        if (net->label == i)
        {
            sum += pow((net->array_outputs[i].weighted_sum - 1.0), 2);
        }
        else
        {
            sum += pow((net->array_outputs[i].weighted_sum - 0.0), 2);
        }
    }

    net->score = sum;
}

/*
 * This function displays the neural network score
 * 
 * @param Neural_Network *net
 */
void showScore(Neural_Network *net)
{
    printf("Neural network score: %f\n", net->score);
}

/*
 *  This functions is the learning algorithm
 *
 *  @param Neural_Network *net
 */
void backpropagate(Neural_Network *net)
{
    // hmmm... ok, i got this..
}