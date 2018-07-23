/* Libraries */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/* Definitions */

#define inputs (28 * 28) // number of inputs
#define outputs 10       // number of outputs

#define train_directory "data/mnist-train-images/txt/"
#define test_directory "data/mnist-test-images/txt/"

#define train_label "data/mnist-train-labels.txt"
#define test_label "data/mnist-test-labels.txt"

/* Tweaking */
#define layers 0                               // number of layers
#define layer_size 15                          // number of neurons for each layer
#define layer_size_h (layers ? layer_size : 0) // number of neurons for each hidden layer

#define min_weight -2 // minimum value for the weights
#define max_weight +2 // maximum value for the weights

#define min_bias -1 // maximum value for the bias
#define max_bias +1 // maximum value for the bias

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

    float weighted_sum;
    float activation;

} sensor_neuron;

/* Neuron */
typedef struct
{
    float bias;
    float weights[layer_size];

    float weighted_sum;
    float activation;

} neuron_neuron;

/* Neural Network */
typedef struct
{
    /* Structure */
    sensor array_sensors[inputs];                     // sensors
    sensor_neuron array_inputs[layer_size];           // sensor neurons
    neuron_neuron array_hidden[layers][layer_size_h]; // layers of neuron neurons
    neuron_neuron array_outputs[outputs];             // output neurons

    /* Information */
    int prediction;
    int label;

} Neural_Network;

/*
 * The rectifier function
 *
 * @param float x
 * @return float sigmoid
 */
float relu(float x)
{
    return x > 0 ? x : 0;
}

/*
 *  This function generates a random float
 *
 *  @param float minimum, float maximum
 *  @return float random
 */
float randomizedFloat(float minimum, float maximum)
{
    return (((float)rand()) / (float)(RAND_MAX)) * (maximum - minimum) + minimum;
}

/*
 *  This functions randomizes all the weights and biases of the neurons in the network
 *
 *  @param Neural_Network *net
 */
void randomize(Neural_Network *net)
{
    // Make randomizer seed
    srand((unsigned int)time(NULL));

    // Randomize sensor neurons weights and biases
    for (int j = 0; j < layer_size; j++)
    {
        net->array_inputs[j].bias = randomizedFloat(min_bias, max_bias);

        for (int k = 0; k < inputs; k++)
        {
            net->array_inputs[j].weights[k] = randomizedFloat(min_weight, max_weight);
        }
    }

    // Randomize hidden layer neurons weights and biases
    for (int l = 0; l < layers; l++)
    {
        for (int j = 0; j < layer_size; j++)
        {
            net->array_hidden[l][j].bias = randomizedFloat(min_bias, max_bias);

            for (int k = 0; k < layer_size; k++)
            {
                net->array_hidden[l][j].weights[k] = randomizedFloat(min_weight, max_weight);
            }
        }
    }

    // Randomize output neurons weights and biases
    for (int j = 0; j < outputs; j++)
    {
        net->array_outputs[j].bias = randomizedFloat(min_bias, max_bias);

        for (int k = 0; k < layer_size; k++)
        {
            net->array_outputs[j].weights[k] = randomizedFloat(min_weight, max_weight);
        }
    }
}

/*
 *  This functions sets the activations of all the sensors in the network, and the label
 *
 *  @param Neural_Network *net, const char *tifImage, const char *txtLabel, int number
 */
void setInput(Neural_Network *net, const char *tifImage, const char *txtLabel, int number)
{
    FILE *image = fopen(tifImage, "r");

    for (int i = 0; i < inputs; i++)
    {
        fscanf(image, "%f", &net->array_sensors[i].activation);
    }

    fclose(image);

    FILE *solution = fopen(txtLabel, "r");

    int trash;

    for (int j = 1; j < number; j++)
    {
        fscanf(solution, "%d", &trash);
    }

    fscanf(solution, "%d", &net->label);

    fclose(solution);
}

/*
 *  This functions calculates the weighted sum of all the neurons in the network
 *  (weighted sum = sum(weights * activations) - bias)
 *
 *  @param Neural_Network *net
 */
void feedForward(Neural_Network *net)
{
    float sum;

    // Calculate input neurons weighted sum
    for (int j = 0; j < layer_size; j++)
    {
        sum = 0;

        for (int k = 0; k < inputs; k++)
        {
            sum += net->array_sensors[k].activation * net->array_inputs[j].weights[k];
        }

        sum += net->array_inputs[j].bias;

        net->array_inputs[j].weighted_sum = sum;

        net->array_inputs[j].activation = relu(sum);
    }

    if (layers)
    {

        // Calculate hidden layer neurons weighted sum
        for (int l = 0; l < layers; l++)
        {
            for (int j = 0; j < layer_size; j++)
            {
                sum = 0;

                for (int k = 0; k < layer_size; k++)
                {
                    sum += net->array_inputs[k].weighted_sum * net->array_hidden[l][j].weights[k];
                }

                sum += net->array_hidden[l][j].bias;

                net->array_hidden[l][j].weighted_sum = sum;

                net->array_hidden[l][j].activation = relu(sum);
            }
        }

        // Calculate output neurons weighted sum
        for (int j = 0; j < outputs; j++)
        {
            sum = 0;

            for (int k = 0; k < layer_size; k++)
            {
                sum += net->array_hidden[layers - 1][k].weighted_sum * net->array_outputs[j].weights[k];
            }

            sum += net->array_outputs[j].bias;

            net->array_outputs[j].weighted_sum = sum;

            net->array_outputs[j].activation = relu(sum);
        }
    }
    else
    {
        // Calculate output neurons weighted sum
        for (int j = 0; j < outputs; j++)
        {
            sum = 0;

            for (int k = 0; k < layer_size; k++)
            {
                sum += net->array_inputs[k].weighted_sum * net->array_outputs[j].weights[k];
            }

            sum += net->array_outputs[j].bias;

            net->array_outputs[j].weighted_sum = sum;

            net->array_outputs[j].activation = relu(sum);
        }
    }
}

/*
 *  This function calculates all partial derivatives
 *
 *  @param Neural_Network *net
 */
void backPropagate(Neural_Network *net)
{
}

/*
 *  This functions creates filename string of an image
 *
 *  @param Neural_Network *net, const char *filename, int number, int flag
 */
void getFilename(char *filename, int number, int flag)
{

    filename[0] = '\0';

    char num[6];

    sprintf(num, "%05d", number);

    char *directory = train_directory;

    if (flag == 1)
    {
        char *directory = test_directory;
    }

    char *extension = ".txt";

    strcat(filename, directory);
    strcat(filename, num);
    strcat(filename, extension);
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

int main(int argc, char const *argv[])
{
    // Create the Neural Network
    Neural_Network smarty_pants;
}
