/*
    neural_network.c
    
    This program creates and trains a Multi-Layered Perceptron using backpropagation.
    
    compilation: gcc neural_network.c -o neural_network.exe -lm
    
    usage: ./neural_network.exe
    usage: ./neural_network.exe [neural_network.bin]
    usage: ./neural_network.exe [neural_network.bin] [...]
    
    author: Joao Fraga
    e-mail: joaoluisfreirefraga@gmail.com
*/

/************* 
 * Libraries *
 *************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/*************** 
 * Definitions *
 ***************/

#define inputs (28 * 28) // number of inputs     (each of the 784 pixels)
#define outputs 10       // number of outputs {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

#define training 60000 // number of training samples
#define testing 10000  // number of testing samples

#define train_directory "data/mnist-train-images/txt/"
#define test_directory "data/mnist-test-images/txt/"

#define train_label "data/mnist-train-labels.txt"
#define test_label "data/mnist-test-labels.txt"

/************ 
 * Tweaking *
 ************/

#define layer_size 15                          // number of neurons for each layer
#define layers 0                               // number of layers  within hidden layer
#define layer_size_h (layers ? layer_size : 0) // number of neurons within hidden layer for each hidden layer

#define min_weight -2 // minimum value for the weights
#define max_weight +2 // maximum value for the weights

#define min_bias -1 // maximum value for the bias
#define max_bias +1 // maximum value for the bias

#define batch_size 100                         // number of examples per batch
#define train_sessions (training / batch_size) // number of training sessions per epoch

#define learning_rate 0.05 // how much to change the weights and biases
#define max_epochs 1000    // maximum number of epochs

/************** 
 * Structures *
 **************/

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

    float error;

} sensor_neuron;

/* Neuron */
typedef struct
{
    float bias;
    float weights[layer_size];

    float weighted_sum;
    float activation;

    float error;

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

/************************
 * Function Definitions *
 ************************/

/*
 *  This functions displays relevant information about network to help with debugging
 *
 *  @param Neural_Network *net, const char *filename
 */
void debug(Neural_Network *net)
{
    printf("v------------------------------------------------------------------------------v\n");

    // Sensors

    printf("\nSensors:\n");

    printf("\nACTIVATION MATRIX:\n\n");

    for (int j = 0; j < inputs; j++)
    {
        printf("[%09.7f]", net->array_sensors[j].activation);

        printf("\n");
    }

    // Inputs

    printf("\nInputs:\n");

    printf("\nWEIGHT MATRIX:\n\n");

    for (int k = 0; k < inputs; k++)
    {

        for (int j = 0; j < layer_size; j++)
        {
            printf("[%09.6f]", net->array_inputs[j].weights[k]);
        }

        printf("\n");
    }

    printf("\nBIAS MATRIX:\n\n");

    for (int j = 0; j < layer_size; j++)
    {
        printf("[%09.6f]", net->array_inputs[j].bias);
    }

    printf("\n");

    printf("\nWEIGHTEDSUM MATRIX:\n\n");

    for (int j = 0; j < layer_size; j++)
    {
        printf("[%09.5f]", net->array_inputs[j].weighted_sum);
    }

    printf("\n");

    printf("\nACTIVATION MATRIX:\n\n");

    for (int j = 0; j < layer_size; j++)
    {
        printf("[%09.7f]", net->array_inputs[j].activation);
    }

    printf("\n");

    printf("\nERROR MATRIX:\n\n");

    for (int j = 0; j < layer_size; j++)
    {
        printf("[%09.6f]", net->array_inputs[j].error);
    }

    printf("\n");

    // Hidden

    if (layers)
    {
        printf("\nHidden:\n");

        for (int l = 0; l < layers; l++)
        {

            printf("\nLayer %d:\n", l);

            printf("\nWEIGHT MATRIX:\n\n");

            for (int k = 0; k < layer_size; k++)
            {

                for (int j = 0; j < layer_size; j++)
                {
                    printf("[%09.6f]", net->array_hidden[l][j].weights[k]);
                }

                printf("\n");
            }

            printf("\nBIAS MATRIX:\n\n");

            for (int j = 0; j < layer_size; j++)
            {
                printf("[%09.6f]", net->array_hidden[l][j].bias);
            }

            printf("\n");

            printf("\nWEIGHTEDSUM MATRIX:\n\n");

            for (int j = 0; j < layer_size; j++)
            {
                printf("[%09.5f]", net->array_hidden[l][j].weighted_sum);
            }

            printf("\n");

            printf("\nACTIVATION MATRIX:\n\n");

            for (int j = 0; j < layer_size; j++)
            {
                printf("[%09.7f]", net->array_hidden[l][j].activation);
            }

            printf("\n");

            printf("\nERROR MATRIX:\n\n");

            for (int j = 0; j < layer_size; j++)
            {
                printf("[%09.6f]", net->array_hidden[l][j].error);
            }

            printf("\n");
        }
    }

    // Output

    printf("\nOutputs:\n");

    printf("\nWEIGHT MATRIX:\n\n");

    for (int k = 0; k < layer_size; k++)
    {

        for (int j = 0; j < outputs; j++)
        {
            printf("[%09.6f]", net->array_outputs[j].weights[k]);
        }

        printf("\n");
    }

    printf("\nBIAS MATRIX:\n\n");

    for (int j = 0; j < outputs; j++)
    {
        printf("[%09.6f]", net->array_outputs[j].bias);
    }

    printf("\n");

    printf("\nWEIGHTEDSUM MATRIX:\n\n");

    for (int j = 0; j < outputs; j++)
    {
        printf("[%09.5f]", net->array_outputs[j].weighted_sum);
    }

    printf("\n");

    printf("\nACTIVATION MATRIX:\n\n");

    for (int j = 0; j < outputs; j++)
    {
        printf("[%09.7f]", net->array_outputs[j].activation);
    }

    printf("\n");

    printf("\nERROR MATRIX:\n\n");

    for (int j = 0; j < outputs; j++)
    {
        printf("[%09.6f]", net->array_outputs[j].error);
    }

    printf("\n");

    printf("\n<------------------------------------------------------------------------------>\n");
}

/*
 * The sigmoid function
 *
 * @param float x
 * @return float sigmoid
 */
float sigmoid(float x)
{
    return 1.0 / (1.0 + expf(-x));
}

/*
 * The sigmoid derivative function
 *
 * @param float x
 * @return float sigmoid
 */
float sigmoidDerivative(float x)
{
    return sigmoid(x) * (1 - sigmoid(x));
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

        net->array_inputs[j].weighted_sum = 0;
        net->array_inputs[j].activation = 0;
        net->array_inputs[j].error = 0;
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

            net->array_hidden[l][j].weighted_sum = 0;
            net->array_hidden[l][j].activation = 0;
            net->array_hidden[l][j].error = 0;
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

        net->array_outputs[j].weighted_sum = 0;
        net->array_outputs[j].activation = 0;
        net->array_outputs[j].error = 0;
    }
}

/*
 *  This functions creates filename string of an image
 *
 *  @param const char *filename, int number, int flag
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
 *  This functions sets the activations of all the sensors in the network, and the label
 *
 *  @param Neural_Network *net, const char *labelInfo, int number, int flag
 */
void setInput(Neural_Network *net, const char *labelInfo, int number, int flag)
{

    char tifInfo[41];

    getFilename(tifInfo, number, flag);

    FILE *image = fopen(tifInfo, "r");

    for (int i = 0; i < inputs; i++)
    {
        fscanf(image, "%f", &net->array_sensors[i].activation);
    }

    fclose(image);

    FILE *label = fopen(labelInfo, "r");

    int trash;

    for (int j = 1; j < number; j++)
    {
        fscanf(label, "%d", &trash);
    }

    fscanf(label, "%d", &net->label);

    fclose(label);
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

        net->array_inputs[j].activation = sigmoid(sum);
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

                net->array_hidden[l][j].activation = sigmoid(sum);
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

            net->array_outputs[j].activation = sigmoid(sum);
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

            net->array_outputs[j].activation = sigmoid(sum);
        }
    }
}

/*
 *  
 *
 *  @param Neural_Network *net
 */
void feedBackward(Neural_Network *net)
{

    // Calculate output neurons error
    for (int j = 0; j < outputs; j++)
    {
        if (j == net->label)
        {
            net->array_outputs[j].error = net->array_outputs[j].activation - 1;
        }
        else
        {
            net->array_outputs[j].error = net->array_outputs[j].activation - 0;
        }
    }

    if (layers)
    {

        // Calculate hidden layer neurons error
        for (int l = layers - 1; l >= 0; l--)
        {
            for (int j = 0; j < layer_size; j++)
            {

                float error = 0;

                if (l == layers - 1)
                {
                    for (int k = 0; k < outputs; k++)
                    {
                        error += net->array_outputs[k].weights[j] * net->array_outputs[k].error;
                    }

                    error = error * sigmoidDerivative(net->array_hidden[l][j].weighted_sum);

                    net->array_hidden[l][j].error = error;
                }
                else
                {

                    for (int k = 0; k < layer_size; k++)
                    {
                        error += net->array_hidden[l + 1][k].weights[j] * net->array_hidden[l + 1][k].error;
                    }

                    error = error * sigmoidDerivative(net->array_hidden[l][j].weighted_sum);

                    net->array_hidden[l][j].error = error;
                }
            }
        }

        // Calculate input neurons error
        for (int j = 0; j < layer_size; j++)
        {

            float error = 0;

            for (int k = 0; k < layer_size; k++)
            {
                error += net->array_hidden[0][k].weights[j] * net->array_hidden[0][k].error;
            }

            error = error * sigmoidDerivative(net->array_inputs[j].weighted_sum);

            net->array_inputs[j].error = error;
        }
    }
    else
    {
        // Calculate input neurons error
        for (int j = 0; j < layer_size; j++)
        {

            float error = 0;

            for (int k = 0; k < outputs; k++)
            {
                error += net->array_outputs[k].weights[j] * net->array_outputs[k].error;
            }

            error = error * sigmoidDerivative(net->array_inputs[j].weighted_sum);

            net->array_inputs[j].error = error;
        }
    }
}

/*
 *  This function propagates values through the network
 *
 *  @param Neural_Network *net
 */
void propagate(Neural_Network *net)
{
    // Forward pass
    feedForward(net);

    // Backward pass
    feedBackward(net);
}

/*
 *  This performs Stochastic Gradient Descent on the network
 *
 *  @param Neural_Network *net
 */
void stochasticGradientDescent(Neural_Network *net)
{
    float input_errors[layer_size][batch_size];
    float hidden_errors[layers][layer_size][batch_size];
    float output_errors[outputs][batch_size];

    // for each epoch
    // for (int e = 0; e < max_epochs; e++)
    {
        // for each batch in epoch
        for (int b = 0; b < train_sessions; b++)
        {
            printf("Batch number: %03d\n", b + 1);

            // for each example in batch
            for (int x = 1; x <= batch_size; x++)
            {
                // Set Activation of input neurons
                setInput(net, train_label, (b * batch_size) + x, 0);

                // Propagate values through network
                propagate(net);

                // CARRY ON HERE //
            }
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

/*****************
 * Main Function *
 *****************/

int main(int argc, char const *argv[])
{
    // Create the Neural Network
    Neural_Network smarty_pants;

    // Randomize the Neural Network
    randomize(&smarty_pants);

    // Set input
    setInput(&smarty_pants, train_label, 1, 0);

    // Propagate values through the Neural Network
    propagate(&smarty_pants);

    // Debug
    debug(&smarty_pants);

    // Stochastic Gradient Descent
    // stochasticGradientDescent(&smarty_pants);
}
