/*
    neural_network.c
    
    This program creates and trains a Neural Network using Stochastic Gradient Decent.
    
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
#define layers 0                               // number of hidden layers
#define layer_size_h (layers ? layer_size : 0) // number of hidden layer neurons

#define min_weight -2 // minimum value for the weights
#define max_weight +2 // maximum value for the weights

#define min_bias -1 // maximum value for the bias
#define max_bias +1 // maximum value for the bias

#define batch_size 100                         // number of examples per batch
#define train_sessions (training / batch_size) // number of training sessions per epoch

#define max_epochs 5               // maximum number of epochs
#define learning 0.15 / batch_size // learning rate

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

    float bias_error[batch_size];
    float weights_error[inputs][batch_size];

} sensor_neuron;

/* Neuron */
typedef struct
{
    float bias;
    float weights[layer_size];

    float weighted_sum;
    float activation;

    float bias_error[batch_size];
    float weights_error[layer_size][batch_size];

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
 *  (weighted sum[l] = sum(weights[l] * activations[l-1]) - bias[l])
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
 *  This function calculates the network prediction
 *
 *  @param Neural_Network *net
 */
void setPrediction(Neural_Network *net)
{
    float max = net->array_outputs[0].activation;

    int prediction = 0;

    for (int j = 0; j < outputs; j++)
    {
        if (net->array_outputs[j].activation > max)
        {
            max = net->array_outputs[j].activation;
            prediction = j;
        }
    }

    net->prediction = prediction;
}

/*
 *  This function propagates backward the error on all neurons in the network
 *  (error[l] = (weights[l+1] * error[l+1]) * derivativeOfActivation(weighted_sum[l])
 *
 *  @param Neural_Network *net
 */
void feedBackward(Neural_Network *net, int x)
{

    // Calculate output neurons error
    for (int j = 0; j < outputs; j++)
    {
        if (j == net->label)
        {
            net->array_outputs[j].bias_error[x] = (net->array_outputs[j].activation - 1.0) * sigmoidDerivative(net->array_outputs[j].weighted_sum);
        }
        else
        {
            net->array_outputs[j].bias_error[x] = (net->array_outputs[j].activation - 0.0) * sigmoidDerivative(net->array_outputs[j].weighted_sum);
        }

        for (int k = 0; k < layer_size; k++)
        {
            if (layers)
            {
                net->array_outputs[j].weights_error[k][x] = net->array_outputs[j].bias_error[x] * net->array_hidden[layers - 1][k].activation;
            }
            else
            {
                net->array_outputs[j].weights_error[k][x] = net->array_outputs[j].bias_error[x] * net->array_inputs[k].activation;
            }
        }
    }

    if (layers)
    {

        // Calculate hidden layer neurons error
        for (int l = layers - 1; l >= 0; l--)
        {
            for (int j = 0; j < layer_size; j++)
            {

                float bias_error = 0;

                if (l == layers - 1)
                {
                    for (int k = 0; k < outputs; k++)
                    {
                        bias_error += net->array_outputs[k].weights[j] * net->array_outputs[k].bias_error[x];
                    }

                    bias_error = bias_error * sigmoidDerivative(net->array_hidden[l][j].weighted_sum);

                    net->array_hidden[l][j].bias_error[x] = bias_error;
                }
                else
                {

                    for (int k = 0; k < layer_size; k++)
                    {
                        bias_error += net->array_hidden[l + 1][k].weights[j] * net->array_hidden[l + 1][k].bias_error[x];
                    }

                    bias_error = bias_error * sigmoidDerivative(net->array_hidden[l][j].weighted_sum);

                    net->array_hidden[l][j].bias_error[x] = bias_error;
                }

                for (int k = 0; k < layer_size; k++)
                {
                    if (l == 0)
                    {
                        net->array_hidden[l][j].weights_error[k][x] = net->array_hidden[l][j].bias_error[x] * net->array_inputs[k].activation;
                    }
                    else
                    {
                        net->array_hidden[l][j].weights_error[k][x] = net->array_hidden[l][j].bias_error[x] * net->array_hidden[l - 1][k].activation;
                    }
                }
            }
        }

        // Calculate input neurons error
        for (int j = 0; j < layer_size; j++)
        {
            float bias_error = 0;

            for (int k = 0; k < layer_size; k++)
            {
                bias_error += net->array_hidden[0][k].weights[j] * net->array_hidden[0][k].bias_error[x];
            }

            bias_error = bias_error * sigmoidDerivative(net->array_inputs[j].weighted_sum);

            net->array_inputs[j].bias_error[x] = bias_error;

            for (int k = 0; k < inputs; k++)
            {
                net->array_inputs[j].weights_error[k][x] = net->array_inputs[j].bias_error[x] * net->array_sensors[k].activation;
            }
        }
    }
    else
    {
        // Calculate input neurons error
        for (int j = 0; j < layer_size; j++)
        {

            float bias_error = 0;

            for (int k = 0; k < outputs; k++)
            {
                bias_error += net->array_outputs[k].weights[j] * net->array_outputs[k].bias_error[x];
            }

            bias_error = bias_error * sigmoidDerivative(net->array_inputs[j].weighted_sum);

            net->array_inputs[j].bias_error[x] = bias_error;

            for (int k = 0; k < inputs; k++)
            {
                net->array_inputs[j].weights_error[k][x] = net->array_inputs[j].bias_error[x] * net->array_sensors[k].activation;
            }
        }
    }
}

/*
 *  This function updates the weights and biases
 *
 *  @param Neural_Network *net
 */
void update(Neural_Network *net)
{
    // Calculate averages
    float average_input_bias[layer_size];
    float average_input_weights[layer_size][inputs];

    float average_hidden_bias[layers][layer_size];
    float average_hidden_weights[layers][layer_size][layer_size];

    float average_output_bias[outputs];
    float average_output_weights[outputs][layer_size];

    float sum_bias;
    float sum_weights;

    // Inputs
    for (int j = 0; j < layer_size; j++)
    {
        sum_bias = 0;

        for (int x = 0; x < batch_size; x++)
        {
            sum_bias += net->array_inputs[j].bias_error[x];
        }

        average_input_bias[j] = sum_bias / batch_size;

        for (int k = 0; k < inputs; k++)
        {
            sum_weights = 0;

            for (int x = 0; x < batch_size; x++)
            {
                sum_weights += net->array_inputs[j].weights_error[k][x];
            }

            average_input_weights[j][k] = sum_weights / batch_size;
        }
    }

    // Hidden
    if (layers)
    {
        for (int l = 0; l < layers; l++)
        {
            for (int j = 0; j < layer_size; j++)
            {
                sum_bias = 0;

                for (int x = 0; x < batch_size; x++)
                {
                    sum_bias += net->array_hidden[l][j].bias_error[x];
                }

                average_hidden_bias[l][j] = sum_bias / batch_size;

                for (int k = 0; k < layer_size; k++)
                {
                    sum_weights = 0;

                    for (int x = 0; x < batch_size; x++)
                    {
                        sum_weights += net->array_hidden[l][j].weights_error[k][x];
                    }

                    average_hidden_weights[l][j][k] = sum_weights / batch_size;
                }
            }
        }
    }

    // Outputs
    for (int j = 0; j < outputs; j++)
    {
        sum_bias = 0;

        for (int x = 0; x < batch_size; x++)
        {
            sum_bias += net->array_outputs[j].bias_error[x];
        }

        average_output_bias[j] = sum_bias / batch_size;

        for (int k = 0; k < layer_size; k++)
        {
            sum_weights = 0;

            for (int x = 0; x < batch_size; x++)
            {
                sum_weights += net->array_outputs[j].weights_error[k][x];
            }

            average_output_weights[j][k] = sum_weights / batch_size;
        }
    }

    // Adjust parameters
    for (int j = 0; j < layer_size; j++)
    {
        net->array_inputs[j].bias = net->array_inputs[j].bias - (learning * average_input_bias[j]);

        for (int k = 0; k < inputs; k++)
        {
            net->array_inputs[j].weights[k] = net->array_inputs[j].weights[k] - (learning * average_input_weights[j][k]);
        }
    }

    if (layers)
    {
        for (int l = 0; l < layers; l++)
        {
            for (int j = 0; j < layer_size; j++)
            {
                net->array_hidden[l][j].bias = net->array_hidden[l][j].bias - (learning * average_hidden_bias[l][j]);

                for (int k = 0; k < layer_size; k++)
                {
                    net->array_hidden[l][k].weights[k] = net->array_hidden[l][k].weights[k] - (learning * average_hidden_weights[l][j][k]);
                }
            }
        }
    }

    for (int j = 0; j < outputs; j++)
    {
        net->array_outputs[j].bias = net->array_outputs[j].bias - (learning * average_output_bias[j]);

        for (int k = 0; k < layer_size; k++)
        {
            net->array_outputs[j].weights[k] = net->array_outputs[j].weights[k] - (learning * average_output_weights[j][k]);
        }
    }
}

/*
 *  This function performs Stochastic Gradient Descent on the network
 *  It is the learning algorithm
 *
 *  @param Neural_Network *net
 */
void stochasticGradientDescent(Neural_Network *net)
{
    // for each epoch
    for (int e = 0; e < max_epochs; e++)
    {
        // for each batch in epoch
        for (int b = 0; b < train_sessions; b++)
        {
            // for each example in batch
            for (int x = 0; x < batch_size; x++)
            {
                // Set Activation of input neurons
                setInput(net, train_label, (b * batch_size) + x + 1, 0);

                // Forward pass
                feedForward(net);

                // Backward pass
                feedBackward(net, x);
            }

            // Update parameters
            update(net);
        }
    }
}

/*
 *  This function tests how good the network is on the testing data
 *
 *  @param Neural_Network *net
 */
void score(Neural_Network *net)
{
    int correct = 0;

    for (int x = 0; x < testing; x++)
    {
        // Set Activation of input neurons
        setInput(net, train_label, x + 1, 1);

        // Propagate values forward
        feedForward(net);

        // Check prediction
        setPrediction(net);

        if (net->prediction == net->label)
        {
            correct++;
        }
    }

    float precision = correct / testing;

    printf("Number of examples correctly classified: %d\n", correct);
    printf("Neural Network precision: %.2f%%\n", precision);
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

    if (argc == 1)
    {
        // Randomize the Neural Network
        randomize(&smarty_pants);

        // Stochastic Gradient Descent
        stochasticGradientDescent(&smarty_pants);

        // Save the Neural Network
        save(&smarty_pants, "smarty_pants.bin");
    }
    else
    {
        // Load the Neural Network
        load(&smarty_pants, argv[1]);
    }

    if (argc <= 2)
    {
        // Test Neural Network
        score(&smarty_pants);
    }
}
