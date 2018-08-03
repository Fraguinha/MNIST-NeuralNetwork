/*
    neural_network.c
    
    This program creates and trains a Neural Network using Stochastic Gradient Decent.
    
    compilation: gcc neural_network.c -o neural_network.exe -lm
    
    usage: ./neural_network.exe                             // Creates new network, trains, and tests it on all test images
    usage: ./neural_network.exe [neural_network.bin]        // Uses specified network, and tests it on all test images
    usage: ./neural_network.exe [neural_network.bin] [...]  // Uses specified network, and tests specified test images
    
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

#define layer_size 30                          // number of neurons for each layer
#define layers 0                               // number of hidden layers
#define layer_size_h (layers ? layer_size : 0) // number of hidden layer neurons

#define batch_size 10                          // number of examples per batch
#define train_sessions (training / batch_size) // number of training sessions per epoch

#define max_epochs 30             // maximum number of epochs
#define learning 3.0 / batch_size // how much to adjust

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
    sensor array_inputs[inputs];                      // sensors
    sensor_neuron array_first[layer_size];            // sensor neurons
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
 *  The activation function
 *
 *  @param float x
 *  @return float activation
 */
float activation(float x)
{
    return 1.0 / (1.0 + expf(-(x)));
}

/*
 *  The activation derivative function
 *
 *  @param float x
 *  @return float activation
 */
float activationDerivative(float x)
{
    return activation(x) * (1.0 - activation(x));
}

/*
 *  This function generates a random float
 *
 *  @param float minimum, float maximum
 *  @return float
 */
float randomizedFloat(float minimum, float maximum)
{
    return (((float)rand()) / (float)(RAND_MAX)) * (maximum - minimum) + minimum;
}

/*
 *  This function generates random distribution of floats within certain mean and variation
 * 
 *  @param float mean, float variation
 *  @return float
 */
float randomNormalDistribution(float mean, float variation)
{
    static int flag = 0;

    float x1, x2, w, y1;
    static float y2;

    if (flag)
    {
        y1 = y2;
        flag = 0;
    }
    else
    {
        do
        {
            x1 = 2.0 * randomizedFloat(0.0, 1.0) - 1.0;
            x2 = 2.0 * randomizedFloat(0.0, 1.0) - 1.0;

            w = x1 * x1 + x2 * x2;

        } while (w >= 1.0);

        w = sqrt((-2.0 * log(w)) / w);

        y1 = x1 * w;
        y2 = x2 * w;

        flag = 1;
    }

    return (mean + y1 * variation);
}

/*
 *  This functions randomizes all the weights and biases of the neurons in the network
 *
 *  @param Neural_Network *net
 */
void randomize(Neural_Network *net)
{
    // Make first randomizer
    srand((unsigned int)time(NULL));

    // Make random seed
    long int seed = rand();

    // Update randomizer with seed
    srand((unsigned int)time(&seed));

    // Randomize first neurons weights and biases
    for (int j = 0; j < layer_size; j++)
    {
        net->array_first[j].bias = randomNormalDistribution(0.0, 1.0);

        for (int k = 0; k < inputs; k++)
        {
            net->array_first[j].weights[k] = randomNormalDistribution(0.0, 1.0) / sqrtf(inputs);
        }
    }

    // Randomize hidden layer neurons weights and biases
    if (layers)
    {
        for (int l = 0; l < layers; l++)
        {
            for (int j = 0; j < layer_size; j++)
            {
                net->array_hidden[l][j].bias = randomNormalDistribution(0.0, 1.0);

                for (int k = 0; k < layer_size; k++)
                {
                    net->array_hidden[l][j].weights[k] = randomNormalDistribution(0.0, 1.0) / sqrtf(layer_size);
                }
            }
        }
    }

    // Randomize output neurons weights and biases
    for (int j = 0; j < outputs; j++)
    {
        net->array_outputs[j].bias = randomNormalDistribution(0.0, 1.0);

        for (int k = 0; k < layer_size; k++)
        {
            net->array_outputs[j].weights[k] = randomNormalDistribution(0.0, 1.0) / sqrtf(layer_size);
        }
    }
}

/*
 *  This functions creates filename string of an image
 *
 *  @param const char *filename, const char *directory, int number
 */
void getFilename(char *filename, const char *directory, int number)
{
    filename[0] = '\0';

    char num[6];

    sprintf(num, "%05d", number);

    char *extension = ".txt";

    strcat(filename, directory);
    strcat(filename, num);
    strcat(filename, extension);
}

/*
 *  This functions sets the activations of all the sensors in the network, and the label
 *
 *  @param Neural_Network *net, const char *dataDirectory, const char *labelInfo, int number
 */
void setInput(Neural_Network *net, const char *dataDirectory, const char *labelInfo, int number)
{
    char tifInfo[41];

    getFilename(tifInfo, dataDirectory, number);

    FILE *image = fopen(tifInfo, "r");

    for (int i = 0; i < inputs; i++)
    {
        fscanf(image, "%f", &net->array_inputs[i].activation);
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
 *  This function calculates the network prediction
 *
 *  @param Neural_Network *net
 */
void setPrediction(Neural_Network *net)
{
    float max = net->array_outputs[0].activation;

    int prediction = 0;

    for (int j = 1; j < outputs; j++)
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
 *  This functions calculates the activation of all the neurons in the network
 *  activation[l] = activation( weighted sum[l] ) => activation( sum( weights[l] * activations[l-1] ) - bias[l] ) )
 *
 *  @param Neural_Network *net
 */
void feedForward(Neural_Network *net)
{
    float sum;

    // Calculate first neurons activation
    for (int j = 0; j < layer_size; j++)
    {
        sum = 0.0;

        for (int k = 0; k < inputs; k++)
        {
            sum += net->array_inputs[k].activation * net->array_first[j].weights[k];
        }

        sum += net->array_first[j].bias;

        net->array_first[j].weighted_sum = sum;

        net->array_first[j].activation = activation(sum);
    }

    if (layers)
    {
        // Calculate hidden layer neurons activation
        for (int l = 0; l < layers; l++)
        {
            for (int j = 0; j < layer_size; j++)
            {
                sum = 0.0;

                if (l == 0)
                {
                    for (int k = 0; k < layer_size; k++)
                    {
                        sum += net->array_first[k].activation * net->array_hidden[l][j].weights[k];
                    }
                }
                else
                {
                    for (int k = 0; k < layer_size; k++)
                    {
                        sum += net->array_hidden[l - 1][k].activation * net->array_hidden[l][j].weights[k];
                    }
                }

                sum += net->array_hidden[l][j].bias;

                net->array_hidden[l][j].weighted_sum = sum;

                net->array_hidden[l][j].activation = activation(sum);
            }
        }

        // Calculate output neurons activation
        for (int j = 0; j < outputs; j++)
        {
            sum = 0.0;

            for (int k = 0; k < layer_size; k++)
            {
                sum += net->array_hidden[layers - 1][k].activation * net->array_outputs[j].weights[k];
            }

            sum += net->array_outputs[j].bias;

            net->array_outputs[j].weighted_sum = sum;

            net->array_outputs[j].activation = activation(sum);
        }
    }
    else
    {
        // Calculate output neurons activation
        for (int j = 0; j < outputs; j++)
        {
            sum = 0.0;

            for (int k = 0; k < layer_size; k++)
            {
                sum += net->array_first[k].activation * net->array_outputs[j].weights[k];
            }

            sum += net->array_outputs[j].bias;

            net->array_outputs[j].weighted_sum = sum;

            net->array_outputs[j].activation = activation(sum);
        }
    }
}

/*
 *  This function propagates backward the error on all neurons in the network
 *  (error[l] = (weights[l+1] * error[l+1]) * derivativeOfActivation(weighted_sum[l])
 *
 *  @param Neural_Network *net, int x
 */
void feedBackward(Neural_Network *net, int x)
{
    // Calculate output neurons error
    for (int j = 0; j < outputs; j++)
    {
        if (j == net->label)
        {
            net->array_outputs[j].bias_error[x] = (net->array_outputs[j].activation - 1.0) * activationDerivative(net->array_outputs[j].weighted_sum);
        }
        else
        {
            net->array_outputs[j].bias_error[x] = (net->array_outputs[j].activation - 0.0) * activationDerivative(net->array_outputs[j].weighted_sum);
        }

        if (layers)
        {
            for (int k = 0; k < layer_size; k++)
            {
                net->array_outputs[j].weights_error[k][x] = net->array_outputs[j].bias_error[x] * net->array_hidden[layers - 1][k].activation;
            }
        }
        else
        {
            for (int k = 0; k < layer_size; k++)
            {
                net->array_outputs[j].weights_error[k][x] = net->array_outputs[j].bias_error[x] * net->array_first[k].activation;
            }
        }
    }

    // Calculate hidden layer neurons error
    if (layers)
    {
        for (int l = layers - 1; l >= 0; l--)
        {
            for (int j = 0; j < layer_size; j++)
            {
                float bias_error = 0.0;

                if (l == layers - 1)
                {
                    for (int k = 0; k < outputs; k++)
                    {
                        bias_error += net->array_outputs[k].weights[j] * net->array_outputs[k].bias_error[x];
                    }

                    bias_error = bias_error * activationDerivative(net->array_hidden[l][j].weighted_sum);

                    net->array_hidden[l][j].bias_error[x] = bias_error;
                }
                else
                {
                    for (int k = 0; k < layer_size; k++)
                    {
                        bias_error += net->array_hidden[l + 1][k].weights[j] * net->array_hidden[l + 1][k].bias_error[x];
                    }

                    bias_error = bias_error * activationDerivative(net->array_hidden[l][j].weighted_sum);

                    net->array_hidden[l][j].bias_error[x] = bias_error;
                }

                if (l == 0)
                {
                    for (int k = 0; k < layer_size; k++)
                    {
                        net->array_hidden[l][j].weights_error[k][x] = net->array_hidden[l][j].bias_error[x] * net->array_first[k].activation;
                    }
                }
                else
                {
                    for (int k = 0; k < layer_size; k++)
                    {
                        net->array_hidden[l][j].weights_error[k][x] = net->array_hidden[l][j].bias_error[x] * net->array_hidden[l - 1][k].activation;
                    }
                }
            }
        }

        // Calculate first neurons error
        for (int j = 0; j < layer_size; j++)
        {
            float bias_error = 0.0;

            for (int k = 0; k < layer_size; k++)
            {
                bias_error += net->array_hidden[0][k].weights[j] * net->array_hidden[0][k].bias_error[x];
            }

            bias_error = bias_error * activationDerivative(net->array_first[j].weighted_sum);

            net->array_first[j].bias_error[x] = bias_error;

            for (int k = 0; k < inputs; k++)
            {
                net->array_first[j].weights_error[k][x] = net->array_first[j].bias_error[x] * net->array_inputs[k].activation;
            }
        }
    }
    else
    {
        // Calculate first neurons error
        for (int j = 0; j < layer_size; j++)
        {
            float bias_error = 0.0;

            for (int k = 0; k < outputs; k++)
            {
                bias_error += net->array_outputs[k].weights[j] * net->array_outputs[k].bias_error[x];
            }

            bias_error = bias_error * activationDerivative(net->array_first[j].weighted_sum);

            net->array_first[j].bias_error[x] = bias_error;

            for (int k = 0; k < inputs; k++)
            {
                net->array_first[j].weights_error[k][x] = net->array_first[j].bias_error[x] * net->array_inputs[k].activation;
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
    // Adjust first
    for (int j = 0; j < layer_size; j++)
    {
        for (int x = 0; x < batch_size; x++)
        {
            net->array_first[j].bias = net->array_first[j].bias - (learning * net->array_first[j].bias_error[x]);

            for (int k = 0; k < inputs; k++)
            {
                net->array_first[j].weights[k] = net->array_first[j].weights[k] - (learning * net->array_first[j].weights_error[k][x]);
            }
        }
    }

    // Adjust hidden
    if (layers)
    {
        for (int l = 0; l < layers; l++)
        {
            for (int j = 0; j < layer_size; j++)
            {
                for (int x = 0; x < batch_size; x++)
                {
                    net->array_hidden[l][j].bias = net->array_hidden[l][j].bias - (learning * net->array_hidden[l][j].bias_error[x]);

                    for (int k = 0; k < layer_size; k++)
                    {
                        net->array_hidden[l][k].weights[k] = net->array_hidden[l][k].weights[k] - (learning * net->array_hidden[l][j].weights_error[k][x]);
                    }
                }
            }
        }
    }

    // Adjust outputs
    for (int j = 0; j < outputs; j++)
    {
        for (int x = 0; x < batch_size; x++)
        {
            net->array_outputs[j].bias = net->array_outputs[j].bias - (learning * net->array_outputs[j].bias_error[x]);

            for (int k = 0; k < layer_size; k++)
            {
                net->array_outputs[j].weights[k] = net->array_outputs[j].weights[k] - (learning * net->array_outputs[j].weights_error[k][x]);
            }
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

    for (int x = 1; x <= testing; x++)
    {
        // Set Activation of input neurons
        setInput(net, test_directory, test_label, x);

        // Propagate values forward
        feedForward(net);

        // Check prediction
        setPrediction(net);

        if (net->prediction == net->label)
        {
            correct++;
        }
    }

    float precision = ((float)correct / (float)testing) * 100;

    printf("Neural Network score: %5d / %5d (%5.2f%%)\n", correct, testing, precision);
}

/*
 *  This function classifies specific images
 *
 *  @param Neural_Network *net
 */
void scoreImages(Neural_Network *net, int argc, char const *argv[])
{
    for (int i = 2; i < argc; i++)
    {
        // Set Activation of input neurons
        setInput(net, test_directory, test_label, atoi(argv[i]));

        // Propagate values forward
        feedForward(net);

        // Check prediction
        setPrediction(net);

        // Print result
        printf("Neural Network prediction of image[%05d]: %d (%d is correct label)\n", atoi(argv[i]), net->prediction, net->label);
    }
}

/*
 *  This function performs Stochastic Gradient Descent on the network
 *  It is the learning algorithm
 *
 *  @param Neural_Network *net
 */
void stochasticGradientDescent(Neural_Network *net, int printFlag)
{
    // for each epoch
    for (int e = 1; e <= max_epochs; e++)
    {
        // for each batch in epoch
        for (int b = 0; b < train_sessions; b++)
        {
            // for each example in batch
            for (int x = 1; x <= batch_size; x++)
            {
                // Set Activation of input neurons
                setInput(net, train_directory, train_label, (b * batch_size) + x);

                // Forward pass
                feedForward(net);

                // Backward pass
                feedBackward(net, x - 1);
            }

            // Update parameters
            update(net);

            if (printFlag)
            {
                // Show status
                printf("Batch %02d: ", b + 1);

                // Score
                score(net);
            }
        }

        // Save the Neural Network
        save(net, "custom.bin");

        if (printFlag)
        {
            // Show status
            printf("Epoch %02d: ", e);

            // Score
            score(net);
        }
    }
}

/*****************
 * Main Function *
 *****************/

int main(int argc, char const *argv[])
{
    // Create the Neural Network
    static Neural_Network net;

    if (argc == 1)
    {
        // Randomize the Neural Network
        randomize(&net);

        // Save the Neural Network
        save(&net, "custom.bin");

        // Stochastic Gradient Descent
        stochasticGradientDescent(&net, 1);
    }
    else
    {
        // Load the Neural Network
        load(&net, argv[1]);
    }

    if (argc <= 2)
    {
        // Test Neural Network
        score(&net);
    }
    else
    {
        // Test Specific Images
        scoreImages(&net, argc, argv);
    }
}
