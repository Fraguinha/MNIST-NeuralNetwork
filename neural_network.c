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

/* Data */
typedef struct Data
{
    float training_inputs[training][inputs];
    int training_labels[training];

    float testing_inputs[testing][inputs];
    int testing_labels[testing];

} Data;

/* Sensor */
typedef struct Sensor
{
    float activation;

} Sensor;

/* Neuron */
typedef struct Sensor_Neuron
{
    float bias;
    float weights[inputs];

    float weighted_sum;
    float activation;

    float bias_error[batch_size];
    float weights_error[inputs][batch_size];

} Sensor_Neuron;

/* Neuron */
typedef struct Neuron_Neuron
{
    float bias;
    float weights[layer_size];

    float weighted_sum;
    float activation;

    float bias_error[batch_size];
    float weights_error[layer_size][batch_size];

} Neuron_Neuron;

/* Neural Network */
typedef struct Neural_Network
{
    /* Structure */
    Sensor array_inputs[inputs];                      // sensors
    Sensor_Neuron array_first[layer_size];            // sensor neurons
    Neuron_Neuron array_hidden[layers][layer_size_h]; // layers of neuron neurons
    Neuron_Neuron array_outputs[outputs];             // output neurons

    /* Information */
    int prediction;
    int label;

} Neural_Network;

/************************
 * Function Definitions *
 ************************/

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
 *  This functions loads data into memory
 *
 *  @param Data *data, const char *dataDirectoryTraining, const char *labelInfoTraining, const char *dataDirectoryTesting, const char *labelInfoTesting
 */
void loadData(Data *data, const char *dataDirectoryTraining, const char *labelInfoTraining, const char *dataDirectoryTesting, const char *labelInfoTesting, int trainFlag)
{
    FILE *image, *label;

    char filename[41];

    if (trainFlag)
    {
        label = fopen(labelInfoTraining, "r");

        for (int x = 0; x < training; x++)
        {
            getFilename(filename, dataDirectoryTraining, x + 1);
            image = fopen(filename, "r");

            for (int j = 0; j < inputs; j++)
            {
                fscanf(image, "%f", &data->training_inputs[x][j]);
            }

            fclose(image);

            fscanf(label, "%d", &data->training_labels[x]);
        }

        fclose(label);
    }

    label = fopen(labelInfoTesting, "r");

    for (int x = 0; x < testing; x++)
    {
        getFilename(filename, dataDirectoryTesting, x + 1);
        image = fopen(filename, "r");

        for (int j = 0; j < inputs; j++)
        {
            fscanf(image, "%f", &data->testing_inputs[x][j]);
        }

        fclose(image);

        fscanf(label, "%d", &data->testing_labels[x]);
    }

    fclose(label);
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
    // Make randomizer
    srand((unsigned int)time(NULL));

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
 *  This functions randomizes the data order
 *
 *  @param Neural_Network *net
 */
void shuffle(Data *data)
{
    // Make randomizer
    srand((unsigned int)time(NULL));

    float temp1[inputs];
    int temp2, r;

    for (int x = 0; x < training - 2; x++)
    {
        r = rand() % training;

        for (int j = 0; j < inputs; j++)
        {
            temp1[j] = data->training_inputs[x][j];

            data->training_inputs[x][j] = data->training_inputs[r][j];
            data->training_inputs[r][j] = temp1[j];
        }

        temp2 = data->training_labels[x];

        data->training_labels[x] = data->training_labels[r];
        data->training_labels[r] = temp2;
    }
}

/*
 *  This functions sets the activations of all the sensors in the network, and the label
 *
 *  @param Neural_Network *net, Data *data, int number, int trainFlag
 */
void setInput(Neural_Network *net, Data *data, int number, int trainFlag)
{
    if (trainFlag)
    {
        for (int j = 0; j < inputs; j++)
        {
            net->array_inputs[j].activation = data->training_inputs[number - 1][j];
        }

        net->label = data->training_labels[number - 1];
    }
    else
    {
        for (int j = 0; j < inputs; j++)
        {
            net->array_inputs[j].activation = data->testing_inputs[number - 1][j];
        }

        net->label = data->testing_labels[number - 1];
    }
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
 *  This function performs Stochastic Gradient Descent on the network
 *  It is the learning algorithm
 *
 *  @param Neural_Network *net, Data *data
 */
void stochasticGradientDescent(Neural_Network *net, Data *data)
{
    // for each epoch
    for (int e = 0; e < max_epochs; e++)
    {
        // Shuffle data
        shuffle(data);

        // for each batch in epoch
        for (int b = 0; b < train_sessions; b++)
        {
            // for each example in batch
            for (int x = 0; x < batch_size; x++)
            {
                // Set activation of input neurons
                setInput(net, data, 1 + (b * batch_size) + x, 1);

                // Forward pass
                feedForward(net);

                // Backward pass
                feedBackward(net, x);
            }

            // Update parameters
            update(net);
        }

        // Save the Neural Network
        save(net, "custom.bin");
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
 *  This function tests how good the network is on the testing data
 *
 *  @param Neural_Network *net, Data *data
 */
void score(Neural_Network *net, Data *data)
{
    int correct = 0;

    for (int x = 1; x <= testing; x++)
    {
        // Set Activation of input neurons
        setInput(net, data, x, 0);

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
 *  @param Neural_Network *net, Data *data, int argc, char const *argv[]
 */
void scoreImages(Neural_Network *net, Data *data, int argc, char const *argv[])
{
    for (int x = 2; x < argc; x++)
    {
        // Set Activation of input neurons
        setInput(net, data, atoi(argv[x]), 0);

        // Propagate values forward
        feedForward(net);

        // Check prediction
        setPrediction(net);

        // Print result
        printf("Neural Network prediction of image[%05d]: %d (%d is correct label)\n", atoi(argv[x]), net->prediction, net->label);
    }
}

/*****************
 * Main Function *
 *****************/

int main(int argc, char const *argv[])
{
    // Create the Neural Network
    Neural_Network *net = malloc(sizeof(Neural_Network));

    // Create the Data Structure
    Data *data = malloc(sizeof(Data));

    if (argc == 1)
    {
        // Load all the data
        loadData(data, train_directory, train_label, test_directory, test_label, 1);

        // Randomize the Neural Network
        randomize(net);

        // Stochastic Gradient Descent
        stochasticGradientDescent(net, data);
    }
    else
    {
        // Load the test data
        loadData(data, train_directory, train_label, test_directory, test_label, 0);

        // Load the Neural Network
        load(net, argv[1]);
    }

    if (argc <= 2)
    {
        // Test Neural Network
        score(net, data);
    }
    else
    {
        // Test Specific Images
        scoreImages(net, data, argc, argv);
    }
}
