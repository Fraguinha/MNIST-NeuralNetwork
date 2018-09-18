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
    double training_inputs[training][inputs];
    int training_labels[training];

    double testing_inputs[testing][inputs];
    int testing_labels[testing];

} Data;

/* Sensor */
typedef struct Sensor
{
    double activation;

} Sensor;

/* Neuron */
typedef struct Sensor_Neuron
{
    double bias;
    double weights[inputs];

    double weighted_sum;
    double activation;

    double bias_error[batch_size];
    double weights_error[inputs][batch_size];

} Sensor_Neuron;

/* Neuron */
typedef struct Neuron_Neuron
{
    double bias;
    double weights[layer_size];

    double weighted_sum;
    double activation;

    double bias_error[batch_size];
    double weights_error[layer_size][batch_size];

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

int check_file(const char *filename, int binFlag)
/*
 *  This functions checks if file exists
 *
 *  @param const char *filename, int binFlag
 *  @return int
 */
{
    FILE *file;

    if (binFlag)
    {

        if ((file = fopen(filename, "rb")))
        {
            fclose(file);

            return 1;
        }
    }
    else
    {
        if ((file = fopen(filename, "r")))
        {
            fclose(file);

            return 1;
        }
    }

    return 0;
}

void getFilename(char *filename, const char *directory, int number)
/*
 *  This functions creates filename string of an image
 *
 *  @param const char *filename, const char *directory, int number
 */
{
    filename[0] = '\0';

    char num[6];

    sprintf(num, "%05d", number);

    char *extension = ".txt";

    strcat(filename, directory);
    strcat(filename, num);
    strcat(filename, extension);
}

void loadData(Data *data, const char *dataDirectoryTraining, const char *labelInfoTraining, const char *dataDirectoryTesting, const char *labelInfoTesting)
/*
 *  This functions loads data into memory
 *
 *  @param Data *data, const char *dataDirectoryTraining, const char *labelInfoTraining, const char *dataDirectoryTesting, const char *labelInfoTesting
 */
{
    FILE *image, *label;

    char filename[41];

    if (check_file(labelInfoTraining, 0))
    {
        label = fopen(labelInfoTraining, "r");
    }
    else
    {
        printf("!! File not found: %s\n", filename);
        exit(1);
    }

    for (int x = 0; x < training; x++)
    {
        getFilename(filename, dataDirectoryTraining, x + 1);

        if (check_file(filename, 0))
        {
            image = fopen(filename, "r");
        }
        else
        {
            printf("!! File not found: %s\n", filename);
            exit(1);
        }

        for (int j = 0; j < inputs; j++)
        {
            fscanf(image, "%lf", &data->training_inputs[x][j]);
        }

        fclose(image);

        fscanf(label, "%d", &data->training_labels[x]);
    }

    fclose(label);

    if (check_file(labelInfoTesting, 0))
    {
        label = fopen(labelInfoTesting, "r");
    }
    else
    {
        printf("!! File not found: %s\n", filename);
        exit(1);
    }

    for (int x = 0; x < testing; x++)
    {
        getFilename(filename, dataDirectoryTesting, x + 1);

        if (check_file(filename, 0))
        {
            image = fopen(filename, "r");
        }
        else
        {
            printf("!! File not found: %s\n", filename);
            exit(1);
        }

        for (int j = 0; j < inputs; j++)
        {
            fscanf(image, "%lf", &data->testing_inputs[x][j]);
        }

        fclose(image);

        fscanf(label, "%d", &data->testing_labels[x]);
    }

    fclose(label);
}

void saveDataBin(Data *data, const char *filename)
/*
 *  This functions saves the network to a binary file
 *
 *  @param Data *data, const char *filename
 */
{
    FILE *fp = fopen(filename, "wb");

    if (fp != NULL)
    {
        fwrite(data, sizeof(Data), 1, fp);
    }

    fclose(fp);
}

void loadDataBin(Data *data, const char *filename)
/*
 *  This functions loads the network from a binary file
 *
 *  @param Data *data, const char *filename
 */
{
    FILE *fp = fopen(filename, "rb");

    if (fp != NULL)
    {
        fread(data, sizeof(Data), 1, fp);
    }

    fclose(fp);
}

void saveNetworkBin(Neural_Network *net, const char *filename)
/*
 *  This functions saves the network to a binary file
 *
 *  @param Neural_Network *net, const char *filename
 */
{
    FILE *fp = fopen(filename, "wb");

    if (fp != NULL)
    {
        fwrite(net, sizeof(Neural_Network), 1, fp);
    }

    fclose(fp);
}

void loadNetworkBin(Neural_Network *net, const char *filename)
/*
 *  This functions loads the network from a binary file
 *
 *  @param Neural_Network *net, const char *filename
 */
{
    FILE *fp = fopen(filename, "rb");

    if (fp != NULL)
    {
        fread(net, sizeof(Neural_Network), 1, fp);
    }

    fclose(fp);
}

double randomizedDouble(double minimum, double maximum)
/*
 *  This function generates a random double
 *
 *  @param double minimum, double maximum
 *  @return double
 */
{
    return (((double)rand()) / (double)(RAND_MAX)) * (maximum - minimum) + minimum;
}

double randomNormalDistribution(double mean, double variation)
/*
 *  This function generates random distribution of doubles within certain mean and variation
 *
 *  @param double mean, double variation
 *  @return double
 */
{
    static int flag = 0;

    double x1, x2, w, y1;
    static double y2;

    if (flag)
    {
        y1 = y2;
        flag = 0;
    }
    else
    {
        do
        {
            x1 = 2.0 * randomizedDouble(0.0, 1.0) - 1.0;
            x2 = 2.0 * randomizedDouble(0.0, 1.0) - 1.0;

            w = x1 * x1 + x2 * x2;

        } while (w >= 1.0);

        w = sqrt((-2.0 * log(w)) / w);

        y1 = x1 * w;
        y2 = x2 * w;

        flag = 1;
    }

    return (mean + y1 * variation);
}

void randomize(Neural_Network *net)
/*
 *  This functions randomizes all the weights and biases of the neurons in the network
 *
 *  @param Neural_Network *net
 */
{
    // Make randomizer
    srand((unsigned int)time(NULL));

    // Randomize first neurons weights and biases
    for (int j = 0; j < layer_size; j++)
    {
        net->array_first[j].bias = randomNormalDistribution(0.0, 1.0);

        for (int k = 0; k < inputs; k++)
        {
            net->array_first[j].weights[k] = randomNormalDistribution(0.0, 1.0) / sqrt(inputs);
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
                    net->array_hidden[l][j].weights[k] = randomNormalDistribution(0.0, 1.0) / sqrt(layer_size);
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
            net->array_outputs[j].weights[k] = randomNormalDistribution(0.0, 1.0) / sqrt(layer_size);
        }
    }
}

void shuffle(Data *data)
/*
 *  This functions randomizes the data order
 *
 *  @param Neural_Network *net
 */
{
    // Make randomizer
    srand((unsigned int)time(NULL));

    double tempArray[inputs];
    int temp, random;

    for (int x = 0; x < training - 2; x++)
    {
        random = rand() % training;

        for (int j = 0; j < inputs; j++)
        {
            tempArray[j] = data->training_inputs[x][j];

            data->training_inputs[x][j] = data->training_inputs[random][j];
            data->training_inputs[random][j] = tempArray[j];
        }

        temp = data->training_labels[x];

        data->training_labels[x] = data->training_labels[random];
        data->training_labels[random] = temp;
    }
}

void setInput(Neural_Network *net, Data *data, int number, int trainFlag)
/*
 *  This functions sets the activations of all the sensors in the network, and the label
 *
 *  @param Neural_Network *net, Data *data, int number, int trainFlag
 */
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

double activation(double x)
/*
 *  The activation function
 *
 *  @param double x
 *  @return double activation
 */
{
    return 1.0 / (1.0 + exp(-(x)));
}

double activationDerivative(double x)
/*
 *  The activation derivative function
 *
 *  @param double x
 *  @return double activation
 */
{
    return activation(x) * (1.0 - activation(x));
}

void feedForward(Neural_Network *net)
/*
 *  This functions calculates the activation of all the neurons in the network
 *  activation[l] = activation( weighted sum[l] ) => activation( sum( weights[l] * activations[l-1] ) - bias[l] ) )
 *
 *  @param Neural_Network *net
 */
{
    double sum;

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

void feedBackward(Neural_Network *net, int x)
/*
 *  This function propagates backward the error on all neurons in the network
 *  (error[l] = (weights[l+1] * error[l+1]) * derivativeOfActivation(weighted_sum[l])
 *
 *  @param Neural_Network *net, int x
 */
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
                double bias_error = 0.0;

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
            double bias_error = 0.0;

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
            double bias_error = 0.0;

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

void update(Neural_Network *net)
/*
 *  This function updates the weights and biases
 *
 *  @param Neural_Network *net
 */
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

void stochasticGradientDescent(Neural_Network *net, Data *data)
/*
 *  This function performs Stochastic Gradient Descent on the network
 *  It is the learning algorithm
 *
 *  @param Neural_Network *net, Data *data
 */
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
    }
}

void setPrediction(Neural_Network *net)
/*
 *  This function calculates the network prediction
 *
 *  @param Neural_Network *net
 */
{
    double max = net->array_outputs[0].activation;

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

void score(Neural_Network *net, Data *data)
/*
 *  This function tests how good the network is on the testing data
 *
 *  @param Neural_Network *net, Data *data
 */
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

    double precision = ((double)correct / (double)testing) * 100;

    printf("Neural Network score: %5d / %5d (%5.2f%%)\n", correct, testing, precision);
}

void predict(Neural_Network *net, Data *data, int argc, char const *argv[])
/*
 *  This function classifies specific images
 *
 *  @param Neural_Network *net, Data *data, int argc, char const *argv[]
 */
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
    printf(":: Creating Neural Network\n");
    // Create the Neural Network
    Neural_Network *net = malloc(sizeof(Neural_Network));

    printf(":: Creating Data Structure\n");
    // Create the Data Structure
    Data *data = malloc(sizeof(Data));

    if (argc == 1)
    {
        if (check_file("data.bin", 1))
        {
            printf(":: Loading data\n");
            // Load the data
            loadDataBin(data, "data.bin");
        }
        else
        {
            printf(":: Loading data...\n");
            // Load the Data
            loadData(data, train_directory, train_label, test_directory, test_label);

            printf(":: Saving data\n");
            // Save the Data
            saveDataBin(data, "data.bin");
        }

        printf(":: Initializing Neural Network\n");
        // Randomize the Neural Network
        randomize(net);

        printf(":: Initializing Learning...\n");
        // Stochastic Gradient Descent
        stochasticGradientDescent(net, data);

        printf(":: Saving Neural Network\n");
        // Save the Neural Network
        saveNetworkBin(net, "custom.bin");
    }
    else
    {
        if (check_file("data.bin", 1))
        {
            printf(":: Loading data\n");
            // Load the data
            loadDataBin(data, "data.bin");
        }
        else
        {
            printf(":: Loading data...\n");
            // Load the Data
            loadData(data, train_directory, train_label, test_directory, test_label);

            printf(":: Saving data\n");
            // Save the Data
            saveDataBin(data, "data.bin");
        }

        if (check_file(argv[1], 1))
        {
            printf(":: Loading Neural Network\n");
            // Load the Neural Network
            loadNetworkBin(net, argv[1]);
        }
        else
        {
            printf("!! File not found: %s\n", argv[1]);
            exit(1);
        }
    }

    if (argc <= 2)
    {
        printf(":: Scoring Neural Network\n");
        // Test Neural Network
        score(net, data);
    }
    else
    {
        printf(":: Predicting Images\n");
        // predict Images
        predict(net, data, argc, argv);
    }

    free(data);
    free(net);

    exit(0);
}
