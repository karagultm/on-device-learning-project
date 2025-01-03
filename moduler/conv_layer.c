
#include "conv_layer.h"

ConvLayer *create_conv_layer(int input_size, int kernel_size, int stride, int filter_size, int input_dimension)
{
    ConvLayer *layer = malloc(sizeof(ConvLayer)); 
    // İki boyutlu array için memory allocation
    layer->weights = malloc(input_dimension * sizeof(double *));
    for (int i = 0; i < input_dimension; i++)
    {
        layer->weights[i] = malloc(kernel_size * sizeof(double));
    }
    for (int c = 0; c < input_dimension; c++)
    {
        for (int k = 0; k < kernel_size; k++)
        {
            layer->weights[c][k] = (double)rand() / RAND_MAX - 0.5;
        }
    }
    // layer->weights[0][0] = 0.54; // x1 in weighti
    // layer->weights[0][1] = -0.31;
    // layer->weights[0][2] = 0.15;

    // layer->weights[1][0] = -0.9812; // y1 in weighti
    // layer->weights[1][1] = 0.123;
    // layer->weights[1][2] = -0.861;

    // layer->weights[2][0] = 0.123;
    // layer->weights[2][1] = -0.331;
    // layer->weights[2][2] = -0.0032;

    layer->bias = (double)rand() / RAND_MAX - 0.5;
    // layer->bias = 0.189632;
    layer->kernel_size = kernel_size;
    layer->input_size = input_size;
    layer->stride = stride;
    layer->filter_size = filter_size; // bence filter size çok gereksiz kalıyor buraya
    layer->input_dimension = input_dimension;
    return layer;
}

Data convolve(Data *input, ConvLayer *conv_layer, int data_count, int output_size)
{
    int output_dimension = conv_layer->input_dimension;
    Data output = create_layer(output_size, output_dimension); // benim bunu freelemem gerekiyor mu ????????
    // bunu şu şekilde çözebilrim outputu parameter olarak alırım ve fonkisyonu void yaparım.

    for (int i = 0; i < output_size; i++)
    {
        double sum = 0;
        for (int c = 0; c < conv_layer->input_dimension; c++)
        {
            for (int k = 0; k < conv_layer->kernel_size; k++)
            {
                if (i * conv_layer->stride + k >= data_count) //değişken atama
                {
                    printf("Index out of bounds: i = %d, k = %d\n", i, k);
                    free_layer(output);
                    // Hata durumunda boş Data döndür
                    Data empty = {NULL};
                    return empty;
                }

                int current_idx = i * conv_layer->stride + k;

                double temp = input[c].values[current_idx];
                double temp2 = conv_layer->weights[c][k]; //
                sum += temp * temp2;
                // sum += input->data[c][current_idx] * conv_layer->weights[c][k];
            }
        }
        sum += conv_layer->bias;
        output.values[i] = relu(sum);
    }

    return output;
}

void free_conv_layer(ConvLayer *layer)
{
    // Önce her bir row'u free et
    for (int i = 0; i < layer->input_dimension; i++)
    {
        free(layer->weights[i]);
    }
    // Sonra pointer array'ini free et
    free(layer->weights);
    // En son layer'ın kendisini free et
    free(layer);
}

void log_conv_layer_details(ConvLayer *layer, FILE *file)
{
    fprintf(file, "\n======================== Convolution Layer Details ========================\n");
    fprintf(file, "Input Size : %d\n", layer->input_size);
    fprintf(file, "Kernel Size: %d\n", layer->kernel_size);
    fprintf(file, "Stride     : %d\n", layer->stride);
    fprintf(file, "Filter Size: %d\n", layer->filter_size);
    fprintf(file, "Bias       : %.6f\n", layer->bias);

    fprintf(file, "\n----------------------------- Weights -----------------------------------\n");
    fprintf(file, "Input Dimension | Index |   Weight     \n");
    fprintf(file, "-------------------------------------------------------------\n");
    for (int c = 0; c < layer->input_dimension; c++)
    {
        for (int i = 0; i < layer->kernel_size; i++)
        {
            fprintf(file, "%15d | %5d | %13.6f\n", c,
                    i, layer->weights[c][i]);
        }
    }
    fprintf(file, "==========================================================================\n\n");
}
