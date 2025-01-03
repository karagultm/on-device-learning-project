
#include "dense_layer.h"

void dense(Data *input, Data output, DenseLayer *dense_layer, int input_size, int output_size, char activation[])
{


    output.values[0] = 0.0;
    for (int i = 0; i < dense_layer->input_size; i++)
    {
        output.values[0] += input->values[i] * dense_layer->weights[i];
    }
    output.values[0] += dense_layer->bias;
    if(strcmp(activation, "relu") == 0)
    {
        output.values[0] = relu(output.values[0]);
    }

}

DenseLayer *create_dense_layer(int input_size, int output_size)
{
    DenseLayer *dense_layer = malloc(sizeof(DenseLayer));
    dense_layer->weights = malloc(input_size * sizeof(double));

    for (int i = 0; i < input_size; i++)
    {
        dense_layer->weights[i] = (double)rand() / RAND_MAX - 0.5;
    }

    dense_layer->bias = (double)rand() / RAND_MAX - 0.5;

    dense_layer->input_size = input_size;
    dense_layer->output_size = output_size;

    return dense_layer;
}

void log_dense_layer_details(DenseLayer *layer, FILE *file)
{
    fprintf(file, "\nDense Layer Details:\n");
    fprintf(file, "Weights: \n");
    for (int i = 0; i < layer->input_size; i++)
    {
        fprintf(file, "[%d]: %f\n", i, layer->weights[i]);
    }
    fprintf(file, "Bias: %f\n", layer->bias);
    fprintf(file, "\n");
}

void free_dense_layer(DenseLayer *layer)
{
    free(layer->weights);
    free(layer);
}