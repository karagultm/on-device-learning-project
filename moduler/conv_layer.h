#ifndef CONV_LAYER_H
#define CONV_LAYER_H
#include "layer.h"
#include "activation.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct
{
    double **weights;
    double bias;
    int filter_size;
    int kernel_size;
    int input_size;
    int stride;
    int  input_dimension;

} ConvLayer;

ConvLayer *create_conv_layer(int input_size, int kernel_size, int stride, int filter_size, int input_dimension);

void free_conv_layer(ConvLayer *layer); 

void log_conv_layer_details(ConvLayer *layer, FILE *file);

void update_weights(ConvLayer *layer, double *gradients, double learning_rate);

Data convolve(Data *input, ConvLayer *conv_layer, int data_count,int output_size);

#endif