#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layer.h"
#include "activation.h"

typedef struct
{
    double *weights;
    double bias;
    int input_size;
    int output_size;
} DenseLayer;

void dense(Data *input, Data output, DenseLayer *dense_layer, int input_size, int output_size, char activation[]);
DenseLayer *create_dense_layer(int input_size, int output_size);
void log_dense_layer_details(DenseLayer *layer, FILE *file);
void free_dense_layer(DenseLayer *layer);
#endif