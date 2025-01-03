#ifndef LAYER_H
#define LAYER_H

#include "sensor_data.h"

typedef struct
{
    double *data;
    int size;
} Layer;

typedef enum
{
    CONV1D,
    AVGPOOL1D,
    GLOBALAVGPOOL1D,
    DENSE,
    SOFTMAX
} LayerType;

Data create_layer(int size, int dimension);
void free_layer(Data layer);
void log_layer_details(Data layer, const char *layer_name, FILE *file, int size);

#endif