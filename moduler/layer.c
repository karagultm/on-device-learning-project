#include <stdlib.h>
#include <stdio.h>
#include "layer.h"

Data create_layer(int size, int dimension)
{
    Data layer ;
    layer.values = malloc(size * sizeof(double));
    return layer;
}

void free_layer(Data layer)
{
    free(layer.values); 
}

void log_layer_details(Data layer, const char *layer_name, FILE *file, int size)
{
    fprintf(file, "\n%s Details:\n", layer_name);
    fprintf(file, "Data: \n");
    for (int i = 0; i < size; i++)
    {
        fprintf(file, "[%d]: %f\n", i, layer.values[i]);
    }
    fprintf(file, "\n");
}
