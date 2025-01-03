#include <stdlib.h>
#include <stdio.h>
#include "avg_pooling.h"

Data avg_pooling(Data input, int pool_size, int stride, int input_size, int output_size, int output_dimension) {

    Data output = create_layer(output_size,output_dimension);

    for (int i = 0; i < output_size; i++) {
        double sum = 0;
        for (int k = 0; k < pool_size; k++) {
            if (i * stride + k >= input_size) {
                printf("Index out of bounds: i = %d, k = %d\n", i, k);
                free_layer(output);
                Data empty = {NULL};
                return empty;
            }

            int current_idx = i * stride + k;
            sum += input.values[current_idx];
        }
        output.values[i] = sum / pool_size;
    }

    return output;
}