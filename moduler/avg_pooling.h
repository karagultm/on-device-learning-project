#ifndef AVG_POOLING_H
#define AVG_POOLING_H
#include "layer.h"

Data avg_pooling(Data input, int pool_size, int stride, int input_size, int output_size, int output_dimension);

#endif