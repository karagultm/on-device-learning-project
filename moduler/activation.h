#ifndef ACTIVATION_H
#define ACTIVATION_H
#include "sensor_data.h"

double relu(double x);
double sigmoid(double x);
void softmax(Data *input, Data output, int input_size,int i) ;
#endif