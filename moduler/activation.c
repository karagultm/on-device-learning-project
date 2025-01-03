#include <math.h>
#include "activation.h"

double relu(double x) {
    return x > 0 ? x : 0;
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

//softmax 
void softmax(Data *input, Data output, int input_size,int i) {  //abi bunllarda identifier hatası veriyor da düzgün çalışıyor wtf
    double sum = 0;
    for (int i = 0; i < input_size; i++) {
        sum += exp(input[i].values[0]);
    }
    
    output.values[0] = exp(input[i].values[0]) / sum;
    
}

