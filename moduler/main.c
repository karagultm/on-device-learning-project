#include <stdio.h>
#include <stdlib.h>

#include "layer.c"
#include "sensor_data.c"
#include "conv_layer.c"
#include "activation.c"
#include "avg_pooling.c"
#include "dense_layer.c"

#define MAX_LINES 1024
// --- Sabitler ---
#define INPUT_SIZE 120       // Giriş uzunluğu (N)
#define INPUT_CHANNELS 3     // Kanal sayısı (C)
#define DENSE_UNITS_1 15     // Dense layer birimleri (M)
#define DENSE_UNITS_2 6      // Dense layer birimleri (M)
#define FIRST_FILTER_SIZE 32 // Conv1D filtre sayısı (F)
#define KERNEL_SIZE 3        // Conv1D çekirdek boyutu (K)
#define STRIDE 1             // Conv1D adım boyutu (p)
#define POOL_SIZE 2
#define POOL_STRIDE 2

#define FILTER_SIZE_32 32
#define FILTER_SIZE_64 64
#define KERNEL_SIZE_3 3
#define STRIDE_1 1
#define POOL_SIZE_2 2
#define POOL_STRIDE_2 2
#define DENSE_UNITS_50 50
#define DENSE_UNITS_6 6

int main()
{
    printf("Convolutional Neural Network\n");
    FILE *log_file = fopen("cnn_1_log.txt", "w");
    if (log_file == NULL)
    {
        printf("Error opening log file!\n");
        return 1;
    }

    srand(42);

    SensorData sensor_data[MAX_LINES];
    int sensor_data_count = read_sensor_data("train_data_2.txt", sensor_data, MAX_LINES);
    log_sensor_data(sensor_data, sensor_data_count, log_file);
    printf("Sensor data read complete.\n");

    Data *data = malloc(sensor_data_count * sizeof(Data)); // freelemem gerekiyor
    for (int i = 0; i < sensor_data_count; i++)
    {
        data[i] = sensor_data[i].data;
    }
    // şuan bu işlem ile birlikte ben sensordatadaki dataları data ya yapışyırmış bulunuyorum

    int kernel_size = 3;
    int filter_size = 6;
    int conv_stride = 1;
    int input_dimension = 3; // x, y, z

    int pool_size = 2;
    int pool_stride = 2;
    int output_size = (sensor_data_count - kernel_size) / conv_stride + 1;
    int avg_pool_output_size = (output_size - pool_size) / pool_stride + 1;
    Data *conv_output = malloc(filter_size * sizeof(Data));
    Data *avg_pool_output = malloc(filter_size * sizeof(Data));
    printf("Starting Convolution and Average Pooling...\n");
    for (int j = 0; j < filter_size; j++)
    {

        // bu create işlemleri convolve içerisnde olabilir gibi duruyor emin değilim. bu adımlar daha düzenli yapılabilir açıkçası.

        ConvLayer *conv_layer = create_conv_layer(sensor_data_count, kernel_size, conv_stride, filter_size, input_dimension);
        log_conv_layer_details(conv_layer, log_file);

        conv_output[j] = convolve(data, conv_layer, sensor_data_count, output_size);

        char layer_name[50];
        sprintf(layer_name, "Output Layer %d", j + 1);
        log_layer_details(conv_output[j], layer_name, log_file, output_size);

        avg_pool_output[j] = avg_pooling(conv_output[j], pool_size, pool_stride, sensor_data_count, avg_pool_output_size, input_dimension);

        char avg_pool_layer_name[50];
        sprintf(avg_pool_layer_name, "Avarage Pooling Layer %d", j + 1);
        log_layer_details(avg_pool_output[j], avg_pool_layer_name, log_file, avg_pool_output_size);

        free_conv_layer(conv_layer);
    }
    int filter_size_2 = 10; // 64 olucak bu
    printf("Convolution and Average Pooling complete.\n");
    int pool_size_2 = 2;
    int pool_stride_2 = 2;
    int output_size_2 = (avg_pool_output_size - kernel_size) / conv_stride + 1;
    int avg_pool_output_size_2 = (output_size_2 - pool_size_2) / pool_stride_2 + 1;

    // filter size 64 convolution layer 2
    Data *conv_output2 = malloc(filter_size_2 * sizeof(Data));
    Data *avg_pool_output2 = malloc(filter_size_2 * sizeof(Data));
    printf("Starting Second Convolution and Average Pooling...\n");
    for (int j = 0; j < filter_size_2; j++)
    {

        ConvLayer *conv_layer2 = create_conv_layer(avg_pool_output_size, kernel_size, conv_stride, filter_size_2, filter_size);
        log_conv_layer_details(conv_layer2, log_file);

        conv_output2[j] = convolve(avg_pool_output, conv_layer2, avg_pool_output_size, output_size_2);

        char layer_name2[50];
        sprintf(layer_name2, "Output Layer %d", j + 1);
        log_layer_details(conv_output2[j], layer_name2, log_file, output_size_2);
        avg_pool_output2[j] = avg_pooling(conv_output2[j], pool_size_2, pool_stride_2, avg_pool_output_size, avg_pool_output_size_2, filter_size);
        char avg_pool_layer_name2[50];
        sprintf(avg_pool_layer_name2, "Avarage Pooling Layer %d", j + 1);
        log_layer_details(avg_pool_output2[j], avg_pool_layer_name2, log_file, avg_pool_output_size_2);

        free_conv_layer(conv_layer2);
    }
    // burda avgpooling outputunu kullanacağım
    printf("Second Convolution and Average Pooling complete.\n");
    // burda global avg pooling yapılacak
    Data *global_avg_pool_output = malloc(filter_size_2 * sizeof(Data));
    printf("Starting Global Average Pooling...\n");
    for (int i = 0; i < filter_size_2; i++)
    {
        global_avg_pool_output[i] = avg_pooling(avg_pool_output2[i], avg_pool_output_size_2, 1, avg_pool_output_size_2, 1, filter_size_2);
        char global_avg_pool_layer_name[50];
        sprintf(global_avg_pool_layer_name, "Global Avarage Pooling Layer %d", i + 1);
        log_layer_details(global_avg_pool_output[i], global_avg_pool_layer_name, log_file, 1);
    }
    printf("Global Average Pooling complete.\n");

    printf("Starting First Dense Layer...\n");

    Data *dense_output = malloc(DENSE_UNITS_1 * sizeof(Data));
    for (int i = 0; i < DENSE_UNITS_1; i++)
    {
        dense_output[i].values = malloc(sizeof(double)); // Bellek ayırma
        DenseLayer *dense_layer = create_dense_layer(filter_size_2, DENSE_UNITS_1);
        log_dense_layer_details(dense_layer, log_file);

        dense(global_avg_pool_output, dense_output[i], dense_layer, filter_size_2, DENSE_UNITS_1, "relu");

        char dense_layer_name[50];
        sprintf(dense_layer_name, "Dense Layer %d", i + 1);
        log_layer_details(dense_output[i], dense_layer_name, log_file, 1);

        free_dense_layer(dense_layer);
    }
    printf("First Dense Layer complete.\n");

    printf("Starting Second Dense Layer...\n");
    Data *dense_output2 = malloc(DENSE_UNITS_2 * sizeof(Data));
    for (int i = 0; i < DENSE_UNITS_2; i++)
    {
        dense_output2[i].values = malloc(sizeof(double)); // Bellek ayırma
        DenseLayer *dense_layer2 = create_dense_layer(DENSE_UNITS_1, DENSE_UNITS_2);
        log_dense_layer_details(dense_layer2, log_file);

        dense(dense_output, dense_output2[i], dense_layer2, DENSE_UNITS_1, DENSE_UNITS_2, "softmax");

        char dense_layer_name2[50];
        sprintf(dense_layer_name2, "Dense Layer %d", i + 1);
        log_layer_details(dense_output2[i], dense_layer_name2, log_file, 1);

        free_dense_layer(dense_layer2);
    }

    // softmax burada uygulanacak
    Data *softmax_output = malloc(DENSE_UNITS_2 * sizeof(Data));
    for (int i = 0; i < DENSE_UNITS_2; i++)
    {
        softmax_output[i].values = malloc(sizeof(double)); // Bellek ayırma
        softmax(dense_output2, softmax_output[i], DENSE_UNITS_2, i);
        char softmax_layer_name[50];
        sprintf(softmax_layer_name, "Softmax Layer %d", i + 1);
        log_layer_details(softmax_output[i], softmax_layer_name, log_file, 1);
    }

    // -----------------------------------------------
    // Cleanup
    printf("Cleaning up...\n");
    // freeing sensor data
    free_sensor_data(sensor_data, sensor_data_count);
    // freeing data
    free(data);
    // freeing first convolution layer
    for (int i = 0; i < filter_size; i++)
    {
        free(conv_output[i].values);
        free(avg_pool_output[i].values);
    }
    free(conv_output);
    free(avg_pool_output);
    // freeing seconde convolution layer
    for (int i = 0; i < filter_size_2; i++)
    {
        free(conv_output2[i].values);
        free(avg_pool_output2[i].values);
    }

    free(conv_output2);
    free(avg_pool_output2);

    // freeing global avg pooling
    for (int i = 0; i < filter_size_2; i++)
    {
        free(global_avg_pool_output[i].values);
    }
    free(global_avg_pool_output);

    // freeing first dense layer
    for (int i = 0; i < DENSE_UNITS_1; i++)
    {
        free(dense_output[i].values);
    }
    free(dense_output);

    // freeing second dense layer
    for (int i = 0; i < DENSE_UNITS_2; i++)
    {
        free(dense_output2[i].values);
    }
    free(dense_output2);

    // freeing softmax layer
    for (int i = 0; i < DENSE_UNITS_2; i++)
    {
        free(softmax_output[i].values);
    }
    free(softmax_output);

    // closing log file
    fclose(log_file);

    printf("Logging complete. Check 'cnn_1_log.txt' for details.\n");
    return 0;
}