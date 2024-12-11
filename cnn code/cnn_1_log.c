#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_LINES 1024
#define MAX_LINE_LENGTH 200

typedef struct
{
    int user_id;
    char activity[20];
    long long timestamp;
    double *x, *y, *z;
} SensorData;

typedef struct
{
    double *x_weights, *y_weights, *z_weights; // filtrenin kanallara göre ağırlıkları
    int kernel_size;                           // filtrenin boyutu
    int input_size;                            // girişin boyutu
    int stride;                                // filtrenin kaydırılma miktarı
    double x_bias, y_bias, z_bias;             // filtrenin biası
} ConvLayer;

typedef enum
{
    CONV1D,          // 0
    AVGPOOL1D,       // 1
    GLOBALAVGPOOL1D, // 2
    DENSE,           // 3
    SOFTMAX          // 4
} LayerType;

typedef struct
{
    double *data; // katman verileri
    int size;     // katman boyutu
} Layer;

// Confusion Matrix için yapı
typedef struct
{
    int true_positives;
    int false_positives;
    int true_negatives;
    int false_negatives;
} ConfusionMatrix;

// aktivation functions
double relu(double x)
{ // relu sıfırın altındaki değerleri sıfıra eşitler
    return x > 0 ? x : 0;
}

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

ConvLayer *create_conv_layer(int input_size, int kernel_size, int stride)
{
    ConvLayer *layer = malloc(sizeof(ConvLayer));
    layer->x_weights = malloc(kernel_size * sizeof(double));
    layer->y_weights = malloc(kernel_size * sizeof(double));
    layer->z_weights = malloc(kernel_size * sizeof(double));

    // ağırlıkları random başlatıtm
    for (int i = 0; i < kernel_size; i++)
    {
        layer->x_weights[i] = (double)rand() / RAND_MAX - 0.5;
        layer->y_weights[i] = (double)rand() / RAND_MAX - 0.5;
        layer->z_weights[i] = (double)rand() / RAND_MAX - 0.5;
    }

    layer->kernel_size = kernel_size;
    layer->input_size = input_size;
    layer->stride = stride;

    layer->x_bias = (double)rand() / RAND_MAX - 0.5;
    layer->y_bias = (double)rand() / RAND_MAX - 0.5;
    layer->z_bias = (double)rand() / RAND_MAX - 0.5;

    return layer;
}

Layer *convolve(SensorData *input, ConvLayer *conv_layer, int data_count)
{
    int output_size = (data_count - conv_layer->kernel_size) / conv_layer->stride + 1; // stride olayını düşünmek lazım
    Layer *output = malloc(sizeof(Layer));
    output->data = malloc(output_size * sizeof(double));
    output->size = output_size;
    int i, j;
    for (i = 0; i < output_size; i++)
    {
        double x_sum = 0, y_sum = 0, z_sum = 0;
        for (j = 0; j < conv_layer->kernel_size; j++)
        {
            // İndeks sınırlarını kontrol et
            if (i * conv_layer->stride + j >= data_count)
            {
                printf("Index out of bounds: i = %d, j = %d\n", i, j);
                return NULL;
            }
            // Bellek erişimini kontrol et
            if (conv_layer->x_weights == NULL || input->x == NULL)
            {
                printf("Null pointer encountered.\n");
                return NULL;
            }
            // çarpımlarını hesaplıyoruz
            x_sum += input->x[i * conv_layer->stride + j] * conv_layer->x_weights[j];
            y_sum += input->y[i * conv_layer->stride + j] * conv_layer->y_weights[j];
            z_sum += input->z[i * conv_layer->stride + j] * conv_layer->z_weights[j];
        }
        // biasları ekledik
        x_sum += conv_layer->x_bias;
        y_sum += conv_layer->y_bias;
        z_sum += conv_layer->z_bias;

        // aktivasyonun uygulanması
        output->data[i] = relu(x_sum + y_sum + z_sum);
    }

    return output;
}

void update_weights(ConvLayer *layer, double *gradients, double learning_rate)
{
    for (int i = 0; i < layer->kernel_size; i++)
    {
        // layer->weights[i] -= learning_rate * gradients[i];
    }
    // layer->bias -= learning_rate * gradients[layer->kernel_size];
}

void log_layer_details(Layer *layer, const char *layer_name, FILE *file)
{
    fprintf(file, "\n%s Details:\n", layer_name);
    fprintf(file, "Size: %d\n", layer->size);
    fprintf(file, "Data: \n");
    for (int i = 0; i < layer->size; i++)
    {
        fprintf(file, "[%d]: %f\n", i, layer->data[i]);
    }
    fprintf(file, "\n");
}
void log_sensor_data(SensorData *data, int count, FILE *file)
{
    fprintf(file, "\nSensor Data:\n");
    fprintf(file, "Total data count: %d\n", count);
    for (int i = 0; i < count; i++)
    {
        fprintf(file, "%d - User: %d, Activity: %s, Timestamp: %lld, X: %f, Y: %f, Z: %f\n",
                i + 1,
                data[i].user_id,
                data[i].activity,
                data[i].timestamp,
                data[i].x,
                data[i].y,
                data[i].z);
    }

    fprintf(file, "\n");
}

void log_conv_layer_details(ConvLayer *layer, FILE *file)
{
    fprintf(file, "\n======================== Convolution Layer Details ========================\n");
    fprintf(file, "Input Size : %d\n", layer->input_size);
    fprintf(file, "Kernel Size: %d\n", layer->kernel_size);
    fprintf(file, "Stride     : %d\n", layer->stride);
    fprintf(file, "X Bias     : %.6f\n", layer->x_bias);
    fprintf(file, "Y Bias     : %.6f\n", layer->y_bias);
    fprintf(file, "Z Bias     : %.6f\n", layer->z_bias);

    fprintf(file, "\n----------------------------- Weights -----------------------------------\n");
    fprintf(file, "Index |   X Weight     |   Y Weight     |   Z Weight     \n");
    fprintf(file, "-------------------------------------------------------------\n");
    for (int i = 0; i < layer->kernel_size; i++)
    {
        fprintf(file, "%5d | %13.6f | %13.6f | %13.6f\n",
                i, layer->x_weights[i], layer->y_weights[i], layer->z_weights[i]);
    }
    fprintf(file, "==========================================================================\n\n");
}

int read_sensor_data(const char *filename, SensorData *data)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("Error opening file!\n");
        return -1;
    }

    char line[MAX_LINE_LENGTH];
    int count = 0;

    while (fgets(line, sizeof(line), file) && count < MAX_LINES)
    {
        // Parse the line
        if (sscanf(line, "%d,%[^,],%lld,%lf,%lf,%lf",
                   &data[count].user_id,
                   data[count].activity,
                   &data[count].timestamp,
                   &data[count].x,
                   &data[count].y,
                   &data[count].z) == 6)
        {
            count++;
        }
    }

    fclose(file);
    return count;
}

int main()
{
    // Open a log file
    FILE *log_file = fopen("cnn_1_log.txt", "w");
    if (log_file == NULL)
    {
        printf("Error opening log file!\n");
        return 1;
    }

    // Set random seed for reproducibility
    srand(42);

    // // Create input layer
    // Layer input;
    // input.size = 99;
    // input.data = malloc(input.size * sizeof(SensorData));

    SensorData sensor_data[MAX_LINES];
    // Read sensor data from file
    int data_count = read_sensor_data("train_data.txt", sensor_data);

    // // Log input layer details
    // log_layer_details(&input, "Input Layer", log_file);

    // log sensor data
    log_sensor_data(sensor_data, data_count, log_file);

    // Create convolution layer
    int kernel_size = 3; // kernel size ım 3 olacak
    int filter_size = 1;
    int stride = 1;
    ConvLayer *conv_layer;
    Layer **outputs = malloc(filter_size * sizeof(Layer *));
    for (int i = 0; i < filter_size; i++)
    {
        conv_layer = create_conv_layer(data_count, kernel_size, stride); // her channel için ayrı ayrı con layeri oluşturuldu.
        //  Log convolution layer details
        log_conv_layer_details(conv_layer, log_file);

        // sensor data zaten bir array olduğundan ayrıca & işareti ile göndermeye gerek duymazmış

        // Convolve işlemi
        outputs[i] = convolve(sensor_data, conv_layer, data_count);

        // Çıktıyı logla
        char layer_name[50];
        sprintf(layer_name, "Output Layer %d", i + 1);
        log_layer_details(outputs[i], layer_name, log_file);

        // Her filtrenin belleğini temizlemeyi unutmayın
        free(conv_layer->x_weights);
        free(conv_layer->y_weights);
        free(conv_layer->z_weights);
        free(conv_layer);
    }

    // Log output layer details
    // log_layer_details(output, "Output Layer", log_file);

    // Close the log file
    fclose(log_file);

    // Free allocated memory
    // free(input.data);
    // free(output->data);
    // free(output);
    // free(conv_layer->weights);
    // free(conv_layer);

    for (int i = 0; i < filter_size; i++)
    {
        free(outputs[i]->data);
        free(outputs[i]);
    }
    free(outputs);

    printf("Logging complete. Check '1d_cnn_log.txt' for details.\n");

    return 0;
}