
#include "sensor_data.h"

int read_sensor_data(const char *filename, SensorData *sensor_data, int max_lines)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("Error opening file!\n");
        return -1;
    }

    char line[200];
    int count = 0;

    while (fgets(line, sizeof(line), file) && count < max_lines)
    {
        // Bellek tahsisi yap
        sensor_data[count].data.values = malloc(3 * sizeof(double));
        if (sensor_data[count].data.values == NULL)
        {
            printf("Memory allocation failed!\n");
            fclose(file);
            return -1;
        }
        if (sscanf(line, "%d,%[^,],%lld,%lf,%lf,%lf",
                   &sensor_data[count].user_id,
                   sensor_data[count].activity,
                   &sensor_data[count].timestamp,
                   &sensor_data[count].data.values[0],       // x
                   &sensor_data[count].data.values[1],       // y
                   &sensor_data[count].data.values[2]) == 6) // z
        {
            count++;
        }
    }

    fclose(file);
    return count;
}

void log_sensor_data(SensorData *sensor_data, int count, FILE *file)
{
    fprintf(file, "\nSensor Data:\n");
    fprintf(file, "Total data count: %d\n", count);
    for (int i = 0; i < count; i++)
    {
        fprintf(file, "%d - User: %d, Activity: %s, Timestamp: %lld, X: %f, Y: %f, Z: %f\n",
                i + 1,
                sensor_data[i].user_id,
                sensor_data[i].activity,
                sensor_data[i].timestamp,
                sensor_data[i].data.values[0],
                sensor_data[i].data.values[1],
                sensor_data[i].data.values[2]);
    }
    fprintf(file, "\n");
}

void free_sensor_data(SensorData *sensor_data, int count)
{
    for (int i = 0; i < count; i++)
    {
        free(sensor_data[i].data.values);
    }
    // free(sensor_data); //bunu malloc yapmadığımdan acaba freelicek miyim

}

