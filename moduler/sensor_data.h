
#ifndef SENSOR_DATA_H
#define SENSOR_DATA_H
#include <stdlib.h>
#include <string.h>
#include <stdio.h> 

typedef struct
{
    double *values;
 
} Data;
typedef struct
{
    int user_id;
    char activity[20];
    long long timestamp;
    Data data;
} SensorData;

int read_sensor_data(const char *filename, SensorData *sensor_data, int max_lines);
void log_sensor_data(SensorData *sensor_data, int count, FILE *file);
void free_sensor_data(SensorData *sensor_data, int count);

#endif