#include "logger.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define TIMESTAMP_SIZE 35
#define MAX_LOG_LENGTH 1024

struct Logger {
    FILE* log_file;
};

static const char* level_to_string(LogLevel level) {
    switch (level) {
        case LOG_DEBUG:    return "DEBUG";
        case LOG_INFO:     return "INFO";
        case LOG_WARNING:  return "WARNING";
        case LOG_ERROR:    return "ERROR";
        case LOG_CRITICAL: return "CRITICAL";
        default:           return "UNKNOWN";
    }
}

Logger* logger_create(const char* filename) {
    Logger* logger = (Logger*)malloc(sizeof(Logger));
    if (!logger) {
        return NULL;
    }

    logger->log_file = fopen(filename, "a");
    if (!logger->log_file) {
        fprintf(stderr, "Error opening log file: %s\n", filename);
        free(logger);
        return NULL;
    }

    return logger;
}

void logger_destroy(Logger* logger) {
    if (logger) {
        if (logger->log_file) {
            fclose(logger->log_file);
        }
        free(logger);
    }
}

#define BUFFER_SIZE 10000

typedef struct {
    char log_entries[BUFFER_SIZE][MAX_LOG_LENGTH];
    int count;
} LogBuffer;

static LogBuffer log_buffer = { .count = 0 };

void logger_log(Logger* logger, LogLevel level, const char* message) {
    if (!logger || !logger->log_file) {
        return;
    }

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    
    // Convert to tm structure
    struct tm* timeinfo = localtime(&ts.tv_sec);
    char timestamp[TIMESTAMP_SIZE];
    size_t base_len = strftime(timestamp, sizeof(timestamp), 
                              "%Y-%m-%d %H:%M:%S", timeinfo);
    
    // Add microseconds to timestamp
    snprintf(timestamp + base_len, sizeof(timestamp) - base_len, 
             ".%06ld", ts.tv_nsec / 1000);

    snprintf(log_buffer.log_entries[log_buffer.count], MAX_LOG_LENGTH, 
             "[%s] %s: %s\n", timestamp, level_to_string(level), message);

    log_buffer.count++;

    // Output to console
    //printf("%s", log_buffer.log_entries[log_buffer.count - 1]);
}

void logger_log_print(Logger* logger, LogLevel level, const char* message) {
    if (!logger || !logger->log_file) {
        return;
    }

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    
    // Convert to tm structure
    struct tm* timeinfo = localtime(&ts.tv_sec);
    char timestamp[TIMESTAMP_SIZE];
    size_t base_len = strftime(timestamp, sizeof(timestamp), 
                              "%Y-%m-%d %H:%M:%S", timeinfo);
    
    // Add microseconds to timestamp
    snprintf(timestamp + base_len, sizeof(timestamp) - base_len, 
             ".%06ld", ts.tv_nsec / 1000);

    snprintf(log_buffer.log_entries[log_buffer.count], MAX_LOG_LENGTH, 
             "[%s] %s: %s\n", timestamp, level_to_string(level), message);

    log_buffer.count++;

    // Output to console
    printf("%s", log_buffer.log_entries[log_buffer.count - 1]);
}



void logger_flush(Logger* logger) {
    if (!logger || !logger->log_file) {
        return;
    }

    for (int i = 0; i < log_buffer.count; i++) {
        fprintf(logger->log_file, "%s", log_buffer.log_entries[i]);
    }
    fflush(logger->log_file);
    log_buffer.count = 0;
}