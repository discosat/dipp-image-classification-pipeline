#ifndef LOGGER_H
#define LOGGER_H
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARNING,
    LOG_ERROR,
    LOG_CRITICAL
} LogLevel;

typedef struct Logger Logger;

// Create a new logger instance
Logger* logger_create(char const* filename);

// Destroy logger instance
void logger_destroy(Logger* logger);

// Log a message
void logger_log(Logger* logger, LogLevel level, const char* message);

void logger_log_print(Logger* logger, LogLevel level, const char* message);

void logger_flush(Logger* logger);

#ifdef __cplusplus
}
#endif
#endif // LOGGER_H