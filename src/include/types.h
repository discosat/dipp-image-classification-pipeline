#ifndef TYPES_H
#define TYPES_H
#include <stddef.h>
#include "metadata.pb-c.h"

typedef enum
{
    STORAGE_MMAP,
    STORAGE_MEM,
    STORAGE_NOT_SET
} StorageMode;

typedef struct ImageBatch
{
    long mtype;               /* message type to read from the message queue */
    int num_images;           /* amount of images */
    int batch_size;           /* size of the image batch */
    int pipeline_id;          /* id of pipeline to utilize for processing */
    int priority;             /* priority of the image batch, e.g. max_latency from SLOs */
    unsigned char *data;      /* address to image data (in shared memory) */
    char filename[111];       /* filename of the image data */
    int shmid;                /* shared memory id for the image data */
    char uuid[37];            /* uuid of the image data */
    int progress;             /* index of the last processed module (-1 if not started) */
    StorageMode storage_mode; /* storage mode for the image data */
} ImageBatch;

typedef enum ModuleParameter__ValueCase
{
    NOT_SET = 0,
    BOOL_VALUE = 2,
    INT_VALUE = 3,
    FLOAT_VALUE = 4,
    STRING_VALUE = 5
} ModuleParameter__ValueCase;

typedef struct ModuleParameter
{
    char *key;
    ModuleParameter__ValueCase value_case;
    union
    {
        int bool_value;
        int int_value;
        float float_value;
        char *string_value;
    };
} ModuleParameter;

typedef struct ModuleParameterList
{
    size_t n_parameters;
    ModuleParameter **parameters;
} ModuleParameterList;

typedef struct MetadataList
{
    size_t n_metadata;
    Metadata **metadata;
} MetadataList;

// Return codes
#define SUCCESS 0
#define FAILURE -1

#endif // TYPES_H