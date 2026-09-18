#include "module.h"
#include "util.h"
#include "logger.h"
#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef enum FAIL_CASE
{
    ARITHMETIC,
    DEREFERENCE_NULL_POINTER,
    ACCESS_PROTECTED_MEMORY,
    WRITING_READ_ONLY_MEMORY,
    STACK_OVERFLOW,
    MEMCPY_TO_INVALID_DESTINATION,
    INDEX_OUT_OF_BOUNDS,
    ACCESS_NULL_FUNCTION_POINTER,
    EXCEEDING_MEMORY_ALLOCATION,
    DOUBLE_FREE,
    BUFFER_OVERFLOW,
    SHM_INCREASE_OVERFLOW,
    HANG,                       /* never returns: exercises the daemon's module timeout */
} FAIL_CASE;

typedef void (*FuncPtr)();

/* Every case must reach a faulting instruction at ANY optimisation level.
 * Built at -O3 (the old flags), the compiler treated the divide-by-zero, the
 * null stores, the out-of-bounds index, the double free and the buffer
 * overflow as undefined behaviour and deleted them, and turned the
 * recursion into an infinite loop: several "faults" were no-ops. The target
 * is now built at -O0 (meson.build) AND the operands are volatile, so the
 * stores and calls are emitted whatever the flags. Two cases cannot fault
 * by themselves on this platform and say so:
 *   - AArch64 integer division by zero does not trap (sdiv yields 0), so the
 *     ARITHMETIC case performs the division and then raises SIGFPE itself:
 *     the fault class is "the module dies of an arithmetic signal", which is
 *     what the daemon has to survive;
 *   - malloc of 1 TB returns NULL under Linux overcommit rather than dying,
 *     so EXCEEDING_MEMORY_ALLOCATION reports it through the error pipe -- the
 *     graceful path (MODULE_EXIT_CUSTOM), deliberately distinct from a crash. */
static volatile int sink;

static int recursion(volatile int depth)
{
    volatile char pad[4096];        /* a real frame per call: 4 KB */
    pad[depth & 4095] = (char)depth;
    return recursion(depth + 1) + pad[0];
}
/* START MODULE IMPLEMENTATION */
void module()
{
    const char *dir = "/home/root/logs/";
    char file_name[256];
    time_t t = time(NULL);
    snprintf(file_name, sizeof(file_name), "crash_%ld.txt", t);

    char full_path[512];
    snprintf(full_path, sizeof(full_path), "%s%s", dir, file_name);

    Logger *logger = logger_create(full_path);
    logger_log_print(logger, LOG_INFO, "Crash module started");

    int fail_case = get_param_int("fail_case");

    switch (fail_case)
    {
    case ARITHMETIC:
    {
        printf("Dividing by zero...\n");
        volatile int x = 1, y = 0;
        sink = x / y;               /* no trap on AArch64: sdiv returns 0 */
        raise(SIGFPE);              /* deliver the arithmetic fault class regardless */
        break;
    }
    case DEREFERENCE_NULL_POINTER:
    {
        printf("Dereferencing null pointer...\n");
        int *volatile null_ptr = NULL;
        *null_ptr = 1;
        break;
    }
    case ACCESS_PROTECTED_MEMORY:
    {
        printf("Accessing protected memory...\n");
        char *volatile char_ptr = (char *)0x8;   /* first page, never mapped */
        *char_ptr = 'a';
        break;
    }
    case WRITING_READ_ONLY_MEMORY:
    {
        printf("Writing to read-only memory...\n");
        char *volatile str = (char *)"Hello";     /* .rodata */
        str[0] = 'h';
        break;
    }
    case STACK_OVERFLOW:
    {
        printf("Initiating infinite recursion...\n");
        sink = recursion(0);
        break;
    }
    case MEMCPY_TO_INVALID_DESTINATION:
    {
        printf("Copying to invalid destination...\n");
        char *volatile dest = NULL;
        const char *src = "Hello";
        memcpy(dest, src, strlen(src));
        break;
    }
    case INDEX_OUT_OF_BOUNDS:
    {
        printf("Indexing out of bounds...\n");
        volatile int arr[5];
        volatile long idx = 1L << 26;             /* 256 MB past the stack: unmapped */
        arr[idx] = 5;
        break;
    }
    case ACCESS_NULL_FUNCTION_POINTER:
    {
        printf("Calling null function pointer...\n");
        FuncPtr volatile funcPtr = NULL;
        funcPtr();
        break;
    }
    case EXCEEDING_MEMORY_ALLOCATION:
    {
        printf("Allocating 1TB of memory...\n");
        size_t tb = 1000ULL * 1000 * 1000 * 1000;
        void *p = malloc(tb);
        if (p == NULL)
        {
            /* Overcommit refuses this outright: report it, do not crash. */
            printf("malloc(1TB) returned NULL; reporting through the error pipe\n");
            signal_error_and_exit(20);
        }
        memset(p, 1, tb);           /* if it ever succeeds, touching it will not */
        break;
    }
    case DOUBLE_FREE:
    {
        printf("Freeing memory twice...\n");
        int *volatile ptr = (int *)malloc(sizeof(int));
        free(ptr);
        free(ptr);                  /* glibc: "double free detected", SIGABRT */
        break;
    }
    case BUFFER_OVERFLOW:
    {
        printf("Overflowing buffer...\n");
        volatile char buffer[10];
        volatile size_t n = 1u << 24;             /* 16 MB past a 10-byte stack buffer */
        memset((char *)buffer, 'A', n);
        break;
    }
    case SHM_INCREASE_OVERFLOW:
    {
        Metadata *input_meta = get_metadata(0);
        Metadata new_meta = METADATA__INIT;
        if (clone_metadata(input_meta, &new_meta) != 0)
        {
            signal_error_and_exit(10);
        }

        unsigned char *image_data;
        uint32_t image_size = get_image_data(0, &image_data);
        printf("Increased image batch to size: %d...\n", input->batch_size + image_size);
        for (int i = 0; i < get_input_num_images() + 1; i++)
        {
            append_result_image(image_data, image_size, &new_meta);
        }
        free(image_data);
        break;
    }
    case HANG:
    {
        printf("Hanging forever...\n");
        for (;;)
            pause();                /* SIGALRM from the daemon's timeout ends this */
        break;
    }
    default:
        printf("'fail_case' not recognized\n");
        break;
    }
    logger_log_print(logger, LOG_INFO, "Demosaic module finished");
    logger_flush(logger);
    logger_destroy(logger);
}
/* END MODULE IMPLEMENTATION */

/* Main function of module (NO NEED TO MODIFY) */
ImageBatch run(ImageBatch *input_batch, ModuleParameterList *module_parameter_list, int *ipc_error_pipe)
{
    ImageBatch result_batch;
    result = &result_batch;
    result->batch_size = 0;
    input = input_batch;
    config = module_parameter_list;
    error_pipe = ipc_error_pipe;
    initialize();

    module();

    finalize();

    return result_batch;
}
