#include "util.h"
#include "globals.h"
#include <unistd.h>
#include <sys/mman.h>

int *error_pipe;

void signal_error_and_exit(uint16_t error_code)
{
    write(error_pipe[1], &error_code, sizeof(uint16_t));
    fprintf(stderr, "Error with code %d occurred.\n", error_code);

    // Detach and free shared memory
    munmap(input->data, input->batch_size);
    exit(EXIT_FAILURE);
}