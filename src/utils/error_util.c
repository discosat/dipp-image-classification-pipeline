#include "util.h"
#include "globals.h"
#include <unistd.h>
#include <sys/mman.h>

int *error_pipe;

void signal_error_and_exit(uint16_t error_code)
{
    if (write(error_pipe[1], &error_code, sizeof(uint16_t)) < 0)
    {
        perror("Error writing to error pipe");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Error with code %d occurred.\n", error_code);

    // Detach and free shared memory
    image_batch_cleanup(input);
    exit(EXIT_FAILURE);
}