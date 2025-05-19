#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>
#include "util.h"
#include <uuid/uuid.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

void finalize()
{

    // create a new memory mapped file
    char file_uuid[37];
    uuid_t uuid;
    uuid_generate_random(uuid);
    uuid_unparse_lower(uuid, file_uuid);

    char filename_prefix[] = "/usr/share/dipp/data/batch_%s.bin";
    char batch_filename[sizeof(filename_prefix) + 37];
    snprintf(batch_filename, sizeof(filename_prefix) + 37, filename_prefix, file_uuid);

    int fd = open(batch_filename, O_RDWR | O_CREAT, 0644);
    if (fd < 0)
    {
        printf("Error opening batch file: %s\n", batch_filename);
        return;
    }

    // Ensure file is large enough
    if (ftruncate(fd, result->batch_size) == -1)
    {
        printf("Error resizing batch file: %s\n", batch_filename);
        close(fd);
        return;
    }

    // Memory map the file
    char *persisted_batch = mmap(NULL, result->batch_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (persisted_batch == MAP_FAILED)
    {
        printf("Error mapping batch file: %s\n", batch_filename);
        close(fd);
        return;
    }

    memcpy(persisted_batch, result->data, result->batch_size);

    close(fd);
    munmap(persisted_batch, result->batch_size);

    // change the filename of the result batch
    strcpy(result->filename, batch_filename);

    result->progress = input->progress + 1;

    // delete the old batch file
    if (remove(input->filename) != 0)
    {
        printf("Error deleting old batch file: %s\n", input->filename);
    }
}
