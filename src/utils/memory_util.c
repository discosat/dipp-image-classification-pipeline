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
    switch (input->storage_mode)
    {
    case STORAGE_MMAP:
        // create a new memory mapped file
        char file_uuid[37];
        uuid_t uuid;
        uuid_generate_random(uuid);
        uuid_unparse_lower(uuid, file_uuid);

        char filename_prefix[] = "/usr/share/dipp/data/batch_%s_%s.bin";
        char batch_filename[sizeof(filename_prefix) + 37 + 37];
        snprintf(batch_filename, sizeof(filename_prefix) + 37 + 37, filename_prefix, input->uuid, file_uuid);

        int fd = open(batch_filename, O_RDWR | O_CREAT, 0644);
        if (fd < 0)
        {
            signal_error_and_exit(303);
            return;
        }

        // Ensure file is large enough
        if (ftruncate(fd, result->batch_size) == -1)
        {
            close(fd);
            signal_error_and_exit(304);
            return;
        }

        // Memory map the file
        char *persisted_batch = mmap(NULL, result->batch_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (persisted_batch == MAP_FAILED)
        {
            close(fd);
            signal_error_and_exit(302);
            return;
        }

        memcpy(persisted_batch, result->data, result->batch_size);

        close(fd);

        if (munmap(persisted_batch, result->batch_size) == -1)
        {
            signal_error_and_exit(301);
            return;
        }

        // change the filename of the result batch
        strcpy(result->filename, batch_filename);

        // delete the old batch file
        if (remove(input->filename) != 0)
        {
            signal_error_and_exit(305);
        }

        break;

    case STORAGE_MEM:
        // Handle shared memory finalization
        struct shmid_ds info;
        if (shmctl(input->shmid, IPC_STAT, &info) == -1)
        {
            signal_error_and_exit(310);
        }

        size_t shm_size = info.shm_segsz;

        if (result->batch_size > shm_size)
        {
            // Resize is needed: Utilize new unique shared memory ID for storing the batch
            int new_shmid = -1;
            struct timespec time;
            // Continously try keys for new shared memory segments
            while (new_shmid == -1)
            {
                if (clock_gettime(CLOCK_MONOTONIC, &time) < 0)
                    signal_error_and_exit(517);

                if ((new_shmid = shmget(time.tv_nsec, result->batch_size, IPC_CREAT | IPC_EXCL | 0666)) != -1)
                    break;

                if (errno == EEXIST)
                    continue;

                signal_error_and_exit(306);
            }

            void *shmaddr = shmat(new_shmid, NULL, 0);
            if (shmaddr == NULL)
            {
                signal_error_and_exit(309);
            }

            memcpy(shmaddr, result->data, result->batch_size);
            free(result->data);

            // Detach and free old shared memory segment
            if (shmdt(input->data) == -1)
            {
                signal_error_and_exit(307);
            }
            if (shmctl(input->shmid, IPC_RMID, NULL) == -1)
            {
                signal_error_and_exit(308);
            }

            result->shmid = new_shmid;

            // detach the new shared memory segment
            if (shmdt(shmaddr) == -1)
            {
                signal_error_and_exit(307);
            }
        }
        else
        {
            // No resize is needed: We can utilize the old shared memory space
            memcpy(input->data, result->data, result->batch_size); // copy new data to shared memory space of old data
            free(result->data);
            result->shmid = input->shmid; // copy the shared memory key, as we are reusing the space
            if (shmdt(input->data) == -1)
            {
                signal_error_and_exit(307);
            }
        }
        break;
    default:
        break;
    }

    result->progress = input->progress + 1;
}
