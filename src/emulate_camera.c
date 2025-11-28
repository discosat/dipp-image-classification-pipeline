#include "module.h"
#include "util.h"
#include <stdio.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/shm.h>
#include <string.h>
#include <time.h>
#include "metadata.pb-c.h"
#include <uuid/uuid.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        printf("Missing arguments: Expected <num_images> <pipeline_id> <observation_id> <image_name>");
        return -1;
    }

    char *image_name = argv[4];
    FILE *fh = fopen(image_name, "r");
    // get size of the file in bytes
    fseek(fh, 0, SEEK_END);
    long fsize = ftell(fh);
    fseek(fh, 0, SEEK_SET);

    printf("Opened file\r\n");

    // Prepare the data
    ImageBatch data;
    data.mtype = 1;
    data.num_images = atoi(argv[1]);
    data.pipeline_id = atoi(argv[2]);

    // generate a random latency between 60 and 120 seconds
    srand(time(0));
    // int latency = rand() % 60 + 60;
    // 1 minute to process query
    int latency = 60;
    // get current time in seconds
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time) < 0)
    {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    // add latency to current time
    data.priority = time.tv_sec + latency; // max_timestamp (in seconds)
    data.progress = -1;                    // default progress
    data.filename[0] = '\0';               // no filename needed
    data.data = NULL;                      // no data needed
    data.storage_mode = STORAGE_MEM;       // use shared memory

    char batch_uuid[37];
    uuid_t uuid;
    uuid_generate_random(uuid);
    uuid_unparse_lower(uuid, batch_uuid);

    strcpy(data.uuid, batch_uuid);

    // Hardcoded bayer image specs
    uint32_t image_height = 2056;
    uint32_t image_width = 2464;
    uint32_t bits_per_pixel = 12;
    uint32_t image_channels = 1;

    uint32_t image_size = fsize;
    Metadata new_meta = METADATA__INIT;
    new_meta.size = image_size;
    new_meta.width = image_width;
    new_meta.height = image_height;
    new_meta.channels = image_channels;
    new_meta.timestamp = time.tv_sec; // use current unix timestamp
    new_meta.bits_pixel = bits_per_pixel;
    new_meta.camera = "1800 U-507c";
    new_meta.obid = atoi(argv[3]);

    printf("Created meta\r\n");

    size_t meta_size = metadata__get_packed_size(&new_meta);
    uint8_t *meta_buf = malloc(meta_size);
    if (!meta_buf)
    {
        perror("malloc");
        return -1;
    }
    size_t packed_len = metadata__pack(&new_meta, meta_buf);
    if (packed_len != meta_size)
    {
        /* keep the authoritative packed length */
        printf("Warning: packed metadata length (%zu) differs from expected size (%zu)\n",
               packed_len, meta_size);
        meta_size = packed_len;
    }

    // Debug dump
    fprintf(stderr, "meta_size=%zu\n", meta_size);
    for (size_t i = 0; i < meta_size; ++i)
    {
        fprintf(stderr, "%02x ", meta_buf[i]);
    }
    fprintf(stderr, "\n");

    printf("Packed meta\r\n");

    uint32_t meta_size32 = (uint32_t)meta_size;
    uint32_t batch_size = (image_size + sizeof(uint32_t) + meta_size32) * data.num_images;

    int shmid = shmget(time.tv_nsec, batch_size, IPC_CREAT | 0666);
    data.shmid = shmid;
    char *shmaddr = shmat(shmid, NULL, 0);
    data.batch_size = batch_size;
    size_t offset = 0;

    for (size_t i = 0; i < data.num_images; i++)
    {
        // Insert metadata size before metadata
        memcpy(shmaddr + offset, &meta_size32, sizeof(meta_size32));
        offset += sizeof(uint32_t);
        printf("Copied size of meta\r\n");
        memcpy(shmaddr + offset, meta_buf, meta_size);
        offset += meta_size;
        printf("Copied meta\r\n");
        // insert image
        fseek(fh, 0, SEEK_SET);
        if (fread(shmaddr + offset, 1, image_size, fh) < 0)
        {
            perror("Error reading image");
            shmdt(shmaddr);
            free(meta_buf);
            return -1;
        }
        offset += image_size;
        printf("Copied image\r\n");
    }
    free(meta_buf);

    // create msg queue
    int msg_queue_id;
    if ((msg_queue_id = msgget(71, 0666 | IPC_CREAT)) == -1)
    {
        perror("msgget error");
    }

    printf("Got queue\r\n");

    // send msg to queue
    if (msgsnd(msg_queue_id, &data, sizeof(data), 0) == -1)
    {
        perror("msgsnd error");
    }

    printf("Image sent!\n");

    shmdt(shmaddr);
    fclose(fh);
}