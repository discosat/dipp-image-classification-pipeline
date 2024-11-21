#include "module.h"
#include "util.h"
#include <stdio.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/shm.h>
#include <string.h>
#include <time.h>
#include "metadata.pb-c.h"

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        printf("Missing arguments: Expected <num_images> <pipeline_id> <image_name>");
        return -1;
    }

    char * image_name = argv[3];

    // Get timestamp (used for SHM key)
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time) < 0)
    {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }

    printf("Got time\r\n");

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
    new_meta.timestamp = 0; // example time (should be using unix timestamp)
    new_meta.bits_pixel = bits_per_pixel;
    new_meta.camera = "rgb";

    printf("Created meta\r\n");

    size_t meta_size = metadata__get_packed_size(&new_meta);
    uint8_t meta_buf[meta_size];
    metadata__pack(&new_meta, meta_buf);

    printf("Packed meta\r\n");

    uint32_t batch_size = (image_size + sizeof(uint32_t) + meta_size) * data.num_images;

    int shmid = shmget(time.tv_nsec, batch_size, IPC_CREAT | 0666);
    data.shmid = shmid;
    char *shmaddr = shmat(shmid, NULL, 0);
    data.batch_size = batch_size;
    int offset = 0;

    printf("Attached to the shared memory\r\n");

    for (size_t i = 0; i < data.num_images; i++)
    {
        // Insert metadata size before metadata
        memcpy(shmaddr + offset, &meta_size, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        printf("Copied size of meta\r\n");
        memcpy(shmaddr + offset, &meta_buf, meta_size);
        offset += meta_size;
        printf("Copied meta\r\n");
        // insert image
        fseek(fh, 0, SEEK_SET);
        fread(shmaddr + offset, 1, image_size, fh);
        offset += image_size;
        printf("Copied image\r\n");
    }

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

    // detach from the shared memory segment
    shmdt(shmaddr);
}