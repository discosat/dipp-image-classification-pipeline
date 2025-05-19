#include "module.h"
#include "util.h"
#include <stdio.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <string.h>
#include <time.h>
#include "metadata.pb-c.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <uuid/uuid.h>

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        printf("Missing arguments: Expected <num_images> <pipeline_id> <max_timestamp> <image_name>");
        return -1;
    }

    char *image_name = argv[4];

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
    data.priority = atoi(argv[3]); // max_timestamp (in seconds)
    data.progress = -1;            // default progress

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

    char *file_uuid = malloc(37);
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
        return -1;
    }

    // Ensure file is large enough
    if (ftruncate(fd, batch_size) == -1)
    {
        printf("Error resizing batch file: %s\n", batch_filename);
        close(fd);
        return -1;
    }

    // Memory map the file
    char *persisted_batch = mmap(NULL, batch_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (persisted_batch == MAP_FAILED)
    {
        printf("Error mapping batch file: %s\n", batch_filename);
        close(fd);
        return -1;
    }

    size_t offset = 0;
    data.batch_size = batch_size;
    strcpy(data.filename, batch_filename);

    for (size_t i = 0; i < data.num_images; i++)
    {
        // Insert metadata size before metadata
        memcpy(persisted_batch + offset, &meta_size, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        printf("Copied size of meta\r\n");
        memcpy(persisted_batch + offset, &meta_buf, meta_size);
        offset += meta_size;
        printf("Copied meta\r\n");
        // insert image
        fseek(fh, 0, SEEK_SET);
        fread(persisted_batch + offset, 1, image_size, fh);
        offset += image_size;
        printf("Copied image\r\n");
    }

    close(fd);
    munmap(persisted_batch, batch_size);

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
}