#include "util.h"
#include "types.h"
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/shm.h>

ImageBatch *input;
ImageBatch *result;
ModuleParameterList *config;

int get_input_num_images()
{
    return input->num_images;
}

size_t get_image_data(int index, unsigned char **out)
{
    Metadata *image_meta = get_metadata(index);
    *out = (unsigned char *)malloc(image_meta->size);
    if (out == NULL)
    {
        signal_error_and_exit(100);
    }
    memcpy(*out, input->data + image_meta->image_offset, image_meta->size);
    return image_meta->size;
}

void append_result_image(unsigned char *data, uint32_t data_size, Metadata *meta)
{
    /* Pack new metadata */
    size_t meta_size = metadata__get_packed_size(meta);
    uint8_t meta_buf[meta_size];
    metadata__pack(meta, meta_buf);
    size_t block_size = data_size + meta_size + sizeof(uint32_t);
    if (result->batch_size == 0)
    {
        result->data = (unsigned char *)malloc(block_size);

        if (result->data == NULL)
        {
            signal_error_and_exit(100);
        }
    }
    else
    {
        unsigned char *tmp = (unsigned char *)realloc(result->data, result->batch_size + block_size);

        if (tmp == NULL)
        {
            signal_error_and_exit(101);
        }

        result->data = tmp;
    }

    if (result->data == NULL)
        return;

    /* Insert meta size, then the metadata, then the image data */
    unsigned char *ptr = result->data + result->batch_size;
    memcpy(ptr, &meta_size, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    memcpy(ptr, meta_buf, meta_size);
    ptr += meta_size;
    memcpy(ptr, data, data_size);

    result->batch_size += block_size;
    result->num_images += 1;
}

void initialize()
{
    result->batch_size = 0;
    result->num_images = 0;
    result->pipeline_id = input->pipeline_id;
    result->priority = input->priority;
    result->mtype = input->mtype;
    result->storage_mode = input->storage_mode;
    strcpy(result->uuid, input->uuid);
    image_batch_read_data(input);
    unpack_metadata();
}

int image_batch_read_data(ImageBatch *batch)
{
    if (!batch)
    {
        return FAILURE;
    }

    switch (batch->storage_mode)
    {
    case STORAGE_MMAP:
    {
        // Memory-mapped file access
        int fd = open(batch->filename, O_RDONLY, 0644);
        if (fd == -1)
        {
            signal_error_and_exit(303);
            return FAILURE;
        }

        batch->data = mmap(NULL, batch->batch_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);

        if (batch->data == MAP_FAILED)
        {
            signal_error_and_exit(302);
            return FAILURE;
        }

        break;
    }
    case STORAGE_MEM:
    {
        // Shared memory access
        batch->data = shmat(batch->shmid, NULL, 0);
        if (batch->data == (void *)-1)
        {
            signal_error_and_exit(308);
            return FAILURE;
        }

        break;
    }
    case STORAGE_NOT_SET:
    default:
        return FAILURE;
    }

    return SUCCESS;
}

int image_batch_cleanup(ImageBatch *batch)
{
    if (!batch)
    {
        return FAILURE;
    }

    int result = SUCCESS;

    switch (batch->storage_mode)
    {
    case STORAGE_MMAP:
    {
        // Unmap memory-mapped file
        if (batch->data && munmap(batch->data, batch->batch_size) == -1)
        {
            signal_error_and_exit(301);
            result = FAILURE;
        }
        break;
    }
    case STORAGE_MEM:
    {
        // Detach from shared memory
        if (batch->data && shmdt(batch->data) == -1)
        {
            signal_error_and_exit(306);
            result = FAILURE;
        }
        break;
    }
    case STORAGE_NOT_SET:
    default:
        break;
    }

    // Clear the data pointer
    batch->data = NULL;

    return result;
}