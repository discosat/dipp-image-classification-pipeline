#include "module.h"
#include "util.h"
#include "globals.h"

/* Define custom error codes */
enum ERROR_CODE
{
    MALLOC_ERR = 1,
    OPENCV_ERR = 2,
    OPENCV_DEM_ERR = 3,
    OPNECV_MAT_ERR = 4,
    OPENCV_ROT_ERR = 5,
    OPENCV_NORM_ERR = 6,
    INVALID_INPUT = 7,
    INVALID_INPUT_VALUES = 8,
    INVALID_NORMALIZATION_VALUE = 9,

};

/* START MODULE IMPLEMENTATION */
void module()
{
    /* Get number of images in input batch */
    int num_images = get_input_num_images();

    if (num_images <= 0)
    {
        signal_error_and_exit(INVALID_INPUT);
    }

    /* Process each image in the batch */
    for (int i = 0; i < num_images; ++i)
    {
        /* Get input image metadata */
        Metadata *input_meta = get_metadata(i);
        int height = input_meta->height;
        int width = input_meta->width;
        int channels = input_meta->channels;
        int timestamp = input_meta->timestamp;
        int bits_pixel = input_meta->bits_pixel;
        char *camera = input_meta->camera;
        int obid = input_meta->obid;

        if (height <= 0 || width <= 0 || channels <= 0)
        {
            signal_error_and_exit(INVALID_INPUT_VALUES);
        }

        /* Get input image data */
        unsigned char *input_image_data;
        size_t input_size = get_image_data(i, &input_image_data);

        // save the content of input_image_data to a file for debugging

        FILE *f = fopen("/home/root/debug_input_image.raw", "wb");
        fwrite(input_image_data, 1, input_size, f);
        fclose(f);

        uint8_t *output_image_data = (uint8_t *)malloc(input_size);
        if (output_image_data == NULL)
        {
            signal_error_and_exit(MALLOC_ERR);
        }

        memcpy(output_image_data, input_image_data, input_size);

        Metadata new_meta = METADATA__INIT;
        if (clone_metadata(input_meta, &new_meta) != 0)
        {
            signal_error_and_exit(MALLOC_ERR);
        }

        /* Append the processed image to the result batch */
        append_result_image(output_image_data, input_size, &new_meta);

        /* Free allocated memory */
        free(input_image_data);
        free(output_image_data);
    }
}

/* END MODULE IMPLEMENTATION */

/* Main function of module (NO NEED TO MODIFY) */
extern "C" ImageBatch run(ImageBatch *input_batch, ModuleParameterList *module_parameter_list, int *ipc_error_pipe)
{
    ImageBatch result_batch;
    result = &result_batch;
    input = input_batch;
    config = module_parameter_list;
    error_pipe = ipc_error_pipe;
    initialize();
    module();
    finalize();
    return result_batch;
}