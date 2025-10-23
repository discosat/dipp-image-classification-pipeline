#include "module.h"
#include "util.h"
#include <opencv2/opencv.hpp>
#include <iostream>

/* Define custom error codes */
enum ERROR_CODE
{
    MALLOC_ERR = 1,
    OPENCV_ERR = 2,
    OPENCV_RES_ERR = 3,
    INVALID_INPUT = 7,
    INVALID_INPUT_VALUES = 8,
    INVALID_NEW_INPUT_VALUES = 9,

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

    /* Example code for iterating a pixel value at a time */
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

        unsigned char *input_image_data;
        size_t size = get_image_data(i, &input_image_data);

        // Retrieve target size
        logger_log(logger, LOG_INFO, "getting target size.");
        int target_size = get_param_int("target_size");
        logger_log(logger, LOG_INFO, "got target size.");

        // Calculate scale to fit within target_size while preserving aspect ratio
        double scale = std::min(static_cast<double>(target_size) / width,
                                static_cast<double>(target_size) / height);

        int new_width = static_cast<int>(width * scale);
        int new_height = static_cast<int>(height * scale);

        if (new_height <= 0 || new_width <= 0)
        {
            signal_error_and_exit(INVALID_NEW_INPUT_VALUES);
        }

        printf("[DEBUG]bits per pixel: %d\n", bits_pixel);
        printf("[DEBUG]channels; %d\n", channels);

        cv::Mat rawImage;
        if (channels == 1)
        {
            rawImage = cv::Mat(height, width, CV_16UC1, (uint16_t *)input_image_data);
        }
        else if (channels == 3)
        {
            rawImage = cv::Mat(height, width, CV_16UC3, (uint16_t *)input_image_data);
        }
        else
        {
            signal_error_and_exit(INVALID_INPUT_VALUES);
        }

        if (rawImage.empty() || rawImage.data == NULL)
        {
            signal_error_and_exit(OPENCV_ERR);
        }

        cv::Mat thumbnailImage;
        cv::resize(rawImage, thumbnailImage, cv::Size(new_width, new_height), 0, 0, cv::INTER_CUBIC);

        if (thumbnailImage.empty() || thumbnailImage.data == NULL)
        {
            signal_error_and_exit(OPENCV_RES_ERR);
        }

        /* Calculate output image size */
        size_t output_size = thumbnailImage.total() * thumbnailImage.elemSize();

        /* Allocate memory for output image data */
        unsigned char *output_image_data = (unsigned char *)malloc(output_size);

        /* Check for malloc error */
        if (output_image_data == NULL)
        {
            signal_error_and_exit(MALLOC_ERR);
        }

        /* Copy demosaiced data to output buffer */
        memcpy(output_image_data, thumbnailImage.data, output_size);

        /* Create output image metadata */
        Metadata new_meta = METADATA__INIT;
        new_meta.size = output_size;
        new_meta.width = new_width;
        new_meta.height = new_height;
        new_meta.channels = channels;
        new_meta.bits_pixel = input_meta->bits_pixel;
        new_meta.timestamp = input_meta->timestamp;
        new_meta.obid = input_meta->obid;
        new_meta.camera = input_meta->camera;

        /* Add custom metadata for demosaicing info */
        add_custom_metadata_int(&new_meta, "resized", target_size);

        /* Append the processed image to the result batch */
        append_result_image(output_image_data, output_size, &new_meta);

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