#include "module.h"
#include "util.h"
#include "globals.h"
#include <opencv2/opencv.hpp>

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

    int normalization_value = get_param_int("normalization_value");

    if (normalization_value != 255 && normalization_value != 65535)
    {
        signal_error_and_exit(INVALID_NORMALIZATION_VALUE);
    }

    /* Process each image in the batch */
    for (int i = 0; i < num_images; ++i)
    {
        /* Get input image metadata */
        Metadata *input_meta = get_metadata(i);
        int height = input_meta->height;
        int width = input_meta->width;
        int channels = input_meta->channels;

        if (height <= 0 || width <= 0 || channels <= 0)
        {
            signal_error_and_exit(INVALID_INPUT_VALUES);
        }

        /* Get input image data */
        unsigned char *input_image_data;
        size_t input_size = get_image_data(i, &input_image_data);

        /* Create OpenCV Mat for raw image (12-bit data in 16-bit container) */
        cv::Mat rawImage(height, width, CV_16UC1, (uint16_t *)input_image_data);

        if (rawImage.empty() || rawImage.data == NULL)
        {
            signal_error_and_exit(OPENCV_ERR);
        }

        /* Perform demosaicing with GRBG pattern */
        cv::Mat demosaicedImage;
        cv::cvtColor(rawImage, demosaicedImage, cv::COLOR_BayerRG2BGR);

        if (demosaicedImage.empty() || demosaicedImage.data == NULL)
        {
            signal_error_and_exit(OPENCV_DEM_ERR);
        }

        /* Apply vertical flip to match camera orientation */
        // cv::Mat finalImage;
        // cv::flip(demosaicedImage, finalImage, 0);  // 0 means vertical flip

        cv::Point2f center(width / 2.0f, height / 2.0f);
        double angle = 180;
        double scale = 1.0;

        cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, scale);

        if (rotation_matrix.empty())
        {
            signal_error_and_exit(OPNECV_MAT_ERR);
        }
        cv::Mat rotated_image;
        cv::warpAffine(demosaicedImage, rotated_image, rotation_matrix, cv::Size(width, height));

        if (rotated_image.empty() || rotated_image.data == NULL)
        {
            signal_error_and_exit(OPENCV_ROT_ERR);
        }

        // std::cout << "rotated_image = " << rotated_image << std::endl;

        cv::Mat normalized_Image;
        cv::normalize(rotated_image, normalized_Image, 0, normalization_value, cv::NORM_MINMAX);

        if (normalized_Image.empty() || normalized_Image.data == NULL)
        {
            signal_error_and_exit(OPENCV_NORM_ERR);
        }

        // std::cout << "normalized_Image = " << normalized_Image << std::endl;

        /* Calculate output image size */
        size_t output_size = normalized_Image.total() * normalized_Image.elemSize();

        /* Allocate memory for output image data */
        unsigned char *output_image_data = (unsigned char *)malloc(output_size);

        /* Check for malloc error */
        if (output_image_data == NULL)
        {
            signal_error_and_exit(MALLOC_ERR);
        }

        /* Copy demosaiced data to output buffer */
        memcpy(output_image_data, normalized_Image.data, output_size);

        // std::cout << "output_image_data copied" << std::endl;
        // for (int idx = 10000; idx < 10100; idx++)
        // {
        //     std::cout << "output_image_data[" << idx << "] = " << static_cast<int>(output_image_data[idx]) << std::endl;
        // }

        int output_bits_pixel;

        if (normalization_value == 255)
        {
            output_bits_pixel = 8; // 8-bit normalization
        }
        else if (normalization_value == 65535)
        {
            output_bits_pixel = 16; // 16-bit normalization
        }
        else
        {
            signal_error_and_exit(INVALID_NORMALIZATION_VALUE);
        }

        Metadata new_meta = METADATA__INIT;
        if (clone_metadata(input_meta, &new_meta) != 0)
        {
            signal_error_and_exit(MALLOC_ERR);
        }

        /* Create output image metadata */
        new_meta.size = output_size;
        new_meta.channels = 3; // BGR output
        new_meta.bits_pixel = output_bits_pixel;

        /* Add custom metadata for demosaicing info */
        add_custom_metadata_string(&new_meta, "processing", "demosaiced");
        add_custom_metadata_int(&new_meta, "output_channels", 3);
        add_custom_metadata_string(&new_meta, "orientation", "flipped_vertical");

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