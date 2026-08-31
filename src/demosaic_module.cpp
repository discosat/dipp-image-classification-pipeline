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

    /* Radiometric calibration. Optional so configs written before this existed
       keep working; the defaults are the fitted values. */
    int radiometric_mode = get_param_int_or("radiometric_mode", 1);
    float calib_gain_r = get_param_float_or("calib_gain_r", 0.040321f);
    float calib_gain_g = get_param_float_or("calib_gain_g", 0.031413f);
    float calib_gain_b = get_param_float_or("calib_gain_b", 0.025168f);
    float calib_off_r = get_param_float_or("calib_offset_r", 17.270f);
    float calib_off_g = get_param_float_or("calib_offset_g", 36.860f);
    float calib_off_b = get_param_float_or("calib_offset_b", 60.976f);

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
        if (radiometric_mode == 0)
        {
            // Legacy behaviour, kept only so the old results can be
            // reproduced. NORM_MINMAX takes its range from the whole frame's
            // extremes, so it behaves completely differently depending on
            // content: four of the five test frames already span 0..4096 and
            // it is nearly a no-op, while a uniformly cloudy frame spans
            // 3324..3838 and gets stretched 8x. That is what put all 99 tiles
            // of the full-cloud frame into a single class.
            cv::normalize(rotated_image, normalized_Image, 0, normalization_value, cv::NORM_MINMAX);
        }
        else
        {
            // Fixed per-band affine: absolute and scene-independent, so
            // identical ground radiance maps to identical pixel values in
            // every frame and a bright scene stays bright.
            //
            // Channel order: cvtColor(..., COLOR_BayerRG2BGR) nominally emits
            // BGR, but that constant does not match this sensor's mosaic
            // phase, so it swaps R and B and the array is in fact R,G,B.
            // Verified against the raw mosaic -- channel 0 tracks the phase
            // that is dark over water and channel 2 the one that is bright,
            // which is only physical for R and B respectively. The gains
            // below are therefore in R,G,B order.
            //
            // Defaults were fitted label-free, matching the pooled pixel
            // statistics of the demosaiced frames to those of the EuroSAT
            // corpus the models are trained on; the offsets rise toward blue,
            // the signature of Rayleigh path radiance. See
            // DIPP/analysis/stage4.5-models/calibration.json.
            cv::Mat f;
            rotated_image.convertTo(f, CV_32FC3);
            cv::multiply(f, cv::Scalar(calib_gain_r, calib_gain_g, calib_gain_b), f);
            cv::add(f, cv::Scalar(calib_off_r, calib_off_g, calib_off_b), f);
            cv::min(f, (double)normalization_value, f);
            cv::max(f, 0.0, f);
            f.convertTo(normalized_Image, CV_16UC3);
        }

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