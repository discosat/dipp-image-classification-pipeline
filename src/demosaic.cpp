#include "module.h"
#include "util.h"
#include "logger.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
namespace fs = std::filesystem;
/* Define custom error codes */
enum ERROR_CODE
{
    MALLOC_ERR = 1,
    PLACEHOLDER = 2,
};

/* START MODULE IMPLEMENTATION */
void module()
{
    fs::path dir("/home/root/logs/");
    fs::path file_name("demosaic_" + std::to_string(std::time(0)) + ".txt");
    std::string full_path = (dir / file_name).string();
    Logger *logger = logger_create(full_path.c_str());

    logger_log_print(logger, LOG_INFO, "Demosaic module started");

    for (int i = 0; i < input->num_images; i++)
    {
        logger_log(logger, LOG_INFO, "Full image started");

        logger_log(logger, LOG_INFO, "Getting metadata");
        Metadata *metadata = get_metadata(i);
        logger_log(logger, LOG_INFO, "Got metadata");

        logger_log(logger, LOG_INFO, "Getting image data.");
        uchar *image_buffer;
        size_t size = get_image_data(i, &image_buffer);
        logger_log(logger, LOG_INFO, "Got image data.");

        logger_log(logger, LOG_INFO, "Creating OpenCV Mat");
        cv::Mat rawImage(metadata->height, metadata->width, CV_16UC1, image_buffer);
        cv::Mat demosaicedImage;
        logger_log(logger, LOG_INFO, "Created OpenCV Mat");

        logger_log(logger, LOG_INFO, "Demosaicing image");
        cv::cvtColor(rawImage, demosaicedImage, cv::COLOR_BayerGR2RGB);
        demosaicedImage *= 16; // scale image to use 16 bits
        logger_log(logger, LOG_INFO, "Demosaiced image");

        logger_log(logger, LOG_INFO, "Normalizing image");
        cv::Mat demosaicedImage_1byte;
        cv::normalize(demosaicedImage, demosaicedImage_1byte, 0, 255, 32, CV_8UC3);
        logger_log(logger, LOG_INFO, "Normalized image");

        // fs::path dir("/home/root/demosaiced/");
        // fs::path file("image_" + std::to_string(std::time(0)) + "_" + std::to_string(i) + ".png");
        // std::string full_path = (dir / file).string();
        // imwrite(full_path, demosaicedImage);

        Metadata new_meta = METADATA__INIT;
        new_meta.channels = 3;
        new_meta.height = demosaicedImage_1byte.size().height;
        new_meta.width = demosaicedImage_1byte.size().width;
        new_meta.size = demosaicedImage_1byte.size().height * demosaicedImage_1byte.size().width * new_meta.channels;
        new_meta.timestamp = metadata->timestamp;
        new_meta.bits_pixel = 8;
        new_meta.camera = metadata->camera;

        /* Append the image to the result batch */
        logger_log(logger, LOG_INFO, "Appending image to result batch.");
        append_result_image(demosaicedImage_1byte.data, new_meta.size, &new_meta);
        logger_log(logger, LOG_INFO, "Appended image to result batch.");

        // Free the input image
        logger_log(logger, LOG_INFO, "Freeing input memory.");
        free(image_buffer);
        logger_log(logger, LOG_INFO, "Freed input memory.");

        logger_log(logger, LOG_INFO, "Full image finished");
    }
    logger_log_print(logger, LOG_INFO, "Demosaic module finished");
    logger_flush(logger);
    logger_destroy(logger);
}
/* END MODULE IMPLEMENTATION */

/* Main function of module (NO NEED TO MODIFY) */
ImageBatch run(ImageBatch *input_batch, ModuleParameterList *module_parameter_list, int *ipc_error_pipe)
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