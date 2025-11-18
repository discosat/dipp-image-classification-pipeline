#include "module.h"
#include "util.h"
#include "logger.h"
#include <filesystem>
#include <jxl/codestream_header.h>
#include <jxl/color_encoding.h>
#include <jxl/encode.h>
#include <jxl/encode_cxx.h>
#include <jxl/types.h>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;
/* Define custom error codes */
enum ERROR_CODE
{
    MALLOC_ERR = 1,
    ENCODE_ERR = 2,
    RUNNER_ERR = 3,
    INFO_ERR = 4,
    FRAME_ERR = 5,
    CHANNEL_ERR = 6,
    ENC_CLOSE_ERR = 7,
};

/* START MODULE IMPLEMENTATION */
void module()
{
    fs::path dir("/home/root/logs/");
    fs::path file_name("encode_" + std::to_string(std::time(0)) + ".txt");
    std::string full_path = (dir / file_name).string();
    Logger *logger = logger_create(full_path.c_str());

    logger_log_print(logger, LOG_INFO, "JPEGXL module started");

    for (int i = 0; i < input->num_images; i++)
    {
        logger_log(logger, LOG_INFO, "Full image started");

        logger_log(logger, LOG_INFO, "Getting metadata");
        Metadata *input_meta = get_metadata(i);
        int height = input_meta->height;
        int width = input_meta->width;
        uint32_t channels = input_meta->channels;
        int timestamp = input_meta->timestamp;
        int bits_pixel = input_meta->bits_pixel;
        char *camera = input_meta->camera;
        logger_log(logger, LOG_INFO, "Got metadata");

        logger_log(logger, LOG_INFO, "Getting image data.");
        uint8_t *image_buffer;
        size_t size = get_image_data(i, &image_buffer);
        logger_log(logger, LOG_INFO, "Got image data.");

        // Initialize encoder.
        logger_log(logger, LOG_INFO, "Initializing encoder.");
        JxlEncoderPtr encoder = JxlEncoderMake(nullptr);
        logger_log(logger, LOG_INFO, "Initialized encoder.");

        // Set basic information about the image.
        logger_log(logger, LOG_INFO, "Initializing basic info.");
        JxlPixelFormat pixel_format = {channels, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}; // RGB, 8-bit
        JxlBasicInfo basic_info = {};
        JxlEncoderInitBasicInfo(&basic_info);
        basic_info.xsize = width;
        basic_info.ysize = height;
        basic_info.bits_per_sample = 8;
        logger_log(logger, LOG_INFO, "Initialized basic info.");

        logger_log(logger, LOG_INFO, "Setting basic info.");
        if (JxlEncoderSetBasicInfo(encoder.get(), &basic_info) != JXL_ENC_SUCCESS)
        {
            std::cerr << "Failed to set basic info." << std::endl;
            signal_error_and_exit(INFO_ERR);
        }
        logger_log(logger, LOG_INFO, "Set basic info.");

        logger_log(logger, LOG_INFO, "Setting color encoding.");
        JxlColorEncoding color_encoding = {};
        JXL_BOOL is_gray = TO_JXL_BOOL(pixel_format.num_channels < 3);
        JxlColorEncodingSetToSRGB(&color_encoding, is_gray);
        if (JXL_ENC_SUCCESS !=
            JxlEncoderSetColorEncoding(encoder.get(), &color_encoding))
        {
            fprintf(stderr, "JxlEncoderSetColorEncoding failed\n");
            signal_error_and_exit(CHANNEL_ERR);
        }
        logger_log(logger, LOG_INFO, "Set color encoding.");

        logger_log(logger, LOG_INFO, "Creating frame settings.");
        JxlEncoderFrameSettings *frame_settings = JxlEncoderFrameSettingsCreate(encoder.get(), nullptr);
        logger_log(logger, LOG_INFO, "Created frame settings.");

        logger_log(logger, LOG_INFO, "Setting frame options.");
        int effort_level = get_param_int("effort_level");
        if (effort_level < 1 || effort_level > 10)
        {
            effort_level = 7; // default effort level as a fallback
        }
        JxlEncoderFrameSettingsSetOption(frame_settings, JXL_ENC_FRAME_SETTING_EFFORT, effort_level);

        // Set the image buffer.
        logger_log(logger, LOG_INFO, "Adding image frame.");
        if (JxlEncoderAddImageFrame(frame_settings, &pixel_format, image_buffer, size) != JXL_ENC_SUCCESS)
        {
            std::cerr << "Failed to add image frame." << std::endl;
            signal_error_and_exit(FRAME_ERR);
        }
        logger_log(logger, LOG_INFO, "Added image frame.");

        logger_log(logger, LOG_INFO, "Closing input");
        JxlEncoderCloseInput(encoder.get());
        logger_log(logger, LOG_INFO, "Closed input.");

        std::vector<uint8_t> compressed_data;

        // Encode the image.
        logger_log(logger, LOG_INFO, "Encoding image.");
        uint8_t buffer[4096];
        while (true)
        {
            uint8_t *next_out = buffer;
            size_t available_size = sizeof(buffer);
            JxlEncoderStatus status = JxlEncoderProcessOutput(encoder.get(), &next_out, &available_size);
            if (status == JXL_ENC_NEED_MORE_OUTPUT)
            {
                compressed_data.insert(compressed_data.end(), buffer, next_out);
            }
            else if (status == JXL_ENC_SUCCESS)
            {
                compressed_data.insert(compressed_data.end(), buffer, next_out);
                break;
            }
            else
            {
                std::cerr << "Encoding failed with status: " << status << std::endl;
                signal_error_and_exit(ENCODE_ERR);
            }
        }
        logger_log(logger, LOG_INFO, "Encoded image.");

        // logger_log(logger, LOG_INFO, "Saving encoded image.");
        // fs::path dir("/home/root/compressed/");
        // fs::path file_name("image_" + std::to_string(std::time(0)) + "_" + std::to_string(i) + ".jxl");
        // std::string full_path = (dir / file_name).string();
        // FILE *file = fopen(full_path.c_str(), "wb");
        // fwrite(compressed_data.data(), sizeof(uint8_t), compressed_data.size(), file);
        // fclose(file);
        // logger_log(logger, LOG_INFO, "Saved encoded image.");

        /* Create image metadata before appending */
        Metadata new_meta = METADATA__INIT;
        if (clone_metadata(input_meta, &new_meta) != 0)
        {
            signal_error_and_exit(MALLOC_ERR);
        }

        new_meta.size = compressed_data.size() * sizeof(uint8_t);

        /* Append the image to the result batch */
        logger_log(logger, LOG_INFO, "Appending image to result batch.");
        append_result_image(compressed_data.data(), new_meta.size, &new_meta);
        logger_log(logger, LOG_INFO, "Appended image to result batch.");

        free(image_buffer);
        logger_log(logger, LOG_INFO, "Full image finished");
    }
    logger_log_print(logger, LOG_INFO, "JPEGXL module finished");
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