#include "module.h"
#include "util.h"
#include "logger.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/delegates/external/external_delegate.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>
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
    INTERPRET_INIT = 2,
    TENSOR_ALLOC = 3,
    INFER_ERR = 4,
    ENCODE_ERR = 5,
    RUNNER_ERR = 6,
    INFO_ERR = 7,
    FRAME_ERR = 8,
    CHANNEL_ERR = 9,
    ENC_CLOSE_ERR = 10,
};

/* START MODULE IMPLEMENTATION */
void module()
{
    fs::path dir("/home/root/logs/");
    fs::path file_name("single_" + std::to_string(std::time(0)) + ".txt");
    std::string full_path = (dir / file_name).string();
    Logger *logger = logger_create(full_path.c_str());

    logger_log(logger, LOG_INFO, "Single module started");

    /* Retrieve module parameters by name (defined in config.yaml) */
    logger_log(logger, LOG_INFO, "getting model filename.");
    char *model_filename = get_param_string("model_filename");
    logger_log(logger, LOG_INFO, "got model filename.");

    // Load the model
    logger_log(logger, LOG_INFO, "Building model from file.");
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_filename);
    logger_log(logger, LOG_INFO, "Built model from file.");

    // Define resolver
    logger_log(logger, LOG_INFO, "Defining resolver");
    tflite::ops::builtin::BuiltinOpResolver resolver;
    logger_log(logger, LOG_INFO, "Resolver defined.");

    // Build the interpreter
    logger_log(logger, LOG_INFO, "building interpreter");
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    if (builder(&interpreter) != kTfLiteOk)
    {
        logger_log(logger, LOG_ERROR, "Failed to build interpreter");
        signal_error_and_exit(INTERPRET_INIT);
    }
    if (interpreter == nullptr)
    {
        logger_log(logger, LOG_ERROR, "Interpreter is null after successful build");
        signal_error_and_exit(INTERPRET_INIT);
    }
    logger_log(logger, LOG_INFO, "Interpreter built.");

    // Load the custom delegate
    logger_log(logger, LOG_INFO, "Loading delegate");
    auto ext_delegate_option =
        TfLiteExternalDelegateOptionsDefault("/usr/lib/libvx_delegate.so");
    auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&ext_delegate_option);
    logger_log(logger, LOG_INFO, "Delegate loaded.");

    // Modify the graph with delegate
    logger_log(logger, LOG_INFO, "Applying delegate");
    interpreter->ModifyGraphWithDelegate(ext_delegate_ptr);
    logger_log(logger, LOG_INFO, "Delegate applied to graph.");

    // Allocate the tensors and get the input tensor
    logger_log(logger, LOG_INFO, "allocating tensors");
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        signal_error_and_exit(TENSOR_ALLOC);
    }
    uint8_t *input_tensor = interpreter->typed_input_tensor<uint8_t>(0);
    logger_log(logger, LOG_INFO, "Tensor allocated.");

    // Get quantization parameters
    logger_log(logger, LOG_INFO, "Getting quantization parameters.");
    const auto *output_tensor = interpreter->output_tensor(0);
    const float scale = output_tensor->params.scale;
    const float zero_point = output_tensor->params.zero_point;
    logger_log(logger, LOG_INFO, "Got quantization parameters.");

    // Get output dimensions
    logger_log(logger, LOG_INFO, "Getting output dimensions.");
    int output = interpreter->outputs()[0];
    TfLiteIntArray *output_dims = interpreter->tensor(output)->dims;
    // assume output dims to be something like (1, 1, ... ,size)
    auto output_size = output_dims->data[output_dims->size - 1];
    logger_log(logger, LOG_INFO, "Got output dimensions.");

    // Initialize encoder.
    logger_log(logger, LOG_INFO, "Initializing encoder.");
    JxlEncoderPtr encoder = JxlEncoderMake(nullptr);
    logger_log(logger, LOG_INFO, "Initialized encoder.");

    for (int i = 0; i < input->num_images; i++)
    {
        logger_log(logger, LOG_INFO, "Full image started");

        logger_log(logger, LOG_INFO, "Getting metadata");
        Metadata *metadata = get_metadata(i);
        logger_log(logger, LOG_INFO, "Got metadata");

        logger_log(logger, LOG_INFO, "Getting image data");
        uchar *image_buffer;
        size_t size = get_image_data(i, &image_buffer);
        logger_log(logger, LOG_INFO, "Got image data");

        logger_log(logger, LOG_INFO, "Creating OpenCV Mat");
        cv::Mat rawImage(metadata->height, metadata->width, CV_16UC1, image_buffer);
        cv::Mat demosaicedImage;

        logger_log(logger, LOG_INFO, "Demosaicing image");
        cv::cvtColor(rawImage, demosaicedImage, cv::COLOR_BayerGR2RGB);
        demosaicedImage *= 16; // scale image to use 16 bits
        logger_log(logger, LOG_INFO, "Demosaiced image");

        logger_log(logger, LOG_INFO, "Normalizing image");
        cv::Mat demosaicedImage_1byte;
        cv::normalize(demosaicedImage, demosaicedImage_1byte, 0, 255, 32, CV_8UC3);
        logger_log(logger, LOG_INFO, "Normalized image");

        // Define output image (patch)
        constexpr uint8_t tile_size = 224;
        uint8_t channels = 3;
        int tile_bytes = tile_size * tile_size * channels * sizeof(uint8_t);
        int tile_idx = 0;

        uint8_t *input_image_data = demosaicedImage_1byte.data;
        uint16_t height = demosaicedImage_1byte.size().height;
        uint16_t width = demosaicedImage_1byte.size().width;

        for (uint16_t height_offset = 0; height_offset + tile_size <= height; height_offset += tile_size)
        {
            for (uint16_t width_offset = 0; width_offset + tile_size <= width; width_offset += tile_size)
            {

                // Define the arrays that store patches to be passed to next modules
                logger_log(logger, LOG_INFO, "allocating output image data.");
                uint8_t *output_image_data = (uint8_t *)malloc(tile_bytes);
                logger_log(logger, LOG_INFO, "allocated output image data.");

                /* Check for malloc error */
                if (output_image_data == NULL)
                {
                    signal_error_and_exit(MALLOC_ERR);
                }

                logger_log(logger, LOG_INFO, "Copying image data.");
                for (uint16_t h = height_offset; h < tile_size + height_offset; h++)
                {
                    // memcpy row of a patch into the output data
                    memcpy(
                        output_image_data + ((h - height_offset) * tile_size * channels),
                        input_image_data + (h * width * channels + width_offset * channels),
                        sizeof(uint8_t) * tile_size * channels);
                }
                logger_log(logger, LOG_INFO, "Copied image data.");

                // memcpy the entire output image into input tensor
                logger_log(logger, LOG_INFO, "Copying image data to tensor.");
                memcpy(
                    input_tensor,
                    output_image_data,
                    tile_bytes);
                logger_log(logger, LOG_INFO, "Copied image data to tensor.");

                // infer and deal with the result
                logger_log(logger, LOG_INFO, "Invoking");
                if (interpreter->Invoke() != kTfLiteOk)
                {
                    signal_error_and_exit(INFER_ERR);
                }
                logger_log(logger, LOG_INFO, "Invoked");

                // Get top class
                logger_log(logger, LOG_INFO, "Getting top class.");
                uint8_t *scores = interpreter->typed_output_tensor<uint8_t>(0);
                float max_val = -1.0;
                int max_cls = -1;
                for (int i = 0; i < output_size; i++)
                {
                    float scaled_score = static_cast<float>(scores[i] - zero_point) * scale;
                    if (scaled_score > max_val)
                    {
                        max_val = scaled_score;
                        max_cls = i;
                    }
                }
                logger_log(logger, LOG_INFO, "Got top class.");

                // send only 20% of the patches
                if (tile_idx % 5 == 0)
                {
                    // Set basic information about the image.
                    logger_log(logger, LOG_INFO, "Initializing basic info.");
                    JxlPixelFormat pixel_format = {channels, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}; // RGB, 8-bit
                    JxlBasicInfo basic_info = {};
                    JxlEncoderInitBasicInfo(&basic_info);
                    basic_info.xsize = tile_size;
                    basic_info.ysize = tile_size;
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

                    // Set the image buffer.
                    logger_log(logger, LOG_INFO, "Adding image frame.");
                    if (JxlEncoderAddImageFrame(frame_settings, &pixel_format, output_image_data, size) != JXL_ENC_SUCCESS)
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
                    logger_log(logger, LOG_INFO, "Encodeing image.");
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

                    Metadata new_meta = METADATA__INIT;
                    new_meta.size = compressed_data.size() * sizeof(uint8_t);
                    new_meta.width = tile_size;
                    new_meta.height = tile_size;
                    new_meta.channels = channels;
                    new_meta.timestamp = metadata->timestamp;
                    new_meta.bits_pixel = 8;
                    new_meta.camera = metadata->camera;

                    logger_log(logger, LOG_INFO, "Appending image to result batch.");
                    append_result_image(compressed_data.data(), new_meta.size, &new_meta);
                    logger_log(logger, LOG_INFO, "Appended image to result batch.");
                }
                tile_idx++;

                // Free the tile memory
                free(output_image_data);
            }
        }

        delete[] image_buffer;
        logger_log(logger, LOG_INFO, "Full image finished");
    }
    logger_log(logger, LOG_INFO, "Single module finished");
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