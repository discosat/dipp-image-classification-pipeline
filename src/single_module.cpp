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
#include <memory>
#include <numeric> // For std::accumulate to calculate size
#include <cstdlib> // srand, rand
#include <ctime>   // time

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
    INVALID_INPUT = 11,
    INVALID_NORMALIZATION_VALUE = 12,
};

/* START MODULE IMPLEMENTATION */
void module()
{
    fs::path dir("/home/root/logs/");
    fs::path file_name("single_" + std::to_string(std::time(0)) + ".txt");
    std::string full_path = (dir / file_name).string();
    Logger *logger = logger_create(full_path.c_str());

    logger_log_print(logger, LOG_INFO, "Single module started");

    int num_images = get_input_num_images();

    if (num_images <= 0)
    {
        signal_error_and_exit(INVALID_INPUT);
    }

    /*
    ===============
    Demosaic setup
    ===============
    */
    // int normalization_value = get_param_int("normalization_value");
    int normalization_value = 255;

    if (normalization_value != 255 && normalization_value != 65535)
    {
        signal_error_and_exit(INVALID_NORMALIZATION_VALUE);
    }

    /*
    ===============
    Tiler setup
    ===============
    */
    logger_log(logger, LOG_INFO, "getting tile size.");
    // int tile_height = get_param_int("tile_height");
    // int tile_width = get_param_int("tile_width");
    int tile_height = 224;
    int tile_width = 224;
    logger_log(logger, LOG_INFO, "got tile size.");

    /*
    ===============
    Classifier setup
    ===============
    */
    /* Retrieve module parameters by name (defined in config.yaml) */
    logger_log(logger, LOG_INFO, "getting model filename.");
    char *model_filename_classifier = get_param_string("model_filename_classifier");
    logger_log(logger, LOG_INFO, "got model filename.");

    // Retrieve the class idx of interest
    // This is the class that will be kept, the other class will be set to all black for better compression
    logger_log(logger, LOG_INFO, "getting class index.");
    // int class_idx = get_param_int("class_index");
    int class_idx = 7;
    logger_log(logger, LOG_INFO, "got class index.");

    // Load the model
    logger_log(logger, LOG_INFO, "Building model from file.");
    std::unique_ptr<tflite::FlatBufferModel> classifier_model =
        tflite::FlatBufferModel::BuildFromFile(model_filename_classifier);
    logger_log(logger, LOG_INFO, "Built model from file.");

    // Define resolver
    logger_log(logger, LOG_INFO, "Defining resolver");
    tflite::ops::builtin::BuiltinOpResolver classifier_resolver;
    logger_log(logger, LOG_INFO, "Resolver defined.");

    // Build the interpreter
    logger_log(logger, LOG_INFO, "building interpreter");
    tflite::InterpreterBuilder builder(*classifier_model, classifier_resolver);
    std::unique_ptr<tflite::Interpreter> classifier_interpreter;
    if (builder(&classifier_interpreter) != kTfLiteOk)
    {
        logger_log(logger, LOG_ERROR, "Failed to build interpreter");
        signal_error_and_exit(INTERPRET_INIT);
    }
    if (classifier_interpreter == nullptr)
    {
        logger_log(logger, LOG_ERROR, "Interpreter is null after successful build");
        signal_error_and_exit(INTERPRET_INIT);
    }
    logger_log(logger, LOG_INFO, "Interpreter built.");

    // Load the custom delegate
    logger_log(logger, LOG_INFO, "Loading delegate");
    auto ext_delegate_option =
        TfLiteExternalDelegateOptionsDefault("/usr/lib/libvx_delegate.so");

    // set the caching options
    const char *allow_cache_key = "allowed_cache_mode";
    const char *allow_cache_value = "true";
    const char *cache_file_key = "cache_file_path";
    // get the last part of the model filename to use in cache file name
    std::string cache_filename_classifier = std::string("cache_") + fs::path(model_filename_classifier).filename().string();

    const char *cache_file_value_classifier = cache_filename_classifier.c_str();
    ext_delegate_option.insert(&ext_delegate_option, allow_cache_key, allow_cache_value);
    ext_delegate_option.insert(&ext_delegate_option, cache_file_key, cache_file_value_classifier);
    ext_delegate_option.insert(&ext_delegate_option, "error_during_init", allow_cache_value);
    ext_delegate_option.insert(&ext_delegate_option, "error_during_prepare", allow_cache_value);
    ext_delegate_option.insert(&ext_delegate_option, "error_during_invoke", allow_cache_value);

    auto ext_delegate_ptr_classifier = TfLiteExternalDelegateCreate(&ext_delegate_option);
    logger_log(logger, LOG_INFO, "Delegate loaded.");

    // Modify the graph with delegate
    logger_log(logger, LOG_INFO, "Applying delegate");
    if (classifier_interpreter->ModifyGraphWithDelegate(ext_delegate_ptr_classifier) != kTfLiteOk)
    {
        logger_log(logger, LOG_ERROR, "Failed to apply delegate to graph");
        signal_error_and_exit(INTERPRET_INIT);
    }
    logger_log(logger, LOG_INFO, "Delegate applied to graph.");

    // Allocate the tensors
    logger_log(logger, LOG_INFO, "allocating tensors");
    if (classifier_interpreter->AllocateTensors() != kTfLiteOk)
    {
        signal_error_and_exit(TENSOR_ALLOC);
    }
    logger_log(logger, LOG_INFO, "Tensors allocated.");

    // Get quantization parameters
    logger_log(logger, LOG_INFO, "Getting quantization parameters.");
    const auto *output_tensor_classifier = classifier_interpreter->output_tensor(0);
    const float scale_classifier = output_tensor_classifier->params.scale;
    const float zero_point_classifier = output_tensor_classifier->params.zero_point;
    logger_log(logger, LOG_INFO, "Got quantization parameters.");

    // Get output dimensions
    logger_log(logger, LOG_INFO, "Getting output dimensions.");
    int output_classifier = classifier_interpreter->outputs()[0];
    TfLiteIntArray *output_dims_classifier = classifier_interpreter->tensor(output_classifier)->dims;
    // assume output dims to be something like (1, 1, ... ,size)
    auto output_size_classifier = output_dims_classifier->data[output_dims_classifier->size - 1];

    logger_log(logger, LOG_INFO, "Got output dimensions.");

    /*
    ===============
    Segmentation setup
    ===============
    */
    /* Retrieve module parameters by name (defined in config.yaml) */
    logger_log(logger, LOG_INFO, "getting model filename.");
    char *model_filename_segmentation = get_param_string("model_filename_segmentation");
    logger_log(logger, LOG_INFO, "got model filename.");

    // Retrieve the class idx of interest
    // This is the class that will be kept, the other class will be set to all black for better compression
    logger_log(logger, LOG_INFO, "getting class index.");
    // int class_idx_segmentation = get_param_int("class_index_segmentation");
    int class_idx_segmentation = 0;
    logger_log(logger, LOG_INFO, "got class index.");

    logger_log(logger, LOG_INFO, "getting threshold percentage.");
    float threshold_percentage = get_param_float("threshold_percentage");
    logger_log(logger, LOG_INFO, "got threshold percentage.");

    // Load the model
    logger_log(logger, LOG_INFO, "Building model from file.");
    std::unique_ptr<tflite::FlatBufferModel> segmentation_model =
        tflite::FlatBufferModel::BuildFromFile(model_filename_segmentation);
    logger_log(logger, LOG_INFO, "Built model from file.");

    // Define resolver
    logger_log(logger, LOG_INFO, "Defining resolver");
    tflite::ops::builtin::BuiltinOpResolver segmentation_resolver;
    logger_log(logger, LOG_INFO, "Resolver defined.");

    // Build the interpreter
    logger_log(logger, LOG_INFO, "building interpreter");
    tflite::InterpreterBuilder builder_segmentation(*segmentation_model, segmentation_resolver);
    std::unique_ptr<tflite::Interpreter> segmentation_interpreter;
    if (builder_segmentation(&segmentation_interpreter) != kTfLiteOk)
    {
        logger_log(logger, LOG_ERROR, "Failed to build interpreter");
        signal_error_and_exit(INTERPRET_INIT);
    }
    if (segmentation_interpreter == nullptr)
    {
        logger_log(logger, LOG_ERROR, "Interpreter is null after successful build");
        signal_error_and_exit(INTERPRET_INIT);
    }
    logger_log(logger, LOG_INFO, "Interpreter built.");

    // Load the custom delegate
    // logger_log(logger, LOG_INFO, "Loading delegate");
    // auto ext_delegate_option_segmentation =
    //     TfLiteExternalDelegateOptionsDefault("/usr/lib/libvx_delegate.so");

    // // get the last part of the model filename to use in cache file name
    // std::string cache_filename_segmentation = std::string("cache_") + fs::path(model_filename_segmentation).filename().string();

    // const char *cache_file_value_segmentation = cache_filename_segmentation.c_str();
    // ext_delegate_option_segmentation.insert(&ext_delegate_option_segmentation, allow_cache_key, allow_cache_value);
    // ext_delegate_option_segmentation.insert(&ext_delegate_option_segmentation, cache_file_key, cache_file_value_segmentation);
    // ext_delegate_option_segmentation.insert(&ext_delegate_option_segmentation, "error_during_init", allow_cache_value);
    // ext_delegate_option_segmentation.insert(&ext_delegate_option_segmentation, "error_during_prepare", allow_cache_value);
    // ext_delegate_option_segmentation.insert(&ext_delegate_option_segmentation, "error_during_invoke", allow_cache_value);
    // auto ext_delegate_ptr_segmentation = TfLiteExternalDelegateCreate(&ext_delegate_option_segmentation);
    // logger_log(logger, LOG_INFO, "Delegate loaded.");

    // Modify the graph with delegate
    logger_log(logger, LOG_INFO, "Applying delegate");
    if (segmentation_interpreter->ModifyGraphWithDelegate(ext_delegate_ptr_classifier) != kTfLiteOk)
    {
        logger_log(logger, LOG_ERROR, "Failed to apply delegate to graph");
        signal_error_and_exit(INTERPRET_INIT);
    }
    logger_log(logger, LOG_INFO, "Delegate applied to graph.");

    // Allocate the tensors and get the input tensor
    logger_log(logger, LOG_INFO, "allocating tensors");
    if (segmentation_interpreter->AllocateTensors() != kTfLiteOk)
    {
        signal_error_and_exit(TENSOR_ALLOC);
    }
    uint8_t *input_tensor_segmentation = segmentation_interpreter->typed_input_tensor<uint8_t>(0);
    logger_log(logger, LOG_INFO, "Tensor allocated.");

    // Get quantization parameters
    logger_log(logger, LOG_INFO, "Getting quantization parameters.");
    const auto *output_tensor_segmentation = segmentation_interpreter->output_tensor(0);
    const float scale_segmentation = output_tensor_segmentation->params.scale;
    const float zero_point_segmentation = output_tensor_segmentation->params.zero_point;
    logger_log(logger, LOG_INFO, "Got quantization parameters.");

    /*
    ===============
    JPEGXL setup
    ===============
    */
    int effort = get_param_int("effort_level");

    for (int i = 0; i < num_images; ++i)
    {
        /*
        ===============
        Demosaic logic
        ===============
        */
        logger_log(logger, LOG_INFO, "Demosaic started");

        /* Get input image metadata */
        Metadata *input_meta = get_metadata(i);
        int height = input_meta->height;
        int width = input_meta->width;
        int channels = input_meta->channels;

        if (height <= 0 || width <= 0 || channels <= 0)
        {
            signal_error_and_exit(INVALID_INPUT);
        }

        /* Get input image data */
        unsigned char *input_image_data;
        get_image_data(i, &input_image_data);

        /* Create OpenCV Mat for raw image (12-bit data in 16-bit container) */
        cv::Mat rawImage(height, width, CV_16UC1, (uint16_t *)input_image_data);

        if (rawImage.empty() || rawImage.data == NULL)
        {
            logger_log(logger, LOG_ERROR, "OpenCV raw image creation failed");
            signal_error_and_exit(FRAME_ERR);
        }

        /* Perform demosaicing with GRBG pattern */
        cv::Mat demosaicedImage;
        cv::cvtColor(rawImage, demosaicedImage, cv::COLOR_BayerRG2BGR);

        if (demosaicedImage.empty() || demosaicedImage.data == NULL)
        {
            logger_log(logger, LOG_ERROR, "Demosaicing failed");
            signal_error_and_exit(FRAME_ERR);
        }

        /* Apply rotation */
        cv::Point2f center(width / 2.0f, height / 2.0f);
        double angle = 180;
        double scale = 1.0;

        cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, scale);

        if (rotation_matrix.empty())
        {
            logger_log(logger, LOG_ERROR, "Rotation matrix creation failed");
            signal_error_and_exit(FRAME_ERR);
        }

        cv::Mat rotated_image;
        cv::warpAffine(demosaicedImage, rotated_image, rotation_matrix, cv::Size(width, height));

        if (rotated_image.empty() || rotated_image.data == NULL)
        {
            logger_log(logger, LOG_ERROR, "Rotation failed");
            signal_error_and_exit(FRAME_ERR);
        }

        cv::Mat normalized_Image;
        cv::normalize(rotated_image, normalized_Image, 0, normalization_value, cv::NORM_MINMAX);

        if (normalized_Image.empty() || normalized_Image.data == NULL)
        {
            logger_log(logger, LOG_ERROR, "Normalization failed");
            signal_error_and_exit(FRAME_ERR);
        }

        /* Calculate demosaiced image size */
        size_t demosaiced_size = normalized_Image.total() * normalized_Image.elemSize();

        /* Allocate memory for demosaiced image data */
        unsigned char *demosaiced_data = (unsigned char *)malloc(demosaiced_size);

        if (demosaiced_data == NULL)
        {
            signal_error_and_exit(MALLOC_ERR);
        }

        /* Copy demosaiced data to buffer */
        memcpy(demosaiced_data, normalized_Image.data, demosaiced_size);

        logger_log(logger, LOG_INFO, "Demosaic finished");

        /*
        ===============
        Tiler logic
        ===============
        */
        logger_log(logger, LOG_INFO, "Tiling started");

        // Parse demosaiced data into uint8_t array
        size_t parsed_size = demosaiced_size / 2; // number of output pixels
        uint8_t *parsed_data = (uint8_t *)malloc(parsed_size);
        if (parsed_data == NULL)
        {
            signal_error_and_exit(MALLOC_ERR);
        }

        if (normalization_value == 255)
        {
            for (size_t pix = 0; pix < parsed_size; pix++)
            {
                parsed_data[pix] = demosaiced_data[2 * pix];
            }
        }
        else if (normalization_value == 65535)
        {
            int shift = 8;
            for (size_t pix = 0; pix < parsed_size; pix++)
            {
                uint16_t val = (uint16_t)demosaiced_data[2 * pix] | ((uint16_t)demosaiced_data[2 * pix + 1] << 8);
                parsed_data[pix] = (uint8_t)(val >> shift);
            }
        }
        else
        {
            signal_error_and_exit(INVALID_NORMALIZATION_VALUE);
        }

        int demosaiced_channels = 3; // BGR output
        int tile_bytes = tile_height * tile_width * demosaiced_channels * sizeof(uint8_t);
        int tile_idx = 0;

        for (uint16_t height_offset = 0; height_offset + tile_height <= height; height_offset += tile_height)
        {
            for (uint16_t width_offset = 0; width_offset + tile_width <= width; width_offset += tile_width)
            {
                logger_log(logger, LOG_INFO, "Processing tile");

                /* Allocate tile buffer */
                uint8_t *tile_data = (uint8_t *)malloc(tile_bytes);
                if (tile_data == NULL)
                {
                    signal_error_and_exit(MALLOC_ERR);
                }

                /* Copy tile data */
                for (uint16_t h = height_offset; h < tile_height + height_offset; h++)
                {
                    memcpy(
                        tile_data + ((h - height_offset) * tile_width * demosaiced_channels),
                        parsed_data + (h * width * demosaiced_channels + width_offset * demosaiced_channels),
                        sizeof(uint8_t) * tile_width * demosaiced_channels);
                }

                /*
                ===============
                Classification logic
                ===============
                */
                logger_log(logger, LOG_INFO, "Classification started");

                // Get input tensor pointer
                int input_index_classifier = classifier_interpreter->inputs()[0];
                uint8_t *input_tensor_classifier = classifier_interpreter->typed_tensor<uint8_t>(input_index_classifier);

                TfLiteTensor *in_tensor_classifier = classifier_interpreter->tensor(input_index_classifier);
                size_t expected_bytes_classifier = in_tensor_classifier->bytes;

                // Copy tile data to classifier tensor
                memcpy(input_tensor_classifier, tile_data, expected_bytes_classifier);

                // Invoke classifier
                if (classifier_interpreter->Invoke() != kTfLiteOk)
                {
                    logger_log(logger, LOG_ERROR, "Classification invoke failed");
                    signal_error_and_exit(INFER_ERR);
                }

                // Get top class
                float max_val_classifier = -1.0;
                int max_cls_classifier = -1;
                uint8_t *scores_classifier = classifier_interpreter->typed_output_tensor<uint8_t>(0);
                for (int j = 0; j < output_size_classifier; ++j)
                {
                    float scaled_score = static_cast<float>(scores_classifier[j] - zero_point_classifier) * scale_classifier;
                    if (scaled_score > max_val_classifier)
                    {
                        max_val_classifier = scaled_score;
                        max_cls_classifier = j;
                    }
                }

                char buffer[100];
                sprintf(buffer, "Tile %d: Class %d with score %.4f", tile_idx, max_cls_classifier, max_val_classifier);
                logger_log(logger, LOG_INFO, buffer);

                // Only process tiles matching the class of interest
                if (max_cls_classifier == class_idx)
                {
                    logger_log(logger, LOG_INFO, "Tile matches class of interest");

                    /*
                    ===============
                    Segmentation logic
                    ===============
                    */
                    logger_log(logger, LOG_INFO, "Segmentation started");

                    // Copy tile data to segmentation tensor
                    memcpy(input_tensor_segmentation, tile_data, tile_bytes);

                    // Invoke segmentation
                    if (segmentation_interpreter->Invoke() != kTfLiteOk)
                    {
                        logger_log(logger, LOG_ERROR, "Segmentation invoke failed");
                        signal_error_and_exit(INFER_ERR);
                    }

                    // Apply segmentation mask
                    uint8_t *scores_segmentation = segmentation_interpreter->typed_output_tensor<uint8_t>(0);

                    int tile_size = tile_height * tile_width * demosaiced_channels;
                    int kept_pixels = tile_size;
                    for (int pix = 0; pix < tile_height * tile_width; pix++)
                    {
                        float scaled_score = static_cast<float>(scores_segmentation[pix] - zero_point_segmentation) * scale_segmentation;
                        bool keep = true;
                        if (scaled_score < 0.5)
                        {
                            if (class_idx_segmentation != 0)
                            {
                                keep = false;
                            }
                        }
                        else
                        {
                            if (class_idx_segmentation != 1)
                            {
                                keep = false;
                            }
                        }

                        if (!keep)
                        {
                            tile_data[pix * demosaiced_channels + 0] = 0;
                            tile_data[pix * demosaiced_channels + 1] = 0;
                            tile_data[pix * demosaiced_channels + 2] = 0;
                            kept_pixels -= demosaiced_channels;
                        }
                    }

                    sprintf(buffer, "Kept pixels: %d out of %d", kept_pixels, tile_size);
                    logger_log(logger, LOG_INFO, buffer);

                    if ((float)kept_pixels > threshold_percentage * (float)tile_size)
                    {
                        logger_log(logger, LOG_INFO, "Tile passes threshold, encoding");

                        /*
                        ===============
                        JPEG XL encoding logic
                        ===============
                        */

                        JxlEncoder *encoder = JxlEncoderCreate(NULL);
                        if (encoder == NULL)
                        {
                            logger_log(logger, LOG_ERROR, "JXL encoder creation failed");
                            signal_error_and_exit(ENC_CLOSE_ERR);
                        }

                        JxlEncoderFrameSettings *settings = JxlEncoderFrameSettingsCreate(encoder, NULL);
                        if (JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, effort))
                        {
                            logger_log(logger, LOG_ERROR, "JXL set effort failed");
                            signal_error_and_exit(ENCODE_ERR);
                        }

                        JxlBasicInfo basic_info;
                        JxlEncoderInitBasicInfo(&basic_info);
                        basic_info.xsize = tile_width;
                        basic_info.ysize = tile_height;
                        basic_info.num_color_channels = demosaiced_channels > 3 ? 3 : demosaiced_channels;
                        basic_info.num_extra_channels = demosaiced_channels - basic_info.num_color_channels;
                        basic_info.bits_per_sample = 8; // tiles are 8-bit
                        basic_info.alpha_bits = basic_info.num_extra_channels > 0 ? 8 : 0;

                        JxlPixelFormat format = {demosaiced_channels, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};

                        if (JxlEncoderSetBasicInfo(encoder, &basic_info))
                        {
                            logger_log(logger, LOG_ERROR, "JXL set basic info failed");
                            signal_error_and_exit(INFO_ERR);
                        }

                        if (JxlEncoderAddImageFrame(settings, &format, tile_data, tile_bytes))
                        {
                            logger_log(logger, LOG_ERROR, "JXL add image frame failed");
                            signal_error_and_exit(ENCODE_ERR);
                        }

                        JxlEncoderCloseInput(encoder);

                        size_t output_buffer_size = tile_bytes;
                        size_t out_buf_remain = output_buffer_size;
                        uint8_t *output_buffer = (uint8_t *)malloc(output_buffer_size);
                        if (output_buffer == NULL)
                        {
                            signal_error_and_exit(MALLOC_ERR);
                        }

                        uint8_t *out_buf_next = output_buffer;
                        if (JxlEncoderProcessOutput(encoder, &out_buf_next, &out_buf_remain))
                        {
                            logger_log(logger, LOG_ERROR, "JXL process output failed");
                            signal_error_and_exit(ENCODE_ERR);
                        }

                        int enc_size = output_buffer_size - out_buf_remain;

                        // Create metadata for output
                        Metadata new_meta = METADATA__INIT;
                        if (clone_metadata(input_meta, &new_meta) != 0)
                        {
                            signal_error_and_exit(MALLOC_ERR);
                        }

                        new_meta.size = enc_size;
                        new_meta.width = tile_width;
                        new_meta.height = tile_height;
                        new_meta.channels = demosaiced_channels;
                        new_meta.bits_pixel = 8;

                        add_custom_metadata_int(&new_meta, "tile_idx", tile_idx);
                        add_custom_metadata_int(&new_meta, "prediction", max_cls_classifier);
                        add_custom_metadata_string(&new_meta, "enc", "jxl");
                        add_custom_metadata_string(&new_meta, "processing", "demosaiced");

                        append_result_image(output_buffer, enc_size, &new_meta);

                        free(output_buffer);
                        JxlEncoderDestroy(encoder);

                        logger_log(logger, LOG_INFO, "Tile encoded and appended");
                    }
                    else
                    {
                        logger_log(logger, LOG_INFO, "Tile filtered by threshold");
                    }
                }
                else
                {
                    logger_log(logger, LOG_INFO, "Tile filtered by classification");
                }

                free(tile_data);
                tile_idx++;
            }
        }

        free(parsed_data);
        free(demosaiced_data);
        free(input_image_data);

        logger_log(logger, LOG_INFO, "Full image finished");
    }

    logger_log_print(logger, LOG_INFO, "Single module finished");
    logger_flush(logger);
    logger_destroy(logger);
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
