#include "module.h"
#include "util.h"
#include "logger.h"

#include <filesystem>
#include <tensorflow/lite/delegates/external/external_delegate.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>

#include <iostream>
#include <memory>
#include <vector>
#include <ctime> // time

namespace fs = std::filesystem;
/* Define custom error codes */
enum ERROR_CODE
{
    MALLOC_ERR = 1,
    INTERPRET_INIT = 2,
    TENSOR_ALLOC = 3,
    INFER_ERR = 4,
};

/* START MODULE IMPLEMENTATION */
void module()
{
    fs::path dir("/home/root/logs/");
    fs::path file_name("tflite_classification_" + std::to_string(std::time(0)) + ".txt");
    std::string full_path = (dir / file_name).string();
    Logger *logger = logger_create(full_path.c_str());

    logger_log_print(logger, LOG_INFO, "TFlite classification module started");

    /* Get number of images in input batch */
    logger_log(logger, LOG_INFO, "Getting number of images in input batch.");
    int num_images = get_input_num_images();
    logger_log(logger, LOG_INFO, "Got number of images in input batch.");

    /* Retrieve module parameters by name (defined in config.yaml) */
    logger_log(logger, LOG_INFO, "getting model filename.");
    char *model_filename = get_param_string("model_filename");
    logger_log(logger, LOG_INFO, "got model filename.");

    // Retrieve the class idx of interest
    // This is the class that will be kept, the other class will be set to all black for better compression
    logger_log(logger, LOG_INFO, "getting class index.");
    int class_idx = get_param_int("class_index");
    logger_log(logger, LOG_INFO, "got class index.");

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

    // set the caching options
    const char *allow_cache_key = "allowed_cache_mode";
    const char *allow_cache_value = "true";
    const char *cache_file_key = "cache_file_path";
    // get the last part of the model filename to use in cache file name
    std::string cache_filename = std::string("cache_") + fs::path(model_filename).filename().string();

    const char *cache_file_value = cache_filename.c_str();
    ext_delegate_option.insert(&ext_delegate_option, allow_cache_key, allow_cache_value);
    ext_delegate_option.insert(&ext_delegate_option, cache_file_key, cache_file_value);
    ext_delegate_option.insert(&ext_delegate_option, "error_during_init", allow_cache_value);
    ext_delegate_option.insert(&ext_delegate_option, "error_during_prepare", allow_cache_value);
    ext_delegate_option.insert(&ext_delegate_option, "error_during_invoke", allow_cache_value);

    auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&ext_delegate_option);
    logger_log(logger, LOG_INFO, "Delegate loaded.");

    // Modify the graph with delegate
    logger_log(logger, LOG_INFO, "Applying delegate");
    if (interpreter->ModifyGraphWithDelegate(ext_delegate_ptr) != kTfLiteOk)
    {
        logger_log(logger, LOG_ERROR, "Failed to apply delegate to graph");
        signal_error_and_exit(INTERPRET_INIT);
    }
    logger_log(logger, LOG_INFO, "Delegate applied to graph.");

    // Allocate the tensors
    logger_log(logger, LOG_INFO, "allocating tensors");
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        signal_error_and_exit(TENSOR_ALLOC);
    }
    logger_log(logger, LOG_INFO, "Tensors allocated.");

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

    for (int i = 0; i < num_images; ++i)
    {
        logger_log(logger, LOG_INFO, "Full image started");

        logger_log(logger, LOG_INFO, "Getting metadata");
        Metadata *input_meta = get_metadata(i);
        // int height = input_meta->height;
        // int width = input_meta->width;
        // int channels = input_meta->channels;
        // int bits_pixel = input_meta->bits_pixel;
        logger_log(logger, LOG_INFO, "Got metadata");

        logger_log(logger, LOG_INFO, "Getting image data");
        uint8_t *input_image_data;
        size_t size = get_image_data(i, &input_image_data);
        logger_log(logger, LOG_INFO, "Got image data.");

        // Get input tensor pointer and expected size each iteration (avoid stale pointer)
        int input_index = interpreter->inputs()[0];
        uint8_t *input_tensor = interpreter->typed_tensor<uint8_t>(input_index);

        TfLiteTensor *in_tensor = interpreter->tensor(input_index);
        size_t expected_bytes = in_tensor->bytes;

        // Copy image data to tensor
        logger_log(logger, LOG_INFO, "Copying image data to tensor.");
        memcpy(input_tensor, input_image_data, expected_bytes);
        logger_log(logger, LOG_INFO, "Copied image data to tensor.");

        // infer and deal with the result
        logger_log(logger, LOG_INFO, "Invoking");
        if (interpreter->Invoke() != kTfLiteOk)
        {
            signal_error_and_exit(INFER_ERR);
        }
        logger_log(logger, LOG_INFO, "Invoked");

        // Get top class
        logger_log(logger, LOG_INFO, "Getting the top class.");
        // uint8_t *scores = interpreter->typed_output_tensor<uint8_t>(0);
        float max_val = -1.0;
        int max_cls = -1;
        uint8_t *scores = interpreter->typed_output_tensor<uint8_t>(0);
        for (int j = 0; j < output_size; ++j)
        {
            float scaled_score = static_cast<float>(scores[j] - zero_point) * scale;
            if (scaled_score > max_val)
            {
                max_val = scaled_score;
                max_cls = j;
            }
        }

        char buffer[100];
        sprintf(buffer, "Top class: %d with score %.4f", max_cls, max_val);
        logger_log(logger, LOG_INFO, buffer);
        // logger_log(logger, LOG_INFO, "Got the top class.");

        // send only the patches that match the class idx of interest
        if (max_cls == class_idx)
        {
            logger_log(logger, LOG_INFO, "Adding the image to the result batch.");
            Metadata new_meta = METADATA__INIT;
            if (clone_metadata(input_meta, &new_meta) != 0)
            {
                signal_error_and_exit(MALLOC_ERR);
            }

            /* Add custom metadata key-value for prediction */
            logger_log(logger, LOG_INFO, "Adding custom metadata.");
            add_custom_metadata_int(&new_meta, "prediction", max_cls);
            logger_log(logger, LOG_INFO, "Added custom metadata.");

            /* Append the image to the result batch */
            logger_log(logger, LOG_INFO, "Appending image to result batch.");
            append_result_image(input_image_data, size, &new_meta);
            logger_log(logger, LOG_INFO, "Appended image to result batch.");
        }

        // Free the input image
        logger_log(logger, LOG_INFO, "Freeing input memory.");
        free(input_image_data);
        logger_log(logger, LOG_INFO, "Freed input memory.");

        logger_log(logger, LOG_INFO, "Full image finished");
    }

    logger_log_print(logger, LOG_INFO, "TFlite classification module finished");
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
