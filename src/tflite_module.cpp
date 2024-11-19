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

namespace fs = std::filesystem;
/* Define custom error codes */
enum ERROR_CODE {
    MALLOC_ERR = 1,
    INTERPRET_INIT = 2,
    TENSOR_ALLOC = 3,
    INFER_ERR = 4,
};

/* START MODULE IMPLEMENTATION */
void module()
{
    fs::path dir ("/home/root/logs/");
    fs::path file_name ("batch_" + std::to_string(std::time(0)) + ".txt");
    std::string full_path = (dir / file_name).string();
    Logger* logger = logger_create(full_path.c_str());

    /* Get number of images in input batch */
    int num_images = get_input_num_images();
    logger_log(logger, LOG_INFO, "Got number of images in input batch.");

    /* Retrieve module parameters by name (defined in config.yaml) */
    char *model_filename = get_param_string("model_filename");
    logger_log(logger, LOG_INFO, "got model filename.");

    // Load the model
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_filename);
    logger_log(logger, LOG_INFO, "Built model from file.");

    // Define resolver
    tflite::ops::builtin::BuiltinOpResolver resolver;
    logger_log(logger, LOG_INFO, "Resolver defined.");

    // Build the interpreter
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    if (builder(&interpreter) != kTfLiteOk) {
    logger_log(logger, LOG_ERROR, "Failed to build interpreter");
    signal_error_and_exit(INTERPRET_INIT);
    }
    if (interpreter == nullptr) {
        logger_log(logger, LOG_ERROR, "Interpreter is null after successful build");
        signal_error_and_exit(INTERPRET_INIT);
    }
    logger_log(logger, LOG_INFO, "Interpreter built.");

    // Load the custom delegate
    auto ext_delegate_option =
      TfLiteExternalDelegateOptionsDefault("/usr/lib/libvx_delegate.so");
    auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&ext_delegate_option);
    logger_log(logger, LOG_INFO, "Delegate loaded.");

    // Modify the graph with delegate
    interpreter->ModifyGraphWithDelegate(ext_delegate_ptr);
    logger_log(logger, LOG_INFO, "Delegate applied to graph.");

    // Allocate the tensors and get the input tensor
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        signal_error_and_exit(TENSOR_ALLOC);
    }
    uint8_t *input_tensor = interpreter->typed_input_tensor<uint8_t>(0);
    logger_log(logger, LOG_INFO, "Tensor allocated.");

    // Get quantization parameters
    const auto* output_tensor = interpreter->output_tensor(0);
    const float scale = output_tensor->params.scale;
    const float zero_point = output_tensor->params.zero_point;
    logger_log(logger, LOG_INFO, "Got quantization parameters.");

    // Get output dimensions
    int output = interpreter->outputs()[0];
    TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
    // assume output dims to be something like (1, 1, ... ,size)
    auto output_size = output_dims->data[output_dims->size - 1];
    logger_log(logger, LOG_INFO, "Got output dimensions.");

    /* Example code for iterating a pixel value at a time */
    for (int i = 0; i < num_images; ++i)
    {
        logger_log(logger, LOG_INFO, "Started the image loop.");
        Metadata *input_meta = get_metadata(i);
        int height = input_meta->height;
        int width = input_meta->width;
        int channels = input_meta->channels;
        int timestamp = input_meta->timestamp;
        int bits_pixel = input_meta->bits_pixel;
        char *camera = input_meta->camera;
        
        uint8_t *input_image_data;
        size_t size = get_image_data(i, &input_image_data);
        logger_log(logger, LOG_INFO, "Got image data.");

        // Define output image (patch)
        constexpr uint8_t tile_size = 224;
        int tile_bytes = tile_size * tile_size * channels * sizeof(uint8_t);

        for ( uint16_t height_offset = 0; height_offset + tile_size <= height; height_offset += tile_size) 
        {
            for ( uint16_t width_offset = 0; width_offset + tile_size <= width; width_offset += tile_size) 
            {

                // Define the arrays that store patches to be passed to next modules
                uint8_t *output_image_data = (uint8_t *)malloc(tile_bytes);
                logger_log(logger, LOG_INFO, "allocated output image data.");

                /* Check for malloc error */
                if (output_image_data == NULL)
                {
                    signal_error_and_exit(MALLOC_ERR);
                }

                for (uint16_t h = height_offset; h < tile_size + height_offset; h++) 
                {
                    // memcpy row of a patch into the output data
                    memcpy(
                        output_image_data + ((h-height_offset) * tile_size * channels), 
                        input_image_data + (h * width * channels + width_offset * channels), 
                        sizeof(uint8_t) * tile_size * channels
                        );
				}
                logger_log(logger, LOG_INFO, "Copied image data.");

                // memcpy the entire output image into input tensor
                memcpy(
                    input_tensor,
                    output_image_data,
                    tile_bytes
                    );
                logger_log(logger, LOG_INFO, "Copied image data to tensor.");
                // infer and deal with the result
                if (interpreter->Invoke() != kTfLiteOk)
                {
                    signal_error_and_exit(INFER_ERR);
                }
                logger_log(logger, LOG_INFO, "Invoked");

                // Get top class
                uint8_t* scores = interpreter->typed_output_tensor<uint8_t>(0);
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
                
                /* Create image metadata before appending */
                Metadata new_meta = METADATA__INIT;
                new_meta.size = tile_bytes;
                new_meta.width = tile_size;
                new_meta.height = tile_size;
                new_meta.channels = channels;
                new_meta.timestamp = timestamp;
                new_meta.bits_pixel = bits_pixel;
                new_meta.camera = camera;

                /* Add custom metadata key-value for prediction */
                add_custom_metadata_int(&new_meta, "prediction", max_cls);
                logger_log(logger, LOG_INFO, "Added custom metadata.");

                /* Append the image to the result batch */
                append_result_image(output_image_data, tile_bytes, &new_meta);
                logger_log(logger, LOG_INFO, "Appended image to result batch.");

                // Free the tile memory
                free(output_image_data);
            }
        }
        
        // Free the input image
        free(input_image_data);
        logger_log(logger, LOG_INFO, "Next img.");

        
    }

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