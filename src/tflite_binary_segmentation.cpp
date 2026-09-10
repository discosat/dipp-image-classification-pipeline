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
enum ERROR_CODE
{
    MALLOC_ERR = 1,
    INTERPRET_INIT = 2,
    TENSOR_ALLOC = 3,
    INFER_ERR = 4,
    SHAPE_MISMATCH = 5,
};

/* START MODULE IMPLEMENTATION */
void module()
{
    fs::path dir("/home/root/logs/");
    fs::path file_name("tflite_segmentation_" + std::to_string(std::time(0)) + ".txt");
    std::string full_path = (dir / file_name).string();
    Logger *logger = logger_create(full_path.c_str());

    logger_log_print(logger, LOG_INFO, "TFlite segmentation module started");

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

    logger_log(logger, LOG_INFO, "getting threshold percentage.");
    float threshold_percentage = get_param_float("threshold_percentage");

    // Per-pixel decision threshold. Was hardcoded at 0.5, which is the point
    // that maximises IoU -- and IoU weighs the two error directions equally,
    // which this application does not. A false negative leaves cloud in the
    // tile; a false positive removes ground. Measured on 38-Cloud, moving this
    // from 0.5 to 0.9 takes falsely-masked clear ground from 4.0% to 1.2%.
    // _or: absent keys must not abort the module, and the fallbacks reproduce
    // the previous hardcoded behaviour exactly, so existing configs are
    // unaffected.
    float pixel_threshold = get_param_float_or("pixel_threshold", 0.5f);
    if (!(pixel_threshold > 0.0f && pixel_threshold < 1.0f))
        pixel_threshold = 0.5f;

    // How a masked pixel is filled. Black maximises the discontinuity at the
    // mask boundary, and that edge is what costs bits: on real tiles, black
    // fill made lossy JPEG XL +119% and JPEG 2000 +132% LARGER than not
    // masking at all. A light low-pass removes the expensive high-frequency
    // cloud texture without creating the edge -- 5x5 measured -5% (JXL) and
    // -28% (J2K), while leaving the masked region at 45.7 dB, so a false
    // positive degrades ground rather than deleting it. Larger kernels are
    // worse: they push the filled region further from its surroundings and
    // recreate the very discontinuity that black fill suffers from.
    int mask_lowpass = get_param_int_or("mask_lowpass", 0);  // 0 = black, else kernel
    if (mask_lowpass < 0 || (mask_lowpass > 0 && mask_lowpass % 2 == 0))
        mask_lowpass = 0;
    logger_log(logger, LOG_INFO, "got threshold percentage.");

    // Load the model
    logger_log(logger, LOG_INFO, "Building model from file.");
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_filename);
    logger_log(logger, LOG_INFO, "Built model from file.");
    if (model == nullptr)
    {
        // Null means a missing or malformed file; the InterpreterBuilder below
        // would otherwise dereference a null unique_ptr.
        logger_log(logger, LOG_ERROR, "Failed to load model file");
        signal_error_and_exit(INTERPRET_INIT);
    }

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

    // NPU delegate is optional so the same build can be run on the A53 cores
    // for comparison. The board's segmentation output sits above 0.5 on every
    // pixel of a cloud-free frame, while the identical flatbuffer on x86 CPU
    // returns 3.3% cloud -- so the question is whether the VX delegate is
    // computing something different, which is only answerable by running the
    // same graph without it. Section IV-A3 already records this failure mode
    // for EfficientNet's swish layers.
    int use_npu = get_param_int_or("use_npu", 1);
    if (!use_npu)
        logger_log_print(logger, LOG_INFO, "NPU delegate DISABLED (CPU inference)");

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

    /* NOTE: error_during_init/prepare/invoke are the VX delegate's error
       INJECTION switches, not error handling. They were being set to the
       same "true" string as the cache flag. */
    auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&ext_delegate_option);
    logger_log(logger, LOG_INFO, "Delegate loaded.");

    // Modify the graph with delegate
    if (use_npu)
    {
        logger_log(logger, LOG_INFO, "Applying delegate");
        if (interpreter->ModifyGraphWithDelegate(ext_delegate_ptr) != kTfLiteOk)
        {
            logger_log(logger, LOG_ERROR, "Failed to apply delegate to graph");
            signal_error_and_exit(INTERPRET_INIT);
        }
        logger_log(logger, LOG_INFO, "Delegate applied to graph.");
    }

    // Allocate the tensors and get the input tensor
    logger_log(logger, LOG_INFO, "allocating tensors");
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        signal_error_and_exit(TENSOR_ALLOC);
    }
    uint8_t *input_tensor = interpreter->typed_input_tensor<uint8_t>(0);
    const TfLiteTensor *in_tensor = interpreter->input_tensor(0);
    const int in_h = in_tensor->dims->data[1];
    const int in_w = in_tensor->dims->data[2];
    const int in_c = in_tensor->dims->data[3];
    logger_log(logger, LOG_INFO, "Tensor allocated.");

    // Get quantization parameters
    logger_log(logger, LOG_INFO, "Getting quantization parameters.");
    const auto *output_tensor = interpreter->output_tensor(0);
    const float scale = output_tensor->params.scale;
    const float zero_point = output_tensor->params.zero_point;
    logger_log(logger, LOG_INFO, "Got quantization parameters.");

    for (int i = 0; i < num_images; ++i)
    {
        logger_log(logger, LOG_INFO, "Full image started");

        logger_log(logger, LOG_INFO, "Getting metadata");
        Metadata *input_meta = get_metadata(i);
        int height = input_meta->height;
        int width = input_meta->width;
        int channels = input_meta->channels;
        logger_log(logger, LOG_INFO, "Got metadata");

        logger_log(logger, LOG_INFO, "Getting image data");
        uint8_t *input_image_data;
        size_t size = get_image_data(i, &input_image_data);
        logger_log(logger, LOG_INFO, "Got image data.");

        // The previous code copied `size` bytes into the tensor without
        // checking the tensor was that large -- a heap overflow the moment a
        // model with a smaller input was configured.
        if (size != (size_t)height * width * channels || channels != in_c)
        {
            char msg[160];
            snprintf(msg, sizeof(msg),
                     "shape mismatch: tile %dx%dx%d (%zu bytes) vs tensor %dx%dx%d",
                     height, width, channels, size, in_h, in_w, in_c);
            logger_log(logger, LOG_ERROR, msg);
            signal_error_and_exit(SHAPE_MISMATCH);
        }

        // Effort scales by downsampling the tile the module is handed, so the
        // ground area per inference stays constant. INTER_AREA matches the
        // downsampling used in training.
        logger_log(logger, LOG_INFO, "Copying image data to tensor.");
        cv::Mat tile(height, width, CV_8UC(channels), input_image_data);
        if (height != in_h || width != in_w)
        {
            cv::Mat scaled(in_h, in_w, CV_8UC(in_c), input_tensor);
            cv::resize(tile, scaled, cv::Size(in_w, in_h), 0, 0, cv::INTER_AREA);
        }
        else
        {
            memcpy(input_tensor, input_image_data, in_tensor->bytes);
        }
        logger_log(logger, LOG_INFO, "Copied image data to tensor.");

        // infer and deal with the result
        logger_log(logger, LOG_INFO, "Invoking");
        if (interpreter->Invoke() != kTfLiteOk)
        {
            signal_error_and_exit(INFER_ERR);
        }
        logger_log(logger, LOG_INFO, "Invoked");

        logger_log(logger, LOG_INFO, "Applying the segmentation mask.");
        uint8_t *raw_scores = interpreter->typed_output_tensor<uint8_t>(0);
        const TfLiteTensor *out_t = interpreter->output_tensor(0);
        const int out_h = out_t->dims->data[1];
        const int out_w = out_t->dims->data[2];

        // The mask is produced at the model's resolution. When the tile was
        // downsampled for a cheaper effort level, the mask must be lifted back
        // to the tile's resolution before it can index tile pixels -- the old
        // code indexed scores[pix] over the tile's height*width and would run
        // off the end of a smaller mask.
        cv::Mat mask_small(out_h, out_w, CV_8UC1, raw_scores);
        cv::Mat mask_full;
        if (out_h != height || out_w != width)
            cv::resize(mask_small, mask_full, cv::Size(width, height), 0, 0,
                       cv::INTER_LINEAR);
        else
            mask_full = mask_small;
        const uint8_t *scores = mask_full.data;

        int img_size = height * width * channels;
        int kept_pixels = img_size;

        // Low-pass source for masked pixels, computed once over the whole tile;
        // masked pixels are then read from it rather than being zeroed.
        cv::Mat blurred;
        if (mask_lowpass > 0)
        {
            cv::Mat src(height, width, channels == 1 ? CV_8UC1 : CV_8UC3,
                        input_image_data);
            cv::GaussianBlur(src, blurred, cv::Size(mask_lowpass, mask_lowpass), 0);
        }

        for (int pix = 0; pix < height * width; pix++)
        {
            float scaled_score = static_cast<float>(scores[pix] - zero_point) * scale;
            bool keep = true;
            if (scaled_score < pixel_threshold)
            {
                // set to class 0
                if (class_idx != 0)
                {
                    keep = false;
                }
            }
            else
            {
                if (class_idx != 1)
                {
                    keep = false;
                }
            }

            if (!keep)
            {
                if (mask_lowpass > 0)
                {
                    const uint8_t *b = blurred.data + pix * channels;
                    for (int c = 0; c < channels; c++)
                        input_image_data[pix * channels + c] = b[c];
                }
                else
                {
                    for (int c = 0; c < channels; c++)
                        input_image_data[pix * channels + c] = 0;
                }
                kept_pixels -= channels;
            }
        }
        logger_log(logger, LOG_INFO, "Applied the segmentation mask.");
        char buffer[100];
        sprintf(buffer, "Kept pixels: %d out of %d", kept_pixels, img_size);
        logger_log(logger, LOG_INFO, buffer);

        if ((float)kept_pixels > threshold_percentage * (float)img_size)
        {
            /* Append the image to the result batch */
            logger_log(logger, LOG_INFO, "Appending image to result batch.");
            Metadata new_meta = METADATA__INIT;
            if (clone_metadata(input_meta, &new_meta) != 0)
            {
                signal_error_and_exit(MALLOC_ERR);
            }
            append_result_image(input_image_data, size, &new_meta);
            logger_log(logger, LOG_INFO, "Appended image to result batch.");
        }

        // Free the input image
        logger_log(logger, LOG_INFO, "Freeing input memory.");
        free(input_image_data);
        logger_log(logger, LOG_INFO, "Freed input memory.");

        logger_log(logger, LOG_INFO, "Full image finished");
    }

    logger_log_print(logger, LOG_INFO, "TFlite segmentation module finished");
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
