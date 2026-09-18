#include "module.h"
#include "util.h"
#include "logger.h"
#include "j2k_encode.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/delegates/external/external_delegate.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <vector>
#include <memory>
#include <string>
#include <cstdlib>
#include <ctime>

namespace fs = std::filesystem;

/* The MONOLITH: demosaic -> tile -> classification -> segmentation -> JPEG 2000
 * in ONE process, the "no decomposition" arm of the comparison.
 *
 * For that comparison to mean anything the monolith must do exactly the work
 * the five modules do, with the same parameters, and differ only in being
 * one process with no inter-module hand-off. It did not: it hardcoded
 * class 7 (Residential -- the target is water, 9), a NORM_MINMAX rendering
 * the pipeline no longer uses, a warpAffine rotation, 224-px tiles, no
 * train_res resize, a JPEG XL encoder at a fixed effort, and a segmentation
 * gate without the mask fill. Every one of those now comes from the config
 * with the modules' own keys and defaults, and the encoder is the modules'
 * (utils/j2k_encode.c). Stage by stage:
 *
 *   demosaic      calib_gain_r (uniform gain, one-pass convertTo, rounded,
 *                 8-bit), cv::rotate 180         -- as demosaic_module.cpp
 *   tile          tile_size (224)                -- as tile_module.cpp
 *   classify      model_filename_classifier, class_index, train_res, NPU
 *                 via the VX delegate            -- as tflite_classification.cpp
 *   segment       model_filename_segmentation, pixel_threshold, mask_lowpass,
 *                 threshold_percentage, class_index_segmentation (0 = keep
 *                 clear)                         -- as tflite_binary_segmentation.cpp
 *   encode        compression_ratio, resolutions, quality_layers,
 *                 resync_markers                 -- as jpeg2000_module.c
 *
 * Both models are loaded once per execution and every tile that classification
 * keeps is segmented and encoded in turn, so the monolith commits to the whole
 * frame's work the moment it starts -- which is the property the paper
 * contrasts with per-module scheduling. */

enum ERROR_CODE
{
    MALLOC_ERR = 1,
    INTERPRET_INIT = 2,
    TENSOR_ALLOC = 3,
    INFER_ERR = 4,
    ENCODE_ERR = 5,
    INVALID_INPUT = 11,
    INVALID_NORMALIZATION_VALUE = 12,
    OPENCV_ERR = 13,
};

struct Model
{
    std::unique_ptr<tflite::FlatBufferModel> model;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    TfLiteDelegate *delegate = nullptr;
};

static std::string model_path(const char *name)
{
    std::string n(name ? name : "");
    if (n.find('/') == std::string::npos)
        n = "/usr/share/models/" + n;
    return n;
}

/* Load a flatbuffer onto the NPU exactly as the two model modules do: the VX
   external delegate with its compile cache keyed by model filename. */
static void load_model(Model &m, const char *filename, Logger *logger, bool use_npu)
{
    m.model = tflite::FlatBufferModel::BuildFromFile(filename);
    if (m.model == nullptr)
    {
        logger_log_print(logger, LOG_ERROR, "model file not found");
        signal_error_and_exit(INTERPRET_INIT);
    }
    tflite::InterpreterBuilder builder(*m.model, m.resolver);
    if (builder(&m.interpreter) != kTfLiteOk || m.interpreter == nullptr)
        signal_error_and_exit(INTERPRET_INIT);
    if (use_npu)
    {
        auto opt = TfLiteExternalDelegateOptionsDefault("/usr/lib/libvx_delegate.so");
        std::string cache = std::string("cache_") + fs::path(filename).filename().string();
        opt.insert(&opt, "allowed_cache_mode", "true");
        opt.insert(&opt, "cache_file_path", cache.c_str());
        m.delegate = TfLiteExternalDelegateCreate(&opt);
        if (m.interpreter->ModifyGraphWithDelegate(m.delegate) != kTfLiteOk)
            signal_error_and_exit(INTERPRET_INIT);
    }
    if (m.interpreter->AllocateTensors() != kTfLiteOk)
        signal_error_and_exit(TENSOR_ALLOC);
}

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
        signal_error_and_exit(INVALID_INPUT);

    /* ---- parameters: the modules' keys and defaults ---- */
    int normalization_value = get_param_int_or("normalization_value", 255);
    if (normalization_value != 255)
        signal_error_and_exit(INVALID_NORMALIZATION_VALUE);   /* 8-bit tiles only */
    float gain = get_param_float_or("calib_gain_r", 0.050f);   /* uniform, zero offset: the frozen rendering */
    int tile_size = get_param_int_or("tile_size", 224);

    /* The 188-byte config slot cannot carry every key explicitly, so the
       defaults here ARE the flight values (slots 3/6/16 and 5/7/18), and a
       model given without a directory is looked up in /usr/share/models. */
    std::string model_cls = model_path(get_param_string("model_filename_classifier"));
    std::string model_seg = model_path(get_param_string("model_filename_segmentation"));
    const char *model_filename_classifier = model_cls.c_str();
    const char *model_filename_segmentation = model_seg.c_str();
    int class_idx = get_param_int_or("class_index", 9);        /* water */
    int train_res = get_param_int_or("train_res", 56);

    int class_idx_segmentation = get_param_int_or("class_index_segmentation", 0);
    float threshold_percentage = get_param_float_or("threshold_percentage", 0.5f);
    float pixel_threshold = get_param_float_or("pixel_threshold", 0.5f);
    if (!(pixel_threshold > 0.0f && pixel_threshold < 1.0f))
        pixel_threshold = 0.5f;
    int mask_lowpass = get_param_int_or("mask_lowpass", 5);
    if (mask_lowpass < 0 || (mask_lowpass > 0 && mask_lowpass % 2 == 0))
        mask_lowpass = 0;

    J2kParams j2k;
    j2k.ratio = get_param_float_or("compression_ratio", 20.0f);
    j2k.resolutions = get_param_int_or("resolutions", 6);
    j2k.layers = get_param_int_or("quality_layers", 1);
    j2k.resync = get_param_int_or("resync_markers", 1);

    bool use_npu = get_param_int_or("use_npu", 1) != 0 && getenv("DIPP_NO_NPU") == NULL;

    /* ---- models, loaded once ---- */
    Model cls, seg;
    load_model(cls, model_filename_classifier, logger, use_npu);
    load_model(seg, model_filename_segmentation, logger, use_npu);

    const TfLiteTensor *cls_in_t = cls.interpreter->input_tensor(0);
    const int cls_in_h = cls_in_t->dims->data[1], cls_in_w = cls_in_t->dims->data[2];
    const int cls_in_c = cls_in_t->dims->data[3];
    const TfLiteTensor *cls_out_t = cls.interpreter->output_tensor(0);
    const float cls_scale = cls_out_t->params.scale;
    const int cls_zp = cls_out_t->params.zero_point;
    const int n_classes = cls_out_t->dims->data[cls_out_t->dims->size - 1];

    const TfLiteTensor *seg_in_t = seg.interpreter->input_tensor(0);
    const int seg_in_h = seg_in_t->dims->data[1], seg_in_w = seg_in_t->dims->data[2];
    const TfLiteTensor *seg_out_t = seg.interpreter->output_tensor(0);
    const float seg_scale = seg_out_t->params.scale;
    const int seg_zp = seg_out_t->params.zero_point;

    int tiles_seen = 0, tiles_classified_kept = 0, tiles_encoded = 0;

    for (int i = 0; i < num_images; ++i)
    {
        Metadata *input_meta = get_metadata(i);
        int height = input_meta->height, width = input_meta->width;
        unsigned char *raw_data;
        size_t raw_size = get_image_data(i, &raw_data);
        if (raw_size < (size_t)width * height * 2)
            signal_error_and_exit(INVALID_INPUT);

        /* ---- demosaic: as demosaic_module.cpp ---- */
        cv::Mat raw(height, width, CV_16UC1, (uint16_t *)raw_data);
        cv::Mat dem, rot, frame;
        cv::cvtColor(raw, dem, cv::COLOR_BayerRG2BGR);
        cv::rotate(dem, rot, cv::ROTATE_180);
        rot.convertTo(frame, CV_8UC3, gain);          /* x*gain, rounded, clipped: one pass */
        if (frame.empty())
            signal_error_and_exit(OPENCV_ERR);
        free(raw_data);
        const int channels = 3;

        /* ---- tile: as tile_module.cpp (bottom rows past the grid are dropped) ---- */
        int tile_idx = 0;
        for (int y0 = 0; y0 + tile_size <= height; y0 += tile_size)
        {
            for (int x0 = 0; x0 + tile_size <= width; x0 += tile_size, tile_idx++)
            {
                tiles_seen++;
                cv::Mat tile = frame(cv::Rect(x0, y0, tile_size, tile_size)).clone();  /* contiguous */

                /* ---- classification: as tflite_classification.cpp ---- */
                uint8_t *cls_in = cls.interpreter->typed_input_tensor<uint8_t>(0);
                if (train_res > 0 && train_res < cls_in_h)
                {
                    cv::Mat small_, dst(cls_in_h, cls_in_w, CV_8UC(cls_in_c), cls_in);
                    cv::resize(tile, small_, cv::Size(train_res, train_res), 0, 0, cv::INTER_AREA);
                    cv::resize(small_, dst, cv::Size(cls_in_w, cls_in_h), 0, 0, cv::INTER_LINEAR);
                }
                else if (tile.rows != cls_in_h || tile.cols != cls_in_w)
                {
                    cv::Mat dst(cls_in_h, cls_in_w, CV_8UC(cls_in_c), cls_in);
                    cv::resize(tile, dst, cv::Size(cls_in_w, cls_in_h), 0, 0, cv::INTER_LINEAR);
                }
                else
                {
                    memcpy(cls_in, tile.data, (size_t)cls_in_h * cls_in_w * cls_in_c);
                }
                if (cls.interpreter->Invoke() != kTfLiteOk)
                    signal_error_and_exit(INFER_ERR);
                const uint8_t *scores = cls.interpreter->typed_output_tensor<uint8_t>(0);
                int max_cls = -1;
                float max_val = -1.0f;
                for (int j = 0; j < n_classes; ++j)
                {
                    float v = (float)(scores[j] - cls_zp) * cls_scale;
                    if (v > max_val)
                    {
                        max_val = v;
                        max_cls = j;
                    }
                }
                char buffer[128];
                snprintf(buffer, sizeof(buffer), "Tile %d: Top class: %d with score %.4f", tile_idx, max_cls, max_val);
                logger_log(logger, LOG_INFO, buffer);
                if (max_cls != class_idx)
                    continue;                                  /* discarded at classification */
                tiles_classified_kept++;

                /* ---- segmentation: as tflite_binary_segmentation.cpp ---- */
                uint8_t *seg_in = seg.interpreter->typed_input_tensor<uint8_t>(0);
                if (tile.rows != seg_in_h || tile.cols != seg_in_w)
                {
                    cv::Mat dst(seg_in_h, seg_in_w, CV_8UC3, seg_in);
                    cv::resize(tile, dst, cv::Size(seg_in_w, seg_in_h), 0, 0, cv::INTER_AREA);
                }
                else
                {
                    memcpy(seg_in, tile.data, (size_t)seg_in_h * seg_in_w * channels);
                }
                if (seg.interpreter->Invoke() != kTfLiteOk)
                    signal_error_and_exit(INFER_ERR);
                uint8_t *raw_scores = seg.interpreter->typed_output_tensor<uint8_t>(0);
                const int out_h = seg_out_t->dims->data[1], out_w = seg_out_t->dims->data[2];
                cv::Mat mask_small(out_h, out_w, CV_8UC1, raw_scores), mask_full;
                if (out_h != tile_size || out_w != tile_size)
                    cv::resize(mask_small, mask_full, cv::Size(tile_size, tile_size), 0, 0, cv::INTER_LINEAR);
                else
                    mask_full = mask_small;
                const uint8_t *mscores = mask_full.data;

                cv::Mat blurred;
                if (mask_lowpass > 0)
                    cv::GaussianBlur(tile, blurred, cv::Size(mask_lowpass, mask_lowpass), 0);
                const int img_size = tile_size * tile_size * channels;
                int kept_pixels = img_size;
                for (int pix = 0; pix < tile_size * tile_size; pix++)
                {
                    float s = (float)(mscores[pix] - seg_zp) * seg_scale;
                    bool keep = (s < pixel_threshold) ? (class_idx_segmentation == 0)
                                                      : (class_idx_segmentation == 1);
                    if (!keep)
                    {
                        uint8_t *px = tile.data + pix * channels;
                        if (mask_lowpass > 0)
                            memcpy(px, blurred.data + pix * channels, channels);
                        else
                            memset(px, 0, channels);
                        kept_pixels -= channels;
                    }
                }
                snprintf(buffer, sizeof(buffer), "Kept pixels: %d out of %d", kept_pixels, img_size);
                logger_log(logger, LOG_INFO, buffer);
                if (!((float)kept_pixels > threshold_percentage * (float)img_size))
                    continue;                                  /* discarded at segmentation */

                /* ---- encode: as jpeg2000_module.c ---- */
                unsigned char *enc = NULL;
                size_t enc_len = 0;
                if (j2k_encode_tile(tile.data, tile_size, tile_size, channels, &j2k, &enc, &enc_len) != 0)
                    signal_error_and_exit(ENCODE_ERR);

                Metadata new_meta = METADATA__INIT;
                if (clone_metadata(input_meta, &new_meta) != 0)
                    signal_error_and_exit(MALLOC_ERR);
                new_meta.size = enc_len;
                new_meta.width = tile_size;
                new_meta.height = tile_size;
                new_meta.channels = channels;
                new_meta.bits_pixel = 8;
                add_custom_metadata_int(&new_meta, "tile_idx", tile_idx);
                add_custom_metadata_int(&new_meta, "prediction", max_cls);
                add_custom_metadata_string(&new_meta, "enc", "j2k");
                add_custom_metadata_string(&new_meta, "processing", "demosaiced");
                append_result_image(enc, enc_len, &new_meta);
                free(enc);
                tiles_encoded++;
            }
        }
    }

    char summary[160];
    snprintf(summary, sizeof(summary), "Single module finished: %d tiles, %d kept by classification, %d encoded",
             tiles_seen, tiles_classified_kept, tiles_encoded);
    logger_log_print(logger, LOG_INFO, summary);
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
