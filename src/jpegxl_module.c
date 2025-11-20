#include "module.h"
#include "util.h"
#include <jxl/encode.h>
#include <stdbool.h>

/* Define custom error codes */
enum ERROR_CODE
{
    MALLOC_ERR = 1,
    JXL_ENC_ENCODER_CREATE = 2,
    JXL_ENC_SET_OPTIONS = 3,
    JXL_ENC_SET_LOSSLESS = 4,
    JXL_ENC_SET_DISTANCE = 5,
    JXL_ENC_SET_INFO = 6,
    JXL_ENC_ADD_IMAGE = 7,
    JXL_ENC_PROCESS = 8,
    META_BITSPIXEL = 9,
};

/* START MODULE IMPLEMENTATION */
void module()
{
    /* Get number of images in input batch */
    int num_images = get_input_num_images();

    int effort = get_param_int("effort_level");
    // int resampling = get_param_int("resampling");
    // float distance = get_param_float("distance");
    // int lossless = distance == 0;

    /* Example code for iterating a pixel value at a time */
    for (int i = 0; i < num_images; ++i)
    {
        Metadata *input_meta = get_metadata(i);
        int size = input_meta->size;
        int height = input_meta->height;
        int width = input_meta->width;
        int channels = input_meta->channels;
        int bits_pixel = input_meta->bits_pixel;

        unsigned char *input_image_data;
        get_image_data(i, &input_image_data);

        JxlEncoder *encoder = JxlEncoderCreate(NULL); // initialize encoder

        if (encoder == NULL)
            signal_error_and_exit(JXL_ENC_ENCODER_CREATE);

        JxlEncoderFrameSettings *settings = JxlEncoderFrameSettingsCreate(encoder, NULL); // creates settings object for configuring how frames are compressed

        if (JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, effort)) // sets compression effort - from configuration
            signal_error_and_exit(JXL_ENC_SET_OPTIONS);

        // if (JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_RESAMPLING, resampling)) // sets sampling strategy - from configuration
        //     signal_error_and_exit(JXL_ENC_SET_OPTIONS);

        // if (lossless && JxlEncoderSetFrameLossless(settings, JXL_TRUE)) // sets lossless
        //     signal_error_and_exit(JXL_ENC_SET_LOSSLESS);

        // if (!lossless && JxlEncoderSetFrameDistance(settings, distance)) // sets lossy based on distance - from configuration
        //     signal_error_and_exit(JXL_ENC_SET_DISTANCE);

        JxlBasicInfo basic_info; // image metadata
        JxlEncoderInitBasicInfo(&basic_info);
        // if (lossless)
        //     basic_info.uses_original_profile = JXL_TRUE;
        basic_info.xsize = width;
        basic_info.ysize = height;
        basic_info.num_color_channels = channels > 3 ? 3 : channels;
        basic_info.num_extra_channels = channels - basic_info.num_color_channels;
        basic_info.bits_per_sample = 8;
        basic_info.alpha_bits = basic_info.num_extra_channels > 0 ? 8 : 0;

        int jxl_type = JXL_TYPE_UINT8;
        if (bits_pixel == 16)
        {
            jxl_type = JXL_TYPE_UINT16;
        }
        else if (bits_pixel == 8)
        {
            jxl_type = JXL_TYPE_UINT8;
        }
        else
        {
            signal_error_and_exit(META_BITSPIXEL);
        }

        JxlPixelFormat format = {channels, jxl_type, JXL_NATIVE_ENDIAN, 0};

        if (JxlEncoderSetBasicInfo(encoder, &basic_info))
            signal_error_and_exit(JXL_ENC_SET_INFO);

        if (JxlEncoderAddImageFrame(settings, &format, input_image_data, size)) // feeds raw pixel data to encoder for compression
            signal_error_and_exit(JXL_ENC_ADD_IMAGE);

        JxlEncoderCloseInput(encoder); // signalizes this is the end of the input

        size_t output_buffer_size = size;
        size_t out_buf_remain = output_buffer_size;
        uint8_t *output_buffer = (uint8_t *)malloc(output_buffer_size);
        uint8_t *out_buf_next = output_buffer;
        if (JxlEncoderProcessOutput(encoder, &out_buf_next, &out_buf_remain))
            signal_error_and_exit(JXL_ENC_PROCESS);

        int enc_size = output_buffer_size - out_buf_remain; // calculate compressed size

        Metadata new_meta = METADATA__INIT;
        if (clone_metadata(input_meta, &new_meta) != 0)
        {
            signal_error_and_exit(MALLOC_ERR);
        }

        /* Create output image metadata */
        new_meta.size = enc_size;

        /* Add custom metadata for demosaicing info */
        add_custom_metadata_string(&new_meta, "enc", "jxl");

        /* Append the processed image to the result batch */
        append_result_image(output_buffer, enc_size, &new_meta);

        /* Remember to free any allocated memory */
        free(input_image_data);
        free(output_buffer);
        JxlEncoderDestroy(encoder);
    }
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