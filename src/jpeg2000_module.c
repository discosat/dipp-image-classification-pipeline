#include "module.h"
#include "util.h"
#include <openjpeg.h>
#include "j2k_encode.h"
#include <stdbool.h>
#include <string.h>

/*
 * JPEG 2000 encoder, offered as an alternative to the JPEG XL module.
 *
 * Why it exists. Two properties measured on real device tiles motivate it:
 *
 *   Compression. At matched quality (~48 dB) JPEG 2000 produced 4472 B against
 *   JPEG XL's 6123 B -- 27% smaller at the same encode time. So switching costs
 *   nothing in efficiency on this content.
 *
 *   Resilience. Progressive decoding and error resilience are different
 *   properties, and only the second survives a lossy link. JPEG XL's
 *   progressive mode handles TRUNCATION: a pass that ends early still renders
 *   what arrived. It does not handle ERASURE -- a frame lost mid-file leaves a
 *   hole, and everything after it is undecodable. JPEG 2000 carries
 *   resynchronisation markers (SOP before each packet, EPH after each packet
 *   header) so a decoder can pick up after damage, and its packets are
 *   independent. With CCSDS-style framing the radio does FEC and a per-frame
 *   CRC, so corrupted frames are DROPPED rather than delivered corrupt -- the
 *   application sees erasures, which is precisely the case JPEG XL cannot
 *   recover from.
 *
 * The mask fill interacts with the codec choice. Blacking out cloud maximises
 * the discontinuity at the mask boundary, and wavelets spend their coefficients
 * on exactly that: black fill made JPEG 2000 +130% LARGER than not masking at
 * all, worse than JPEG XL's +116%. With the 5x5 low-pass fill the segmentation
 * module now applies, JPEG 2000 gains -28% where JPEG XL gains only -5%. So
 * this codec is the wrong choice under the old fill and the better one under
 * the new.
 *
 * openjpeg 2.4.0 is already present on the board and in the SDK sysroot, so
 * this needs no new runtime dependency -- swapping the encoder is one module
 * upload.
 */

enum ERROR_CODE
{
    MALLOC_ERR = 1,
    OPJ_IMAGE_CREATE = 2,
    OPJ_CODEC_CREATE = 3,
    OPJ_SETUP_ENCODER = 4,
    OPJ_STREAM_CREATE = 5,
    OPJ_START_COMPRESS = 6,
    OPJ_ENCODE = 7,
    OPJ_END_COMPRESS = 8,
    META_BITSPIXEL = 9,
};

void module()
{
    int num_images = get_input_num_images();

    /* Compression ratio: 1 means lossless (reversible 5/3 wavelet), anything
     * greater selects the irreversible 9/7 wavelet at that target ratio. This
     * is the knob that trades quality for size, the counterpart of JPEG XL's
     * distance -- and the lesson from the JPEG XL module is that an effort
     * ladder must move this, not just how hard the encoder searches. */
    float ratio = get_param_float_or("compression_ratio", 20.0f);

    /* Wavelet decomposition levels. More levels help large images and cost
     * time; 6 is openjpeg's default and sensible for a 224x224 tile. */
    int resolutions = get_param_int_or("resolutions", 6);

    /* Quality layers. Multiple layers make the codestream SNR-scalable: a
     * truncated downlink still decodes at lower quality. */
    int layers = get_param_int_or("quality_layers", 1);

    /* Resynchronisation markers. Off by default because they cost a little
     * size, on when the link is expected to drop frames. */
    int resync = get_param_int_or("resync_markers", 0);

    for (int i = 0; i < num_images; ++i)
    {
        Metadata *input_meta = get_metadata(i);
        int width = input_meta->width;
        int height = input_meta->height;
        int channels = input_meta->channels;
        if (input_meta->bits_pixel != 8)
            signal_error_and_exit(META_BITSPIXEL);

        unsigned char *input_image_data;
        (void)get_image_data(i, &input_image_data);

        J2kParams p = {ratio, resolutions, layers, resync};
        unsigned char *enc = NULL;
        size_t enc_len = 0;
        int rc = j2k_encode_tile(input_image_data, width, height, channels, &p, &enc, &enc_len);
        if (rc != 0)
            signal_error_and_exit(rc);       /* J2K_ERR_* map onto this module's codes */
        int enc_size = (int)enc_len;

        Metadata new_meta = METADATA__INIT;
        if (clone_metadata(input_meta, &new_meta) != 0)
            signal_error_and_exit(MALLOC_ERR);
        new_meta.size = enc_size;
        add_custom_metadata_string(&new_meta, "enc", "j2k");

        append_result_image(enc, enc_size, &new_meta);

        free(enc);
        free(input_image_data);
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
