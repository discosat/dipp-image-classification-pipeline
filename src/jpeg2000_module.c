#include "module.h"
#include "util.h"
#include <openjpeg.h>
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

/* Growable memory sink, so the codestream never touches the filesystem. */
typedef struct
{
    unsigned char *buf;
    size_t len;      /* bytes written */
    size_t cap;
    size_t pos;
} MemSink;

static OPJ_SIZE_T mem_write(void *p_buffer, OPJ_SIZE_T n, void *user)
{
    MemSink *s = (MemSink *)user;
    if (s->pos + n > s->cap)
    {
        size_t want = (s->pos + n) * 2;
        unsigned char *nb = (unsigned char *)realloc(s->buf, want);
        if (nb == NULL)
            return (OPJ_SIZE_T)-1;
        s->buf = nb;
        s->cap = want;
    }
    memcpy(s->buf + s->pos, p_buffer, n);
    s->pos += n;
    if (s->pos > s->len)
        s->len = s->pos;
    return n;
}

static OPJ_OFF_T mem_skip(OPJ_OFF_T n, void *user)
{
    MemSink *s = (MemSink *)user;
    s->pos += n;
    if (s->pos > s->len)
        s->len = s->pos;
    return n;
}

static OPJ_BOOL mem_seek(OPJ_OFF_T n, void *user)
{
    MemSink *s = (MemSink *)user;
    s->pos = (size_t)n;
    return OPJ_TRUE;
}

/* START MODULE IMPLEMENTATION */
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
        size_t size = get_image_data(i, &input_image_data);

        opj_cparameters_t params;
        opj_set_default_encoder_parameters(&params);
        params.tcp_numlayers = layers;
        params.cp_disto_alloc = 1;
        for (int l = 0; l < layers; l++)
        {
            /* Layer l targets a progressively finer ratio, so truncating the
             * codestream after any layer yields a valid lower-quality image. */
            params.tcp_rates[l] = (ratio <= 1.0f)
                                      ? 0.0f      /* 0 means lossless in openjpeg */
                                      : ratio * (float)(layers - l) / (float)layers;
        }
        params.irreversible = (ratio <= 1.0f) ? 0 : 1;   /* 5/3 vs 9/7 wavelet */
        params.numresolution = resolutions;
        if (resync)
            params.csty |= 0x02 | 0x04;                 /* SOP and EPH markers */

        opj_image_cmptparm_t cmptparm[4];
        memset(cmptparm, 0, sizeof(cmptparm));
        int ncomp = channels > 4 ? 4 : channels;
        for (int c = 0; c < ncomp; c++)
        {
            cmptparm[c].prec = 8;
            cmptparm[c].bpp = 8;
            cmptparm[c].sgnd = 0;
            cmptparm[c].dx = 1;
            cmptparm[c].dy = 1;
            cmptparm[c].w = width;
            cmptparm[c].h = height;
        }
        opj_image_t *image = opj_image_create(
            ncomp, cmptparm, ncomp >= 3 ? OPJ_CLRSPC_SRGB : OPJ_CLRSPC_GRAY);
        if (image == NULL)
            signal_error_and_exit(OPJ_IMAGE_CREATE);
        image->x0 = 0;
        image->y0 = 0;
        image->x1 = width;
        image->y1 = height;

        /* Interleaved bytes in, planar int32 components out. */
        for (int p = 0; p < width * height; p++)
            for (int c = 0; c < ncomp; c++)
                image->comps[c].data[p] = input_image_data[p * channels + c];

        opj_codec_t *codec = opj_create_compress(OPJ_CODEC_J2K);
        if (codec == NULL)
            signal_error_and_exit(OPJ_CODEC_CREATE);
        if (!opj_setup_encoder(codec, &params, image))
            signal_error_and_exit(OPJ_SETUP_ENCODER);

        MemSink sink = {0};
        sink.cap = size / 4 + 4096;
        sink.buf = (unsigned char *)malloc(sink.cap);
        if (sink.buf == NULL)
            signal_error_and_exit(MALLOC_ERR);

        opj_stream_t *stream = opj_stream_default_create(OPJ_FALSE); /* writer */
        if (stream == NULL)
            signal_error_and_exit(OPJ_STREAM_CREATE);
        opj_stream_set_user_data(stream, &sink, NULL);
        opj_stream_set_write_function(stream, mem_write);
        opj_stream_set_skip_function(stream, mem_skip);
        opj_stream_set_seek_function(stream, mem_seek);

        if (!opj_start_compress(codec, image, stream))
            signal_error_and_exit(OPJ_START_COMPRESS);
        if (!opj_encode(codec, stream))
            signal_error_and_exit(OPJ_ENCODE);
        if (!opj_end_compress(codec, stream))
            signal_error_and_exit(OPJ_END_COMPRESS);

        int enc_size = (int)sink.len;

        Metadata new_meta = METADATA__INIT;
        if (clone_metadata(input_meta, &new_meta) != 0)
            signal_error_and_exit(MALLOC_ERR);
        new_meta.size = enc_size;
        add_custom_metadata_string(&new_meta, "enc", "j2k");

        append_result_image(sink.buf, enc_size, &new_meta);

        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        opj_image_destroy(image);
        free(sink.buf);
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
