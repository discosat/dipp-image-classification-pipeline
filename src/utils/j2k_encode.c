#include <stdlib.h>
#include <string.h>
#include <openjpeg.h>
#include "j2k_encode.h"

/* Growable memory sink, so the codestream never touches the filesystem. */
typedef struct
{
    unsigned char *buf;
    size_t len;      /* bytes written */
    size_t cap;
    size_t pos;
} MemSink;

/* Grow the sink so that `need` bytes are addressable from offset 0.
 *
 * openjpeg reserves space with skip/seek and backfills packet-length markers
 * later, so the position can legitimately move beyond what has been written.
 * Growing only inside mem_write would leave mem_skip free to advance pos and
 * len past cap without allocating, and the module then reads sink.buf[0..len)
 * when handing the codestream to append_result_image. Reserving here closes
 * that off.
 *
 * This is defensive, not a fix for an observed failure -- and the record needs
 * to be straight, because it was briefly written up as one. The
 * MODULE_EXIT_CRASH (600201) that prompted it was not in this module at all:
 * err_current_module is i+1, so 600201 is pipeline 2 module index 0, which is
 * libdemosaic. The board had leaked 61 SysV segments holding 1.59 GB of its
 * 1.95 GB (DIPP's STORAGE_MEM path detaches without IPC_RMID), so the first
 * module to claim a full-resolution frame died, and every encoder config failed
 * identically whatever its parameters. All 16 grid configs encode 99 tiles
 * clean under AddressSanitizer against openjpeg 2.4 on the host, at every
 * quality-layer count.
 */
static int mem_reserve(MemSink *s, size_t need)
{
    if (need <= s->cap)
        return 1;
    size_t want = need * 2;
    unsigned char *nb = (unsigned char *)realloc(s->buf, want);
    if (nb == NULL)
        return 0;
    /* Zero the new region: skipped bytes are read back when markers are
     * backfilled, and uninitialised memory would make the codestream
     * nondeterministic. */
    memset(nb + s->cap, 0, want - s->cap);
    s->buf = nb;
    s->cap = want;
    return 1;
}

static OPJ_SIZE_T mem_write(void *p_buffer, OPJ_SIZE_T n, void *user)
{
    MemSink *s = (MemSink *)user;
    if (!mem_reserve(s, s->pos + n))
        return (OPJ_SIZE_T)-1;
    memcpy(s->buf + s->pos, p_buffer, n);
    s->pos += n;
    if (s->pos > s->len)
        s->len = s->pos;
    return n;
}

static OPJ_OFF_T mem_skip(OPJ_OFF_T n, void *user)
{
    MemSink *s = (MemSink *)user;
    /* n is signed and openjpeg rewinds with negative skips. Computing the new
     * position in size_t would underflow to an astronomical value in that case
     * and make mem_reserve attempt a gigantic realloc, so do the arithmetic
     * signed. */
    OPJ_OFF_T newpos = (OPJ_OFF_T)s->pos + n;
    if (newpos < 0)
        return (OPJ_OFF_T)-1;
    if (!mem_reserve(s, (size_t)newpos))
        return (OPJ_OFF_T)-1;
    s->pos = (size_t)newpos;
    if (s->pos > s->len)
        s->len = s->pos;
    return n;
}

static OPJ_BOOL mem_seek(OPJ_OFF_T n, void *user)
{
    MemSink *s = (MemSink *)user;
    if (n < 0)
        return OPJ_FALSE;
    if (!mem_reserve(s, (size_t)n))
        return OPJ_FALSE;
    s->pos = (size_t)n;
    return OPJ_TRUE;
}

/* START MODULE IMPLEMENTATION */

int j2k_encode_tile(const unsigned char *pixels, int width, int height, int channels,
                    const J2kParams *p, unsigned char **out, size_t *out_len)
{
    opj_cparameters_t params;
    opj_set_default_encoder_parameters(&params);
    params.tcp_numlayers = p->layers;
    params.cp_disto_alloc = 1;
    for (int l = 0; l < p->layers; l++)
    {
        /* Layer l targets a progressively finer ratio, so truncating the
         * codestream after any layer yields a valid lower-quality image. */
        params.tcp_rates[l] = (p->ratio <= 1.0f)
                                  ? 0.0f      /* 0 means lossless in openjpeg */
                                  : p->ratio * (float)(p->layers - l) / (float)p->layers;
    }
    params.irreversible = (p->ratio <= 1.0f) ? 0 : 1;   /* 5/3 vs 9/7 wavelet */
    params.numresolution = p->resolutions;
    if (p->resync)
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
        return J2K_ERR_IMAGE_CREATE;
    image->x0 = 0;
    image->y0 = 0;
    image->x1 = width;
    image->y1 = height;

    /* Interleaved bytes in, planar int32 components out. */
    for (int px = 0; px < width * height; px++)
        for (int c = 0; c < ncomp; c++)
            image->comps[c].data[px] = pixels[px * channels + c];

    int rc = 0;
    opj_codec_t *codec = opj_create_compress(OPJ_CODEC_J2K);
    if (codec == NULL)
    {
        opj_image_destroy(image);
        return J2K_ERR_CODEC_CREATE;
    }
    if (!opj_setup_encoder(codec, &params, image))
    {
        opj_destroy_codec(codec);
        opj_image_destroy(image);
        return J2K_ERR_SETUP_ENCODER;
    }

    MemSink sink = {0};
    sink.cap = (size_t)width * height * channels / 4 + 4096;
    sink.buf = (unsigned char *)malloc(sink.cap);
    if (sink.buf == NULL)
    {
        opj_destroy_codec(codec);
        opj_image_destroy(image);
        return J2K_ERR_MALLOC;
    }

    opj_stream_t *stream = opj_stream_default_create(OPJ_FALSE); /* writer */
    if (stream == NULL)
        rc = J2K_ERR_STREAM_CREATE;
    else
    {
        opj_stream_set_user_data(stream, &sink, NULL);
        opj_stream_set_write_function(stream, mem_write);
        opj_stream_set_skip_function(stream, mem_skip);
        opj_stream_set_seek_function(stream, mem_seek);
        if (!opj_start_compress(codec, image, stream))
            rc = J2K_ERR_START_COMPRESS;
        else if (!opj_encode(codec, stream))
            rc = J2K_ERR_ENCODE;
        else if (!opj_end_compress(codec, stream))
            rc = J2K_ERR_END_COMPRESS;
        opj_stream_destroy(stream);
    }
    opj_destroy_codec(codec);
    opj_image_destroy(image);
    if (rc)
    {
        free(sink.buf);
        return rc;
    }
    *out = sink.buf;
    *out_len = sink.len;
    return 0;
}
