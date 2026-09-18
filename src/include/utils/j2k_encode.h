#ifndef J2K_ENCODE_H
#define J2K_ENCODE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* JPEG 2000 encode of one interleaved 8-bit tile into a malloc'd codestream.
 * Shared by the jpeg2000 module and the monolith (single_module), so the two
 * arms of the decomposed-vs-monolithic comparison run the same encoder with
 * the same parameters -- the monolith used to carry a JPEG XL encoder of its
 * own with a hardcoded effort. */
typedef struct
{
    float ratio;       /* 1 = lossless (5/3), >1 = irreversible 9/7 at this ratio */
    int resolutions;   /* wavelet levels, 6 = openjpeg default */
    int layers;        /* quality layers: SNR-scalable codestream */
    int resync;        /* SOP + EPH markers */
} J2kParams;

/* Returns 0 on success with *out (malloc'd) and *out_len set; otherwise one
   of J2K_ERR_* below, and nothing is allocated. */
enum
{
    J2K_ERR_MALLOC = 1,
    J2K_ERR_IMAGE_CREATE = 2,
    J2K_ERR_CODEC_CREATE = 3,
    J2K_ERR_SETUP_ENCODER = 4,
    J2K_ERR_STREAM_CREATE = 5,
    J2K_ERR_START_COMPRESS = 6,
    J2K_ERR_ENCODE = 7,
    J2K_ERR_END_COMPRESS = 8,
};
int j2k_encode_tile(const unsigned char *pixels, int width, int height, int channels,
                    const J2kParams *p, unsigned char **out, size_t *out_len);

#ifdef __cplusplus
}
#endif
#endif
