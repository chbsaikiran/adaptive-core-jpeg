/*
 * jcimageload.h
 *
 * Shared "load one image file into memory as JSAMPLE rows" helper, used by
 * both jcparallelbench.c (benchmark over a directory) and jcadaptive.c
 * (single-image adaptive-core encode).  Originally part of
 * jcparallelbench.c; factored out here so both tools share one
 * implementation instead of duplicating it.
 *
 * Supported formats: JPEG (8-bit only), BMP, PNG, PPM/PGM -- dispatched by
 * file extension.  BMP/PNG/PPM/PGM are decoded via cjpeg's own existing
 * reader modules (jinit_read_bmp()/jinit_read_png()/jinit_read_ppm(),
 * declared in cdjpeg.h) rather than reimplementing those formats.
 *
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 */

#ifndef JCIMAGELOAD_H
#define JCIMAGELOAD_H

#include "jpeglib.h"

/* One fully-decoded image, ready to hand to jpar_encode_strip(s)_*(). */
typedef struct {
  JSAMPLE *data;  /* contiguous height*row_stride bytes */
  JSAMPROW *rows; /* height row pointers into data */
  JDIMENSION width, height;
  int components;
  J_COLOR_SPACE color_space;
} jcil_loaded_image;

/* Loads `path` (dispatching on its filename extension: .jpg/.jpeg, .bmp,
 * .png, .ppm/.pgm) into *img.  Returns FALSE (silently, no message) if the
 * extension isn't one of those, and FALSE (with a message on stderr) if
 * the file can't be decoded or isn't 8-bit. */
EXTERN(boolean) jcil_load_image(const char *path, jcil_loaded_image *img);

EXTERN(void) jcil_free_loaded_image(jcil_loaded_image *img);

/* Returns the max component v_samp_factor libjpeg picks by default for the
 * given input color space -- needed by callers to compute MCU row height
 * for jpar_compute_strip_bounds()/jpar_encode_strips_parallel(). */
EXTERN(int) jcil_get_max_v_samp_factor(J_COLOR_SPACE in_color_space,
                                        int input_components);

#endif /* JCIMAGELOAD_H */
