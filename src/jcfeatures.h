/*
 * jcfeatures.h
 *
 * Pre-encode image-content feature extraction for the complexity
 * classifier (complexity_model.c / jcmodel.h).  This is a C port of
 * scripts/ml/extract_features.py, computing the same 9 features from
 * already-decoded pixel rows so a caller can classify an image before
 * choosing how many cores to encode it with -- see jcadaptive.c.
 *
 * This is a best-effort numerical port, not a bit-exact one: the Sobel
 * edge features use simple clamped-edge boundary handling (scipy's default
 * is a "reflect" boundary), and floating-point rounding differs from
 * numpy/scipy's implementation in the usual ways a from-scratch port does.
 * Both sides were cross-checked on real images (see scripts/ml/README.md)
 * and track closely -- close enough for a tree-based classifier's fixed
 * thresholds -- but this is not guaranteed to reproduce the Python
 * pipeline's feature values to the last bit.
 *
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 */

#ifndef JCFEATURES_H
#define JCFEATURES_H

#include "jpeglib.h"

#define JCFEAT_NUM_FEATURES 9

/* The downsample factor the committed complexity_model.c was trained at --
 * callers feeding features to score() must extract them with
 * jcfeat_extract_downsampled() at this factor.  Keep in sync with
 * scripts/ml/extract_features.py's DEFAULT_DOWNSAMPLE. */
#define JCFEAT_MODEL_DOWNSAMPLE 4

/*
 * Feature order (must match scripts/ml/extract_features.py's FEATURE_NAMES,
 * and thus the input order complexity_model.c's score() expects):
 *   [0] bytes_per_mpx    on-disk source file size per megapixel
 *   [1] entropy          grayscale pixel histogram Shannon entropy
 *   [2] edge_density     mean Sobel gradient magnitude
 *   [3] edge_variance    variance of Sobel gradient magnitude
 *   [4] r_variance       red channel pixel variance
 *   [5] g_variance       green channel pixel variance
 *   [6] b_variance       blue channel pixel variance
 *   [7] mean_saturation  mean HSV saturation (0-255 scale, PIL convention)
 *   [8] dct_ac_energy    mean |AC coefficient| sum per 8x8 block
 *
 * `rows` must hold `height` scanlines of `components` samples/pixel
 * (1 = grayscale, 3 = RGB -- the same contract jpar_encode_strip() uses).
 * `file_size_bytes` is the on-disk size of the source file being encoded
 * (the same proxy extract_features.py uses, computed there from the
 * source PNG's file size).
 */
EXTERN(void) jcfeat_extract(JSAMPARRAY rows, JDIMENSION width,
                             JDIMENSION height, int components,
                             unsigned long file_size_bytes,
                             double features_out[JCFEAT_NUM_FEATURES]);

/*
 * Same features, computed from a decimated copy of the image: only every
 * `downsample`-th pixel of every `downsample`-th row is looked at (no
 * averaging), cutting the cost by roughly downsample^2.  downsample <= 1
 * is identical to jcfeat_extract().  bytes_per_mpx still refers to the
 * source image's full megapixel count.
 *
 * Decimation shifts the texture features (neighbouring samples are further
 * apart, so edge_density/edge_variance/dct_ac_energy read higher than at
 * full resolution), so a classifier must be trained on features extracted
 * at the same factor -- scripts/ml/extract_features.py --downsample N
 * mirrors this exactly.
 */
EXTERN(void) jcfeat_extract_downsampled(JSAMPARRAY rows, JDIMENSION width,
                                         JDIMENSION height, int components,
                                         unsigned long file_size_bytes,
                                         int downsample,
                                         double features_out[JCFEAT_NUM_FEATURES]);

#endif /* JCFEATURES_H */
