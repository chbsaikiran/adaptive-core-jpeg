/*
 * jcfeatures.c
 *
 * See jcfeatures.h.
 *
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jcfeatures.h"

#define JCFEAT_PI 3.14159265358979323846

/* --- grayscale conversion (matches Pillow's RGB->"L" weights) ---------- */

static void to_grayscale(JSAMPARRAY rows, JDIMENSION width, JDIMENSION height,
                          int components, unsigned char *gray) {
  JDIMENSION x, y;

  for (y = 0; y < height; y++) {
    JSAMPROW row = rows[y];

    for (x = 0; x < width; x++) {
      if (components == 1) {
        gray[(size_t)y * width + x] = row[x];
      } else {
        int r = row[x * components + 0];
        int g = row[x * components + 1];
        int b = row[x * components + 2];
        int l = (r * 299 + g * 587 + b * 114 + 500) / 1000;

        if (l > 255) l = 255;
        if (l < 0) l = 0;
        gray[(size_t)y * width + x] = (unsigned char)l;
      }
    }
  }
}

/* --- entropy ------------------------------------------------------------ */

static double shannon_entropy(const unsigned char *gray, size_t n) {
  unsigned long hist[256];
  double entropy = 0.0;
  size_t i;
  int b;

  memset(hist, 0, sizeof(hist));
  for (i = 0; i < n; i++)
    hist[gray[i]]++;

  for (b = 0; b < 256; b++) {
    double p;

    if (hist[b] == 0)
      continue;
    p = (double)hist[b] / (double)n;
    entropy -= p * (log(p) / log(2.0));
  }
  return entropy;
}

/* --- Sobel edge density/variance ---------------------------------------- */

static JDIMENSION clamp_coord(long v, JDIMENSION limit) {
  if (v < 0)
    return 0;
  if (v >= (long)limit)
    return limit - 1;
  return (JDIMENSION)v;
}

static void sobel_stats(const unsigned char *gray, JDIMENSION width,
                         JDIMENSION height, double *density_out,
                         double *variance_out) {
  static const int kx[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
  static const int ky[3][3] = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };
  double sum = 0.0, sum_sq = 0.0;
  size_t n = (size_t)width * height;
  JDIMENSION x, y;

  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      int gx = 0, gy = 0;
      int dy, dx;
      double mag;

      for (dy = -1; dy <= 1; dy++) {
        JDIMENSION sy = clamp_coord((long)y + dy, height);

        for (dx = -1; dx <= 1; dx++) {
          JDIMENSION sx = clamp_coord((long)x + dx, width);
          int v = gray[(size_t)sy * width + sx];

          gx += kx[dy + 1][dx + 1] * v;
          gy += ky[dy + 1][dx + 1] * v;
        }
      }

      mag = sqrt((double)gx * (double)gx + (double)gy * (double)gy);
      sum += mag;
      sum_sq += mag * mag;
    }
  }

  {
    double mean = sum / (double)n;
    double mean_sq = sum_sq / (double)n;

    *density_out = mean;
    *variance_out = mean_sq - mean * mean;
  }
}

/* --- per-channel variance + mean HSV saturation -------------------------
 * Saturation uses PIL's integer formula: s = (max==0) ? 0 : 255*(max-min)/max
 * (colorsys/Pillow's RGB->HSV convention, 0-255 scale). */

static void channel_stats(JSAMPARRAY rows, JDIMENSION width,
                           JDIMENSION height, int components, double *r_var,
                           double *g_var, double *b_var, double *mean_sat) {
  double r_sum = 0, g_sum = 0, b_sum = 0;
  double r_sq = 0, g_sq = 0, b_sq = 0, sat_sum = 0;
  size_t n = (size_t)width * height;
  JDIMENSION x, y;

  for (y = 0; y < height; y++) {
    JSAMPROW row = rows[y];

    for (x = 0; x < width; x++) {
      int r, g, b, maxc, minc, s;

      if (components == 1) {
        r = g = b = row[x];
      } else {
        r = row[x * components + 0];
        g = row[x * components + 1];
        b = row[x * components + 2];
      }

      r_sum += r;
      g_sum += g;
      b_sum += b;
      r_sq += (double)r * r;
      g_sq += (double)g * g;
      b_sq += (double)b * b;

      maxc = r;
      if (g > maxc) maxc = g;
      if (b > maxc) maxc = b;
      minc = r;
      if (g < minc) minc = g;
      if (b < minc) minc = b;
      s = (maxc == 0) ? 0 : (255 * (maxc - minc)) / maxc;
      sat_sum += s;
    }
  }

  {
    double rm = r_sum / (double)n, gm = g_sum / (double)n,
           bm = b_sum / (double)n;

    *r_var = r_sq / (double)n - rm * rm;
    *g_var = g_sq / (double)n - gm * gm;
    *b_var = b_sq / (double)n - bm * bm;
    *mean_sat = sat_sum / (double)n;
  }
}

/* --- 8x8 block DCT AC energy ---------------------------------------------
 * Orthonormal DCT-II (matches scipy.fftpack.dct(..., norm="ortho")),
 * applied separably (order doesn't affect the result: 2D separable DCT is
 * Y = C X C^T regardless of which axis is transformed first). */

static double dct_cos[8][8];
static int dct_cos_ready = 0;

static void init_dct_cos(void) {
  int k, n;

  if (dct_cos_ready)
    return;
  for (k = 0; k < 8; k++)
    for (n = 0; n < 8; n++)
      dct_cos[k][n] = cos((JCFEAT_PI / 8.0) * ((double)n + 0.5) * (double)k);
  dct_cos_ready = 1;
}

static void dct8(const double in[8], double out[8]) {
  int k, n;

  for (k = 0; k < 8; k++) {
    double sum = 0.0;

    for (n = 0; n < 8; n++)
      sum += in[n] * dct_cos[k][n];
    out[k] = sum * ((k == 0) ? sqrt(1.0 / 8.0) : sqrt(2.0 / 8.0));
  }
}

static double dct_ac_energy_of_block(const unsigned char *gray,
                                      JDIMENSION stride, JDIMENSION bx,
                                      JDIMENSION by) {
  double block[8][8], tmp[8][8], coeffs[8][8];
  int r, c;
  double ac_sum;

  for (r = 0; r < 8; r++)
    for (c = 0; c < 8; c++)
      block[r][c] = (double)gray[(size_t)(by + r) * stride + (bx + c)];

  for (r = 0; r < 8; r++)
    dct8(block[r], tmp[r]);

  for (c = 0; c < 8; c++) {
    double col_in[8], col_out[8];

    for (r = 0; r < 8; r++)
      col_in[r] = tmp[r][c];
    dct8(col_in, col_out);
    for (r = 0; r < 8; r++)
      coeffs[r][c] = col_out[r];
  }

  ac_sum = 0.0;
  for (r = 0; r < 8; r++)
    for (c = 0; c < 8; c++)
      ac_sum += fabs(coeffs[r][c]);
  ac_sum -= fabs(coeffs[0][0]);

  return ac_sum;
}

static double mean_dct_ac_energy(const unsigned char *gray, JDIMENSION width,
                                  JDIMENSION height) {
  JDIMENSION h8 = (height / 8) * 8, w8 = (width / 8) * 8;
  double total = 0.0;
  unsigned long blocks = 0;
  JDIMENSION bx, by;

  for (by = 0; by < h8; by += 8) {
    for (bx = 0; bx < w8; bx += 8) {
      total += dct_ac_energy_of_block(gray, width, bx, by);
      blocks++;
    }
  }

  return (blocks > 0) ? (total / (double)blocks) : 0.0;
}

/* --- public entry point -------------------------------------------------- */

EXTERN(void) jcfeat_extract(JSAMPARRAY rows, JDIMENSION width,
                             JDIMENSION height, int components,
                             unsigned long file_size_bytes,
                             double features_out[JCFEAT_NUM_FEATURES]) {
  unsigned char *gray;
  double megapixels = ((double)width * (double)height) / 1000000.0;
  double r_var, g_var, b_var, mean_sat, density, variance;

  gray = (unsigned char *)malloc((size_t)width * height);
  if (!gray) {
    int i;

    for (i = 0; i < JCFEAT_NUM_FEATURES; i++)
      features_out[i] = 0.0;
    return;
  }

  to_grayscale(rows, width, height, components, gray);
  init_dct_cos();

  features_out[0] = (double)file_size_bytes / megapixels;
  features_out[1] = shannon_entropy(gray, (size_t)width * height);

  sobel_stats(gray, width, height, &density, &variance);
  features_out[2] = density;
  features_out[3] = variance;

  channel_stats(rows, width, height, components, &r_var, &g_var, &b_var,
                &mean_sat);
  features_out[4] = r_var;
  features_out[5] = g_var;
  features_out[6] = b_var;
  features_out[7] = mean_sat;

  features_out[8] = mean_dct_ac_energy(gray, width, height);

  free(gray);
}
