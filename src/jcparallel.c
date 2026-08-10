/*
 * jcparallel.c
 *
 * Experimental strip-based multi-core JPEG compression helpers.
 * See jcparallel.h for the design rationale.
 *
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 */

#include "jinclude.h"
#include "jpeglib.h"
#include "jcparallel.h"
#include <setjmp.h>

#ifdef JPAR_USE_OPENMP
#include <omp.h>
#endif


GLOBAL(int)
jpar_compute_strip_bounds(JDIMENSION image_height, int max_v_samp_factor,
                           int num_threads, jpar_strip *strips)
{
  JDIMENSION mcu_row_height, total_mcu_rows, mcu_rows_per_strip, rows_done;
  int i, n;

  if (num_threads < 1)
    num_threads = 1;
  if (max_v_samp_factor < 1)
    max_v_samp_factor = 1;

  mcu_row_height = (JDIMENSION)(DCTSIZE * max_v_samp_factor);
  total_mcu_rows = (image_height + mcu_row_height - 1) / mcu_row_height;

  /* Never hand out more strips than we have whole MCU rows to give them --
   * every strip needs at least one MCU row of real data so its Huffman
   * encoder has something to reset its DC predictors on. */
  n = num_threads;
  if (total_mcu_rows == 0)
    n = 1;
  else if ((JDIMENSION)n > total_mcu_rows)
    n = (int)total_mcu_rows;

  mcu_rows_per_strip = total_mcu_rows / (JDIMENSION)n;
  if (mcu_rows_per_strip < 1)
    mcu_rows_per_strip = 1;

  rows_done = 0;
  for (i = 0; i < n; i++) {
    JDIMENSION start_row = rows_done;
    JDIMENSION end_row;

    if (i == n - 1) {
      /* Last strip absorbs every remaining scanline, including whatever
       * partial MCU row hangs off the bottom of the image -- this is
       * exactly where a single-threaded encode's own bottom-edge padding
       * would kick in, so behavior matches. */
      end_row = image_height;
    } else {
      end_row = start_row + mcu_rows_per_strip * mcu_row_height;
      if (end_row > image_height)
        end_row = image_height;
    }

    strips[i].start_row = start_row;
    strips[i].num_rows = end_row - start_row;
    rows_done = end_row;
  }

  return n;
}


/* Private error handler so a failure in one strip's encode can't take down
 * the whole process (or, worse, longjmp across an OpenMP thread boundary).
 * Each call to jpar_encode_strip() sets one of these up on its own stack. */

struct jpar_error_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf setjmp_buffer;
};

METHODDEF(void)
jpar_error_exit(j_common_ptr cinfo)
{
  struct jpar_error_mgr *myerr = (struct jpar_error_mgr *)cinfo->err;

  longjmp(myerr->setjmp_buffer, 1);
}


GLOBAL(boolean)
jpar_encode_strip(JSAMPARRAY image_rows, JDIMENSION image_width,
                   const jpar_strip *strip, int input_components,
                   J_COLOR_SPACE in_color_space, int quality,
                   unsigned char **jpeg_buf, unsigned long *jpeg_size)
{
  struct jpeg_compress_struct cinfo;
  struct jpar_error_mgr jerr;
  JSAMPROW *strip_rows = image_rows + strip->start_row;

  *jpeg_buf = NULL;
  *jpeg_size = 0;

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = jpar_error_exit;

  if (setjmp(jerr.setjmp_buffer)) {
    /* libjpeg longjmp'd here after a fatal error. */
    jpeg_destroy_compress(&cinfo);
    if (*jpeg_buf != NULL) {
      free(*jpeg_buf);
      *jpeg_buf = NULL;
    }
    *jpeg_size = 0;
    return FALSE;
  }

  jpeg_create_compress(&cinfo);
  jpeg_mem_dest(&cinfo, jpeg_buf, jpeg_size);

  cinfo.image_width = image_width;
  cinfo.image_height = strip->num_rows;
  cinfo.input_components = input_components;
  cinfo.in_color_space = in_color_space;

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE);

  /* Every strip must use the same, standard Huffman tables so that the
   * per-strip bitstreams can later be spliced together (via restart
   * markers) without each strip carrying its own DHT segments. */
  cinfo.optimize_coding = FALSE;

  /* Note: libjpeg-turbo's *encode*-side chroma downsampler (jcsample.c) is
   * a plain box filter over each h_samp_factor x v_samp_factor group by
   * default -- it reads only the group's own rows, with no dependency on
   * neighboring MCU rows, so it has no cross-strip-boundary effect here.
   * (The historical do_fancy_downsampling compress flag exists for
   * libjpeg7+ API compatibility but isn't actually read anywhere in this
   * codebase -- INPUT_SMOOTHING_SUPPORTED's smooth downsampler, which
   * *does* need context rows, is instead gated on cinfo.smoothing_factor,
   * which defaults to 0/off.)  If a caller of this function ever sets
   * smoothing_factor, that assumption breaks and this would need
   * revisiting. */

  jpeg_start_compress(&cinfo, TRUE);

  while (cinfo.next_scanline < cinfo.image_height) {
    JSAMPROW row_pointer[1];

    row_pointer[0] = strip_rows[cinfo.next_scanline];
    (void)jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);

  return TRUE;
}


GLOBAL(boolean)
jpar_encode_strips_parallel(JSAMPARRAY image_rows, JDIMENSION image_width,
                             JDIMENSION image_height, int max_v_samp_factor,
                             int input_components,
                             J_COLOR_SPACE in_color_space, int quality,
                             int num_threads, jpar_strip *strips,
                             unsigned char **jpeg_bufs,
                             unsigned long *jpeg_sizes, int *num_strips_out)
{
  int n, i;
  boolean all_ok = TRUE;

  n = jpar_compute_strip_bounds(image_height, max_v_samp_factor, num_threads,
                                 strips);
  *num_strips_out = n;

  for (i = 0; i < n; i++) {
    jpeg_bufs[i] = NULL;
    jpeg_sizes[i] = 0;
  }

  /* Each loop iteration reads only strips[i] and writes only
   * jpeg_bufs[i]/jpeg_sizes[i] -- no shared mutable state between
   * iterations, so this is safe to parallelize with a plain worksharing
   * loop and no locking. */
#ifdef JPAR_USE_OPENMP
  #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
  for (i = 0; i < n; i++) {
    (void)jpar_encode_strip(image_rows, image_width, &strips[i],
                             input_components, in_color_space, quality,
                             &jpeg_bufs[i], &jpeg_sizes[i]);
  }

  for (i = 0; i < n; i++) {
    if (jpeg_bufs[i] == NULL)
      all_ok = FALSE;
  }

  return all_ok;
}
