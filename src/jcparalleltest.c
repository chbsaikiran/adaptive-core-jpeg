/*
 * jcparalleltest.c
 *
 * Correctness smoke test for the experimental strip-based multi-core
 * encoder in jcparallel.c/.h.
 *
 * This test:
 *   1. Generates a synthetic RGB image (gradient + pseudo-noise, so every
 *      8x8 block has real high-frequency content to exercise the DCT).
 *   2. Encodes it once, single-threaded, as a normal whole-image JPEG --
 *      this is the "reference".
 *   3. For several thread counts (1, 2, 3, 4), splits the same image into
 *      strips with jpar_compute_strip_bounds() and encodes each strip with
 *      jpar_encode_strip()/jpar_encode_strips_parallel().
 *   4. Decodes every per-strip JPEG independently and reassembles the
 *      decoded rows into a full-image buffer at the strip's original row
 *      offset.
 *   5. Byte-compares the reassembled pixels against the reference decode.
 *   6. Separately, also encodes the same image with
 *      jpar_encode_strips_spliced() (same thread counts) and decodes *that*
 *      single spliced JPEG directly with one ordinary decode pass, byte-
 *      comparing against the same reference.
 *
 * Step 4-5 decode each strip as its own standalone JPEG, isolating the
 * strip-splitting/per-strip-encoding logic; step 6 instead exercises the
 * restart-marker splice (jpar_splice_strips()) that merges strips into one
 * real JPEG file, the form jcadaptive.c actually ships.
 *
 * Exit code is 0 if every thread count passes both checks, 1 otherwise.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <setjmp.h>

#include "jpeglib.h"
#include "jcparallel.h"

#define TEST_WIDTH    640
#define TEST_HEIGHT   481      /* deliberately NOT a multiple of the MCU row
                                   height, to exercise partial-MCU handling
                                   at the bottom of the image */
#define TEST_QUALITY  85


typedef struct {
  struct jpeg_error_mgr pub;
  jmp_buf setjmp_buffer;
} test_error_mgr;

static void
test_error_exit(j_common_ptr cinfo)
{
  test_error_mgr *myerr = (test_error_mgr *)cinfo->err;

  (*cinfo->err->output_message) (cinfo);
  longjmp(myerr->setjmp_buffer, 1);
}


/* Tiny deterministic PRNG (xorshift32) so the test image is reproducible
 * without depending on any external test asset. */
static unsigned int
xorshift32(unsigned int *state)
{
  unsigned int x = *state;

  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

static void
generate_test_image(JSAMPLE *buf, int width, int height)
{
  unsigned int rng = 0x1234567u;
  int x, y;

  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      JSAMPLE *p = buf + (size_t)y * width * 3 + (size_t)x * 3;

      p[0] = (JSAMPLE)((x * 255) / (width - 1));     /* R: horiz gradient */
      p[1] = (JSAMPLE)((y * 255) / (height - 1));     /* G: vert gradient */
      p[2] = (JSAMPLE)(xorshift32(&rng) & 0xFF);      /* B: pseudo-noise */
    }
  }
}


/* Returns the max component v_samp_factor libjpeg picks by default for the
 * given input color space -- needed to compute MCU row height. */
static int
get_max_v_samp_factor(J_COLOR_SPACE in_color_space, int input_components)
{
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  int i, max_v = 1;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  cinfo.image_width = 16;
  cinfo.image_height = 16;
  cinfo.input_components = input_components;
  cinfo.in_color_space = in_color_space;
  jpeg_set_defaults(&cinfo);

  for (i = 0; i < cinfo.num_components; i++) {
    if (cinfo.comp_info[i].v_samp_factor > max_v)
      max_v = cinfo.comp_info[i].v_samp_factor;
  }

  jpeg_destroy_compress(&cinfo);
  return max_v;
}


static boolean
encode_reference(JSAMPARRAY rows, JDIMENSION width, JDIMENSION height,
                  int quality, unsigned char **jpeg_buf,
                  unsigned long *jpeg_size)
{
  struct jpeg_compress_struct cinfo;
  test_error_mgr jerr;

  *jpeg_buf = NULL;
  *jpeg_size = 0;

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = test_error_exit;

  if (setjmp(jerr.setjmp_buffer)) {
    jpeg_destroy_compress(&cinfo);
    return FALSE;
  }

  jpeg_create_compress(&cinfo);
  jpeg_mem_dest(&cinfo, jpeg_buf, jpeg_size);

  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE);

  jpeg_start_compress(&cinfo, TRUE);

  while (cinfo.next_scanline < cinfo.image_height) {
    JSAMPROW row_pointer[1];

    row_pointer[0] = rows[cinfo.next_scanline];
    (void)jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);

  return TRUE;
}


/* Decodes one standalone JPEG (a full image or one strip's mini-JPEG) as
 * RGB and writes its scanlines into out_buf starting at row out_start_row
 * (out_row_stride bytes per row). */
static boolean
decode_jpeg(const unsigned char *jpeg_buf, unsigned long jpeg_size,
            JDIMENSION expected_width, JSAMPLE *out_buf,
            JDIMENSION out_row_stride, JDIMENSION out_start_row,
            JDIMENSION *decoded_rows_out)
{
  struct jpeg_decompress_struct cinfo;
  test_error_mgr jerr;

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = test_error_exit;

  if (setjmp(jerr.setjmp_buffer)) {
    jpeg_destroy_decompress(&cinfo);
    return FALSE;
  }

  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, jpeg_buf, jpeg_size);
  (void)jpeg_read_header(&cinfo, TRUE);
  cinfo.out_color_space = JCS_RGB;

  /* This test decodes each strip as its own standalone JPEG and stitches
   * the *decoded pixels* back together -- it does not exercise the (not
   * yet implemented) restart-marker splice that would let a single decode
   * pass see continuous chroma context across former strip boundaries.
   * With "fancy" chroma upsampling (the decompress-side default), each
   * strip's own top/bottom would be treated as a real image edge and
   * edge-replicated, which a single continuous decode of the reference
   * image never does at that same (interior) row.  Disabling it here, on
   * both the reference and every strip, removes that decode-only artifact
   * so this comparison actually isolates encode-side correctness -- see
   * jcparallel.h for the full explanation. */
  cinfo.do_fancy_upsampling = FALSE;

  jpeg_start_decompress(&cinfo);

  if (cinfo.output_width != expected_width || cinfo.output_components != 3) {
    jpeg_destroy_decompress(&cinfo);
    return FALSE;
  }

  while (cinfo.output_scanline < cinfo.output_height) {
    JSAMPROW row_pointer[1];
    JDIMENSION row = out_start_row + cinfo.output_scanline;

    row_pointer[0] = out_buf + (size_t)row * out_row_stride;
    (void)jpeg_read_scanlines(&cinfo, row_pointer, 1);
  }

  *decoded_rows_out = cinfo.output_height;

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  return TRUE;
}


int
main(void)
{
  JDIMENSION row_stride = (JDIMENSION)TEST_WIDTH * 3;
  JSAMPLE *image_buf, *ref_pixels, *assembled_pixels;
  JSAMPROW *rows;
  unsigned char *ref_jpeg_buf = NULL;
  unsigned long ref_jpeg_size = 0;
  int max_v_samp_factor;
  static const int thread_counts[] = { 1, 2, 3, 4 };
  size_t num_thread_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);
  size_t tc;
  int overall_ok = 1;
  int y;

  image_buf = (JSAMPLE *)malloc((size_t)row_stride * TEST_HEIGHT);
  rows = (JSAMPROW *)malloc(sizeof(JSAMPROW) * TEST_HEIGHT);
  ref_pixels = (JSAMPLE *)malloc((size_t)row_stride * TEST_HEIGHT);
  assembled_pixels = (JSAMPLE *)malloc((size_t)row_stride * TEST_HEIGHT);
  if (!image_buf || !rows || !ref_pixels || !assembled_pixels) {
    fprintf(stderr, "out of memory\n");
    return 1;
  }

  generate_test_image(image_buf, TEST_WIDTH, TEST_HEIGHT);
  for (y = 0; y < TEST_HEIGHT; y++)
    rows[y] = image_buf + (size_t)y * row_stride;

  max_v_samp_factor = get_max_v_samp_factor(JCS_RGB, 3);
  printf("Image: %dx%d, quality=%d, max_v_samp_factor=%d "
         "(MCU row height=%d scanlines)\n", TEST_WIDTH, TEST_HEIGHT,
         TEST_QUALITY, max_v_samp_factor, DCTSIZE * max_v_samp_factor);

  printf("Encoding single-threaded reference JPEG...\n");
  if (!encode_reference(rows, TEST_WIDTH, TEST_HEIGHT, TEST_QUALITY,
                         &ref_jpeg_buf, &ref_jpeg_size)) {
    fprintf(stderr, "reference encode failed\n");
    return 1;
  }
  {
    JDIMENSION decoded_rows;

    if (!decode_jpeg(ref_jpeg_buf, ref_jpeg_size, TEST_WIDTH, ref_pixels,
                      row_stride, 0, &decoded_rows) ||
        decoded_rows != (JDIMENSION)TEST_HEIGHT) {
      fprintf(stderr, "reference decode failed\n");
      free(ref_jpeg_buf);
      return 1;
    }
  }
  free(ref_jpeg_buf);

  for (tc = 0; tc < num_thread_counts; tc++) {
    int num_threads = thread_counts[tc];
    jpar_strip strips[4];
    unsigned char *jpeg_bufs[4];
    unsigned long jpeg_sizes[4];
    int num_strips = 0;
    int i;
    boolean encode_ok, decode_ok = TRUE;
    clock_t t0, t1;

    t0 = clock();
    encode_ok = jpar_encode_strips_parallel(rows, TEST_WIDTH, TEST_HEIGHT,
                                             max_v_samp_factor, 3, JCS_RGB,
                                             TEST_QUALITY, num_threads,
                                             strips, jpeg_bufs, jpeg_sizes,
                                             &num_strips);
    t1 = clock();

    if (!encode_ok) {
      printf("[FAIL] threads=%d: one or more strips failed to encode\n",
             num_threads);
      overall_ok = 0;
      for (i = 0; i < num_strips; i++)
        if (jpeg_bufs[i])
          free(jpeg_bufs[i]);
      continue;
    }

    for (i = 0; i < num_strips; i++) {
      JDIMENSION decoded_rows;

      if (!decode_jpeg(jpeg_bufs[i], jpeg_sizes[i], TEST_WIDTH,
                        assembled_pixels, row_stride, strips[i].start_row,
                        &decoded_rows) ||
          decoded_rows != strips[i].num_rows)
        decode_ok = FALSE;
      free(jpeg_bufs[i]);
    }

    if (!decode_ok) {
      printf("[FAIL] threads=%d: one or more strips failed to decode\n",
             num_threads);
      overall_ok = 0;
      continue;
    }

    {
      size_t total_bytes = (size_t)row_stride * TEST_HEIGHT;
      size_t diff_at = total_bytes;
      size_t k;

      for (k = 0; k < total_bytes; k++) {
        if (assembled_pixels[k] != ref_pixels[k]) {
          diff_at = k;
          break;
        }
      }

      if (diff_at == total_bytes) {
        double ms = 1000.0 * (double)(t1 - t0) / CLOCKS_PER_SEC;

        printf("[PASS] threads=%d strips=%d: decoded pixels match reference "
               "exactly (%.1f ms wall time -- informational only, not a "
               "benchmark)\n", num_threads, num_strips, ms);
      } else {
        JDIMENSION row = (JDIMENSION)(diff_at / row_stride);
        JDIMENSION col = (JDIMENSION)((diff_at % row_stride) / 3);
        int comp = (int)(diff_at % 3);

        printf("[FAIL] threads=%d strips=%d: first mismatch at row=%u "
               "col=%u component=%d (got %d, expected %d)\n", num_threads,
               num_strips, (unsigned)row, (unsigned)col, comp,
               assembled_pixels[diff_at], ref_pixels[diff_at]);
        overall_ok = 0;
      }
    }

    /* Same image/thread count, but through the spliced single-file path
     * (jpar_encode_strips_spliced()/jpar_splice_strips()) -- decodes one
     * real JPEG in one pass, rather than reassembling separately-decoded
     * strips. */
    {
      unsigned char *spliced_buf = NULL;
      unsigned long spliced_size = 0;
      int spliced_num_strips = 0;
      boolean splice_encode_ok, splice_decode_ok;

      splice_encode_ok = jpar_encode_strips_spliced(
          rows, TEST_WIDTH, TEST_HEIGHT, max_v_samp_factor, 3, JCS_RGB,
          TEST_QUALITY, num_threads, &spliced_buf, &spliced_size,
          &spliced_num_strips);

      if (!splice_encode_ok) {
        printf("[FAIL] threads=%d: spliced encode failed\n", num_threads);
        overall_ok = 0;
      } else {
        JDIMENSION decoded_rows;

        splice_decode_ok = decode_jpeg(spliced_buf, spliced_size, TEST_WIDTH,
                                        assembled_pixels, row_stride, 0,
                                        &decoded_rows) &&
                            decoded_rows == (JDIMENSION)TEST_HEIGHT;
        free(spliced_buf);

        if (!splice_decode_ok) {
          printf("[FAIL] threads=%d: spliced JPEG failed to decode\n",
                 num_threads);
          overall_ok = 0;
        } else {
          size_t total_bytes = (size_t)row_stride * TEST_HEIGHT;
          size_t diff_at = total_bytes;
          size_t k;

          for (k = 0; k < total_bytes; k++) {
            if (assembled_pixels[k] != ref_pixels[k]) {
              diff_at = k;
              break;
            }
          }

          if (diff_at == total_bytes) {
            printf("[PASS] threads=%d strips=%d spliced: decoded pixels "
                   "match reference exactly\n", num_threads,
                   spliced_num_strips);
          } else {
            JDIMENSION row = (JDIMENSION)(diff_at / row_stride);
            JDIMENSION col = (JDIMENSION)((diff_at % row_stride) / 3);
            int comp = (int)(diff_at % 3);

            printf("[FAIL] threads=%d strips=%d spliced: first mismatch at "
                   "row=%u col=%u component=%d (got %d, expected %d)\n",
                   num_threads, spliced_num_strips, (unsigned)row,
                   (unsigned)col, comp, assembled_pixels[diff_at],
                   ref_pixels[diff_at]);
            overall_ok = 0;
          }
        }
      }
    }
  }

  free(image_buf);
  free(rows);
  free(ref_pixels);
  free(assembled_pixels);

  if (overall_ok) {
    printf("ALL PASS\n");
    return 0;
  }
  printf("SOME TESTS FAILED\n");
  return 1;
}
