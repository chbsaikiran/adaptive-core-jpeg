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

  /* Restart after every MCU row.  This does two things for
   * jpar_splice_strips(): (1) it gives every strip -- regardless of its
   * own height -- the same, width-derived restart spacing, so a single
   * DRI value (taken from strip 0) is valid for the whole spliced image;
   * (2) MCUs_per_row isn't a public field before jpeg_start_compress(), so
   * it's recomputed here the same way libjpeg itself derives it, from
   * comp_info[].h_samp_factor (already set by jpeg_set_defaults() above,
   * via jpeg_set_colorspace()). */
  {
    int ci, max_h = 1;
    JDIMENSION mcus_per_row;

    for (ci = 0; ci < cinfo.num_components; ci++) {
      if (cinfo.comp_info[ci].h_samp_factor > max_h)
        max_h = cinfo.comp_info[ci].h_samp_factor;
    }
    mcus_per_row = (image_width + (JDIMENSION)(max_h * DCTSIZE) - 1) /
                   (JDIMENSION)(max_h * DCTSIZE);
    cinfo.restart_interval = mcus_per_row;
  }

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


/* ---- Splicing per-strip JPEGs into one file, via restart markers ---- */

#define JPAR_M_SOF0  0xC0
#define JPAR_M_SOS   0xDA
#define JPAR_M_EOI   0xD9
#define JPAR_M_RST0  0xD0
#define JPAR_M_RST7  0xD7

/* Walks strip 0's marker segments from SOI through the end of the SOS
 * segment (i.e. up to where entropy-coded scan data begins).  Segments in
 * this region are simple marker+length+payload TLVs with no byte-stuffing
 * ambiguity (byte-stuffing only applies inside entropy-coded scan data),
 * so this is a plain linear walk.  Returns FALSE on any unexpected/
 * malformed marker structure. */
static boolean
jpar_find_header_end(const unsigned char *buf, unsigned long size,
                      unsigned long *sos_end_out, unsigned long *sof0_pos_out)
{
  unsigned long pos;

  *sof0_pos_out = (unsigned long)-1;

  if (size < 2 || buf[0] != 0xFF || buf[1] != 0xD8 /* SOI */)
    return FALSE;
  pos = 2;

  for (;;) {
    unsigned int marker;
    unsigned long seg_len;

    if (pos + 4 > size || buf[pos] != 0xFF)
      return FALSE;
    marker = buf[pos + 1];
    pos += 2;

    if (marker == JPAR_M_SOS) {
      seg_len = ((unsigned long)buf[pos] << 8) | buf[pos + 1];
      pos += seg_len;
      if (pos > size)
        return FALSE;
      *sos_end_out = pos;
      return TRUE;
    }

    if (marker == JPAR_M_SOF0)
      *sof0_pos_out = pos - 2;

    seg_len = ((unsigned long)buf[pos] << 8) | buf[pos + 1];
    if (seg_len < 2 || pos + seg_len > size)
      return FALSE;
    pos += seg_len;
  }
}

/* Byte-stuffing-aware scan from `start` (the first byte of entropy-coded
 * scan data) for the EOI marker.  Restart markers and stuffed 0xFF 0x00
 * pairs are treated as ordinary scan data and skipped over; any other
 * marker encountered here is unexpected for a single-scan baseline strip
 * JPEG produced by jpar_encode_strip(), and is treated as a parse error. */
static boolean
jpar_find_entropy_end(const unsigned char *buf, unsigned long size,
                       unsigned long start, unsigned long *entropy_end_out)
{
  unsigned long pos = start;

  while (pos + 1 < size) {
    if (buf[pos] == 0xFF) {
      unsigned int next = buf[pos + 1];

      if (next == 0x00) {
        pos += 2;
        continue;
      }
      if (next == JPAR_M_EOI) {
        *entropy_end_out = pos;
        return TRUE;
      }
      if (next >= JPAR_M_RST0 && next <= JPAR_M_RST7) {
        pos += 2;
        continue;
      }
      return FALSE; /* unexpected marker inside entropy data */
    }
    pos++;
  }
  return FALSE;
}

GLOBAL(boolean)
jpar_splice_strips(unsigned char *const *jpeg_bufs,
                    const unsigned long *jpeg_sizes, int num_strips,
                    JDIMENSION full_image_height, unsigned char **out_buf,
                    unsigned long *out_size)
{
  unsigned long header_len, sof0_pos;
  unsigned long *entropy_starts = NULL, *entropy_ends = NULL;
  unsigned long total_entropy = 0, out_capacity, out_pos;
  int i;
  boolean ok = TRUE;

  *out_buf = NULL;
  *out_size = 0;

  if (num_strips < 1)
    return FALSE;

  if (!jpar_find_header_end(jpeg_bufs[0], jpeg_sizes[0], &header_len,
                             &sof0_pos))
    return FALSE;
  /* SOF0 payload: marker(2) length(2) precision(1) height(2) width(2) ... --
   * need bytes up through the width field (index sof0_pos+8) to be safe to
   * touch, though only the height field (sof0_pos+5,+6) is patched. */
  if (sof0_pos == (unsigned long)-1 || sof0_pos + 9 > jpeg_sizes[0])
    return FALSE;

  entropy_starts = (unsigned long *)malloc(sizeof(unsigned long) *
                                            (size_t)num_strips);
  entropy_ends = (unsigned long *)malloc(sizeof(unsigned long) *
                                          (size_t)num_strips);
  if (!entropy_starts || !entropy_ends) {
    free(entropy_starts);
    free(entropy_ends);
    return FALSE;
  }

  for (i = 0; i < num_strips; i++) {
    unsigned long this_header_len, this_sof0_pos, this_end;

    if (i == 0) {
      this_header_len = header_len;
    } else if (!jpar_find_header_end(jpeg_bufs[i], jpeg_sizes[i],
                                      &this_header_len, &this_sof0_pos)) {
      ok = FALSE;
      break;
    }

    if (!jpar_find_entropy_end(jpeg_bufs[i], jpeg_sizes[i], this_header_len,
                                &this_end)) {
      ok = FALSE;
      break;
    }

    /* Trim a restart marker libjpeg may have emitted right at the end of
     * this strip's own scan (happens when the strip's MCU count is an
     * exact multiple of the restart interval) -- a fresh boundary marker
     * is inserted between strips below instead, so this would otherwise be
     * a harmless but confusing duplicate right next to it. */
    if (this_end - this_header_len >= 2 &&
        jpeg_bufs[i][this_end - 2] == 0xFF &&
        jpeg_bufs[i][this_end - 1] >= JPAR_M_RST0 &&
        jpeg_bufs[i][this_end - 1] <= JPAR_M_RST7)
      this_end -= 2;

    entropy_starts[i] = this_header_len;
    entropy_ends[i] = this_end;
    total_entropy += (this_end - this_header_len);
  }

  if (ok) {
    /* header + all strips' entropy data + one boundary marker (2 bytes)
     * between every pair of strips + final EOI (2 bytes). */
    out_capacity = header_len + total_entropy +
                   (unsigned long)(num_strips - 1) * 2 + 2;
    *out_buf = (unsigned char *)malloc(out_capacity);
    if (!*out_buf) {
      ok = FALSE;
    } else {
      unsigned char *out = *out_buf;
      unsigned long pos;
      unsigned int seq;

      memcpy(out, jpeg_bufs[0], header_len);
      out_pos = header_len;

      /* Patch SOF0's height field to the full spliced image height (each
       * strip's own header, discarded here except strip 0's, describes
       * only its own strip height). */
      out[sof0_pos + 5] = (unsigned char)((full_image_height >> 8) & 0xFF);
      out[sof0_pos + 6] = (unsigned char)(full_image_height & 0xFF);

      for (i = 0; i < num_strips; i++) {
        unsigned long len = entropy_ends[i] - entropy_starts[i];

        if (i > 0) {
          out[out_pos++] = 0xFF;
          out[out_pos++] = JPAR_M_RST0; /* renumbered below */
        }
        memcpy(out + out_pos, jpeg_bufs[i] + entropy_starts[i], len);
        out_pos += len;
      }

      /* Globally renumber every restart marker in the spliced entropy
       * stream -- both the ones libjpeg emitted inside each strip and the
       * boundary ones just inserted above -- so RST0..RST7 cycle correctly
       * from the start of the scan.  (Each strip's own internal numbering
       * independently restarts from 0, since it was encoded as its own
       * compress session; a decoder validates that the sequence is
       * continuous.) */
      pos = header_len;
      seq = 0;
      while (pos + 1 < out_pos) {
        if (out[pos] == 0xFF) {
          unsigned int next = out[pos + 1];

          if (next == 0x00) {
            pos += 2;
            continue;
          }
          if (next >= JPAR_M_RST0 && next <= JPAR_M_RST7) {
            out[pos + 1] = (unsigned char)(JPAR_M_RST0 + (seq & 7));
            seq++;
            pos += 2;
            continue;
          }
        }
        pos++;
      }

      out[out_pos++] = 0xFF;
      out[out_pos++] = JPAR_M_EOI;
      *out_size = out_pos;
    }
  }

  free(entropy_starts);
  free(entropy_ends);
  return ok;
}

GLOBAL(boolean)
jpar_encode_strips_spliced(JSAMPARRAY image_rows, JDIMENSION image_width,
                            JDIMENSION image_height, int max_v_samp_factor,
                            int input_components,
                            J_COLOR_SPACE in_color_space, int quality,
                            int num_threads, unsigned char **out_buf,
                            unsigned long *out_size, int *num_strips_out)
{
  jpar_strip *strips;
  unsigned char **jpeg_bufs;
  unsigned long *jpeg_sizes;
  boolean encode_ok, splice_ok;
  int i, n = 0;

  *out_buf = NULL;
  *out_size = 0;
  *num_strips_out = 0;

  if (num_threads < 1)
    num_threads = 1;

  strips = (jpar_strip *)malloc(sizeof(jpar_strip) * (size_t)num_threads);
  jpeg_bufs = (unsigned char **)malloc(sizeof(unsigned char *) *
                                        (size_t)num_threads);
  jpeg_sizes = (unsigned long *)malloc(sizeof(unsigned long) *
                                        (size_t)num_threads);
  if (!strips || !jpeg_bufs || !jpeg_sizes) {
    free(strips);
    free(jpeg_bufs);
    free(jpeg_sizes);
    return FALSE;
  }

  encode_ok = jpar_encode_strips_parallel(
      image_rows, image_width, image_height, max_v_samp_factor,
      input_components, in_color_space, quality, num_threads, strips,
      jpeg_bufs, jpeg_sizes, &n);
  *num_strips_out = n;

  splice_ok = encode_ok && jpar_splice_strips(jpeg_bufs, jpeg_sizes, n,
                                               image_height, out_buf,
                                               out_size);

  for (i = 0; i < n; i++)
    free(jpeg_bufs[i]);
  free(strips);
  free(jpeg_bufs);
  free(jpeg_sizes);

  return splice_ok;
}
