/*
 * jcparallel.h
 *
 * Experimental strip-based multi-core JPEG compression helpers.
 *
 * This is NOT part of the public libjpeg or TurboJPEG API.  It is a
 * standalone building block for a strip-parallel encoder:  an image is cut
 * into horizontal strips, each strip is compressed independently (and, when
 * built with OpenMP, in parallel across threads) into its own standalone
 * JPEG stream, and the resulting per-strip streams are spliced into a
 * single JPEG file using restart markers (jpar_splice_strips() /
 * jpar_encode_strips_spliced(), below).
 *
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 */

#ifndef JCPARALLEL_H
#define JCPARALLEL_H

#include "jpeglib.h"


/* Describes one horizontal slice of a source image, in source scanlines. */
typedef struct {
  JDIMENSION start_row;         /* first source scanline (0-based) */
  JDIMENSION num_rows;          /* number of source scanlines in this strip */
} jpar_strip;


/*
 * Divides an image of the given height into at most num_threads horizontal
 * strips and writes their bounds into `strips` (which must have room for
 * num_threads entries).  Every strip boundary is aligned to a whole number
 * of MCU rows (MCU row height = DCTSIZE * max_v_samp_factor), *except* the
 * boundary at the very bottom of the image, where the last strip simply
 * absorbs whatever scanlines remain (including any partial MCU row).
 *
 * This alignment rule is what makes strip-parallel encoding produce the
 * same decoded pixels as a normal single-threaded encode:
 *   - Chroma downsampling groups (e.g. the 2x2 pixel blocks used by 4:2:0)
 *     never straddle a strip boundary, because every interior boundary
 *     falls on a whole MCU row.
 *   - Only the last strip can end on a partial MCU row, and it does so at
 *     exactly the same place (the true bottom of the image) that a single-
 *     threaded encode's own edge-replication padding would kick in.
 *
 * Returns the number of strips actually produced.  This is <= num_threads;
 * it can be smaller if the image is too short to give every thread at
 * least one MCU row (each strip always owns at least one MCU row, so every
 * strip's Huffman encoder gets to reset its DC predictors on real data).
 */
EXTERN(int) jpar_compute_strip_bounds(JDIMENSION image_height,
                                       int max_v_samp_factor, int num_threads,
                                       jpar_strip *strips);

/*
 * Encodes one horizontal strip of a source image as a complete, standalone
 * baseline JPEG (its own SOI...EOI) into a freshly malloc'd memory buffer.
 *
 * `image_rows` must be an array of image_height scanline pointers for the
 * *whole* source image, laid out exactly as jpeg_write_scanlines() expects
 * (input_components samples per pixel, in in_color_space's channel order);
 * this function reads only the rows belonging to `strip`.
 *
 * optimize_coding is forced off, so every strip produced by this function
 * for a given quality/color space uses the same, standard Huffman tables --
 * a prerequisite for later splicing strips together via restart markers.
 * (See jcparallel.c for why the default chroma downsampler is *not* an
 * issue here: unlike JPEG *decoding*'s optional "fancy" upsampler, the
 * encode-side downsampler libjpeg-turbo uses by default has no dependency
 * on rows outside its own MCU-row group, so it can't see across a strip
 * boundary in the first place.)
 *
 * CAVEAT FOR CALLERS THAT VERIFY OUTPUT BY DECODING EACH STRIP SEPARATELY:
 * if you decode a strip's standalone JPEG on its own (rather than after
 * splicing all strips into one file via restart markers), the *decoder*
 * will treat that strip's top/bottom as real image edges and may apply
 * edge-replication there if "fancy" chroma upsampling is enabled (the
 * decompress-side do_fancy_upsampling, on by default) -- producing pixels
 * that differ from decoding the same region inside a single continuous
 * image, purely as an artifact of decoding strips in isolation.  This
 * caveat applies only when decoding strips separately, as jcparalleltest.c
 * does to isolate encode-side correctness; jpar_encode_strips_spliced()
 * below produces one continuous JPEG, where decoding sees continuous
 * chroma context across former strip boundaries and this caveat does not
 * apply.
 *
 * On success, returns TRUE and sets *jpeg_buf/*jpeg_size; the caller must
 * free(*jpeg_buf) when done with it.  On failure, returns FALSE and leaves
 * *jpeg_buf NULL / *jpeg_size 0.
 *
 * This function is self-contained (its own jpeg_compress_struct, its own
 * error handler, no shared mutable state) and safe to call concurrently
 * from multiple threads, each on a different strip of the same image.
 */
EXTERN(boolean) jpar_encode_strip(JSAMPARRAY image_rows, JDIMENSION image_width,
                                   const jpar_strip *strip,
                                   int input_components,
                                   J_COLOR_SPACE in_color_space, int quality,
                                   unsigned char **jpeg_buf,
                                   unsigned long *jpeg_size);

/*
 * Convenience driver: computes strip bounds (as jpar_compute_strip_bounds)
 * and then encodes every strip (as jpar_encode_strip), one per strip, into
 * `jpeg_bufs`/`jpeg_sizes` (each of which must have room for num_threads
 * entries).  When built with JPAR_USE_OPENMP defined, the per-strip encodes
 * run in parallel across up to num_threads OpenMP threads; otherwise they
 * run sequentially in strip order.
 *
 * *num_strips_out receives the actual number of strips (see
 * jpar_compute_strip_bounds).  Only the first *num_strips_out entries of
 * `strips`/`jpeg_bufs`/`jpeg_sizes` are meaningful.
 *
 * Returns TRUE only if every strip encoded successfully.  On a partial
 * failure, the caller is still responsible for free()ing whichever
 * jpeg_bufs[i] entries are non-NULL.
 */
EXTERN(boolean) jpar_encode_strips_parallel(JSAMPARRAY image_rows,
                                             JDIMENSION image_width,
                                             JDIMENSION image_height,
                                             int max_v_samp_factor,
                                             int input_components,
                                             J_COLOR_SPACE in_color_space,
                                             int quality, int num_threads,
                                             jpar_strip *strips,
                                             unsigned char **jpeg_bufs,
                                             unsigned long *jpeg_sizes,
                                             int *num_strips_out);

/*
 * Splices `num_strips` standalone per-strip JPEGs (as produced by
 * jpar_encode_strip(), which sets a restart interval of one MCU row on
 * every strip specifically so this works -- see jcparallel.c) into a
 * single valid baseline JPEG of the given full image height, using restart
 * markers at each strip boundary.
 *
 * How this works: every strip is encoded with a restart marker after every
 * MCU row, so restart boundaries land at a fixed, width-derived spacing
 * that's identical across all strips (unlike strip *height*, which can
 * differ for the last strip -- see jpar_compute_strip_bounds()).  This
 * function takes strip 0's header verbatim (SOI through the SOS segment --
 * same quality/quantization/Huffman tables and restart interval on every
 * strip already, by construction), patches its SOF0 height field to the
 * full image height, concatenates every strip's entropy-coded scan data
 * (trimming any restart marker libjpeg happened to emit immediately before
 * a strip's own EOI, and inserting exactly one restart marker of its own
 * at each strip boundary instead), renumbers every restart marker
 * (RST0..RST7, cycling) sequentially across the whole spliced scan -- each
 * strip's own internal numbering restarts from 0, since it was encoded as
 * an independent compress session -- and appends a single EOI.
 *
 * Decoding the spliced result sees one continuous scan: DC prediction
 * resets at each restart marker exactly where each strip's own independent
 * encode would have started predicting from 0 anyway, so this reproduces
 * the same decoded pixels as jpar_encode_strip() per strip while being one
 * standalone, complete JPEG file.
 *
 * On success, returns TRUE and sets *out_buf/*out_size (caller must
 * free(*out_buf)).  Returns FALSE if num_strips < 1 or any strip's buffer
 * doesn't parse as the expected single-scan baseline JPEG structure.
 */
EXTERN(boolean) jpar_splice_strips(unsigned char *const *jpeg_bufs,
                                    const unsigned long *jpeg_sizes,
                                    int num_strips,
                                    JDIMENSION full_image_height,
                                    unsigned char **out_buf,
                                    unsigned long *out_size);

/*
 * Convenience driver: like jpar_encode_strips_parallel(), but returns one
 * spliced JPEG buffer (via jpar_splice_strips()) instead of an array of
 * per-strip buffers.  Manages its own strip/jpeg_bufs/jpeg_sizes scratch
 * arrays internally (sized to num_threads) and frees the per-strip buffers
 * before returning.
 *
 * *num_strips_out receives the actual number of strips used (informational
 * -- see jpar_compute_strip_bounds()).  On success, returns TRUE and sets
 * *out_buf/*out_size (caller must free(*out_buf)).  Returns FALSE if any
 * strip failed to encode or splicing failed.
 */
EXTERN(boolean) jpar_encode_strips_spliced(JSAMPARRAY image_rows,
                                            JDIMENSION image_width,
                                            JDIMENSION image_height,
                                            int max_v_samp_factor,
                                            int input_components,
                                            J_COLOR_SPACE in_color_space,
                                            int quality, int num_threads,
                                            unsigned char **out_buf,
                                            unsigned long *out_size,
                                            int *num_strips_out);

#endif /* JCPARALLEL_H */
