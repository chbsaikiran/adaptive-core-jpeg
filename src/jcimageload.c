/*
 * jcimageload.c
 *
 * See jcimageload.h.  Extracted from jcparallelbench.c so jcadaptive.c can
 * reuse the same image-loading logic without duplicating it.
 *
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 */

#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cdjpeg.h" /* jinit_read_bmp/jinit_read_png/jinit_read_ppm */
#include "jcimageload.h"
#include "jpeglib.h"

#ifdef _WIN32
#define jcil_strcasecmp _stricmp
#else
#include <strings.h> /* strcasecmp */
#define jcil_strcasecmp strcasecmp
#endif

typedef struct {
  struct jpeg_error_mgr pub;
  jmp_buf setjmp_buffer;
} jcil_error_mgr;

static void jcil_error_exit(j_common_ptr cinfo) {
  jcil_error_mgr *myerr = (jcil_error_mgr *)cinfo->err;

  (*cinfo->err->output_message)(cinfo);
  longjmp(myerr->setjmp_buffer, 1);
}

EXTERN(void) jcil_free_loaded_image(jcil_loaded_image *img) {
  free(img->data);
  free(img->rows);
  img->data = NULL;
  img->rows = NULL;
}

/* Loads an 8-bit JPEG file.  Returns FALSE (with img left zeroed) if the
 * file can't be decoded or isn't 8-bit. */
static boolean jcil_load_jpeg(const char *path, jcil_loaded_image *img) {
  struct jpeg_decompress_struct cinfo;
  jcil_error_mgr jerr;
  FILE *f = fopen(path, "rb");
  JDIMENSION row_stride;

  memset(img, 0, sizeof(*img));
  if (!f)
    return FALSE;

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = jcil_error_exit;

  if (setjmp(jerr.setjmp_buffer)) {
    jpeg_destroy_decompress(&cinfo);
    fclose(f);
    jcil_free_loaded_image(img);
    return FALSE;
  }

  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, f);
  (void)jpeg_read_header(&cinfo, TRUE);

  if (cinfo.data_precision != 8) {
    fprintf(stderr, "skip %s: data precision %d not supported (need 8-bit)\n",
            path, cinfo.data_precision);
    jpeg_destroy_decompress(&cinfo);
    fclose(f);
    return FALSE;
  }

  cinfo.out_color_space = (cinfo.num_components == 1) ? JCS_GRAYSCALE : JCS_RGB;
  jpeg_start_decompress(&cinfo);

  row_stride = (JDIMENSION)cinfo.output_width * cinfo.output_components;
  img->width = cinfo.output_width;
  img->height = cinfo.output_height;
  img->components = cinfo.output_components;
  img->color_space = cinfo.out_color_space;
  img->data = (JSAMPLE *)malloc((size_t)row_stride * img->height);
  img->rows = (JSAMPROW *)malloc(sizeof(JSAMPROW) * img->height);
  if (!img->data || !img->rows) {
    jpeg_destroy_decompress(&cinfo);
    fclose(f);
    jcil_free_loaded_image(img);
    return FALSE;
  }
  {
    JDIMENSION y;

    for (y = 0; y < img->height; y++)
      img->rows[y] = img->data + (size_t)y * row_stride;
  }

  while (cinfo.output_scanline < cinfo.output_height) {
    JSAMPROW row = img->rows[cinfo.output_scanline];

    (void)jpeg_read_scanlines(&cinfo, &row, 1);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(f);
  return TRUE;
}

/* Loads a BMP ('B'), PNG ('N'), or PPM/PGM ('P') file by driving cjpeg's
 * own existing reader modules (the same ones cjpeg.c uses) directly, one
 * scanline at a time, copying each into our own persistent buffer -- the
 * reader's buffer is reused/overwritten on every get_pixel_rows() call, so
 * it can't be kept as-is the way jpar_encode_strip() needs its input. */
static boolean jcil_load_via_cjpeg_reader(const char *path, char format,
                                          jcil_loaded_image *img) {
  struct jpeg_compress_struct cinfo;
  jcil_error_mgr jerr;
  cjpeg_source_ptr src;
  FILE *f = fopen(path, "rb");
  JDIMENSION row_stride, scanline;

  memset(img, 0, sizeof(*img));
  if (!f)
    return FALSE;

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = jcil_error_exit;

  if (setjmp(jerr.setjmp_buffer)) {
    jpeg_destroy_compress(&cinfo);
    fclose(f);
    jcil_free_loaded_image(img);
    return FALSE;
  }

  jpeg_create_compress(&cinfo);
  cinfo.in_color_space = JCS_UNKNOWN; /* let the reader pick */
  cinfo.data_precision = 8;

  if (format == 'B')
    src = jinit_read_bmp(&cinfo, TRUE);
  else if (format == 'N')
    src = jinit_read_png(&cinfo);
  else
    src = jinit_read_ppm(&cinfo);
  src->input_file = f;
  (*src->start_input)(&cinfo, src);

  /* BMP's use_inversion_array=TRUE path (needed to turn its bottom-up row
   * order into the top-down order everything else expects) requests a
   * virtual array via cinfo->mem, but that array's backing store isn't
   * actually allocated until realize_virt_arrays() runs -- normally done
   * for us inside jpeg_start_compress(), which this path never calls.
   * (A no-op for PNG/PPM, which never request a virtual array.) */
  (*cinfo.mem->realize_virt_arrays)((j_common_ptr)&cinfo);

  if (cinfo.data_precision != 8) {
    fprintf(stderr, "skip %s: data precision %d not supported (need 8-bit)\n",
            path, cinfo.data_precision);
    jpeg_destroy_compress(&cinfo);
    fclose(f);
    return FALSE;
  }

  row_stride = (JDIMENSION)cinfo.image_width * cinfo.input_components;
  img->width = cinfo.image_width;
  img->height = cinfo.image_height;
  img->components = cinfo.input_components;
  img->color_space = cinfo.in_color_space;
  img->data = (JSAMPLE *)malloc((size_t)row_stride * img->height);
  img->rows = (JSAMPROW *)malloc(sizeof(JSAMPROW) * img->height);
  if (!img->data || !img->rows) {
    jpeg_destroy_compress(&cinfo);
    fclose(f);
    jcil_free_loaded_image(img);
    return FALSE;
  }
  {
    JDIMENSION y;

    for (y = 0; y < img->height; y++)
      img->rows[y] = img->data + (size_t)y * row_stride;
  }

  /* Local scanline counter -- deliberately not cinfo.next_scanline, since
   * that field belongs to the compress state machine we never start here
   * (no jpeg_start_compress()/jpeg_write_scanlines() calls in this path). */
  scanline = 0;
  while (scanline < cinfo.image_height) {
    JDIMENSION num_rows = (*src->get_pixel_rows)(&cinfo, src);
    JDIMENSION i;

    for (i = 0; i < num_rows && scanline < cinfo.image_height; i++, scanline++)
      memcpy(img->rows[scanline], src->buffer[i], row_stride);
  }

  (*src->finish_input)(&cinfo, src);
  jpeg_destroy_compress(&cinfo);
  fclose(f);
  return TRUE;
}

static boolean jcil_has_extension(const char *name, const char *ext) {
  size_t name_len = strlen(name), ext_len = strlen(ext);

  if (ext_len > name_len)
    return FALSE;
  return jcil_strcasecmp(name + (name_len - ext_len), ext) == 0;
}

EXTERN(boolean) jcil_load_image(const char *path, jcil_loaded_image *img) {
  const char *name = path;
  const char *slash;

  /* Match on the filename only, so a path containing ".png/" etc in a
   * directory component can't confuse the extension check. */
  for (slash = path; *slash; slash++) {
    if (*slash == '/' || *slash == '\\')
      name = slash + 1;
  }

  if (jcil_has_extension(name, ".jpg") || jcil_has_extension(name, ".jpeg"))
    return jcil_load_jpeg(path, img);
  if (jcil_has_extension(name, ".bmp"))
    return jcil_load_via_cjpeg_reader(path, 'B', img);
  if (jcil_has_extension(name, ".png"))
    return jcil_load_via_cjpeg_reader(path, 'N', img);
  if (jcil_has_extension(name, ".ppm") || jcil_has_extension(name, ".pgm"))
    return jcil_load_via_cjpeg_reader(path, 'P', img);
  return FALSE;
}

EXTERN(int) jcil_get_max_v_samp_factor(J_COLOR_SPACE in_color_space,
                                        int input_components) {
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
