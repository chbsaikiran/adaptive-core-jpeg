/*
 * jcparallelbench.c
 *
 * Per-image benchmark driver for the experimental strip-based multi-core
 * encoder in jcparallel.c/.h.  See jcparallel.h for the encoder's design
 * rationale; see jcparalleltest.c for its pixel-correctness test.
 *
 * Unlike jcparalleltest.c (one synthetic image, informational timing),
 * this tool exists to produce real per-image timing *data*: given a
 * directory of images and a fixed thread count, it decodes each image
 * once, then repeatedly times just the jpar_encode_strips_parallel() call
 * (never the decode/file I/O), and prints one CSV row per image with its
 * mean and minimum encode time.  Running this twice -- once with
 * --threads 2, once with --threads 4 -- over the same directory produces
 * the two per-image timing columns that a downstream step can compare
 * (e.g. to label images "complex" if 4 cores meaningfully beat 2 cores,
 * "simple" otherwise, for training an image-complexity classifier that
 * picks a core count up front).
 *
 * Timing uses CLOCK_MONOTONIC wall-clock time, not clock().  clock()
 * measures CPU time, which sums across every OpenMP thread -- a 4-thread
 * run doing the same wall-clock-visible work as a 2-thread run would
 * report *similar or greater* CPU time despite finishing sooner, which
 * would defeat the entire point of a 2-core-vs-4-core comparison.
 *
 * Supported input formats: JPEG (8-bit/baseline or arithmetic; 12-/16-bit
 * precision files are skipped, since jpar_encode_strip's plain
 * jpeg_compress_struct API is 8-bit only), BMP, and PPM/PGM.  BMP/PPM/PGM
 * are decoded by reusing cjpeg's own existing, tested reader modules
 * (jinit_read_bmp()/jinit_read_ppm() from rdbmp.c/rdppm.c, declared in
 * cdjpeg.h) rather than reimplementing those formats.  Any other
 * extension is silently skipped, so pointing this at a real-world photo
 * folder that also contains stray non-image files is safe.
 *
 * A failure to load or encode one image is a warning, not a fatal error;
 * the run continues with the remaining images.
 *
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <setjmp.h>
#include <sys/stat.h>

#include "jpeglib.h"
#include "cdjpeg.h"             /* jinit_read_bmp/jinit_read_ppm, cjpeg_source_ptr */
#include "jcparallel.h"

/*
 * This tool was written against POSIX (dirent.h for directory listing,
 * strings.h for strcasecmp, clock_gettime(CLOCK_MONOTONIC) for timing).
 * MSVC/Windows has none of those, so on _WIN32 we provide just enough of
 * that surface -- a tiny opendir/readdir over _findfirst/_findnext, a
 * strcasecmp alias, an S_ISREG fallback, and a clock_gettime backed by C11
 * timespec_get -- to let the rest of this file build and run unchanged.
 * Everywhere else, just use the real headers.
 */
#ifdef _WIN32

#include <io.h>                 /* _findfirst / _findnext / _finddata_t */

#define strcasecmp _stricmp
#define strdup     _strdup

#ifndef S_ISREG
#define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
#endif

typedef struct {
  intptr_t handle;
  struct _finddata_t info;
  int pending;                  /* 1 while `info` holds an unconsumed entry */
} DIR;

struct dirent {
  char d_name[260];
};

static DIR *
opendir(const char *name)
{
  DIR *d = (DIR *)malloc(sizeof(*d));
  char pattern[4096];

  if (!d)
    return NULL;
  snprintf(pattern, sizeof(pattern), "%s\\*", name);
  d->handle = _findfirst(pattern, &d->info);
  if (d->handle == -1) {
    free(d);
    return NULL;
  }
  d->pending = 1;
  return d;
}

static struct dirent *
readdir(DIR *d)
{
  static struct dirent ent;    /* non-reentrant, matching POSIX readdir() */
  size_t n;

  if (!d->pending && _findnext(d->handle, &d->info) != 0)
    return NULL;
  d->pending = 0;

  n = strlen(d->info.name);
  if (n >= sizeof(ent.d_name))
    n = sizeof(ent.d_name) - 1;
  memcpy(ent.d_name, d->info.name, n);
  ent.d_name[n] = '\0';
  return &ent;
}

static int
closedir(DIR *d)
{
  if (!d)
    return -1;
  if (d->handle != -1)
    _findclose(d->handle);
  free(d);
  return 0;
}

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

/* Wall-clock time (not clock(), so it doesn't sum across OpenMP threads --
 * see the file header).  timespec_get(TIME_UTC) is C11 and present in
 * MSVC's <time.h>; it isn't guaranteed monotonic, but across the short
 * back-to-back encodes this benchmark times that's not a concern. */
static int
clock_gettime(int clk_id, struct timespec *tp)
{
  (void)clk_id;
  return timespec_get(tp, TIME_UTC) == TIME_UTC ? 0 : -1;
}

#else /* !_WIN32 */

#include <strings.h>
#include <dirent.h>

#endif /* _WIN32 */

#define DEFAULT_QUALITY  85
#define DEFAULT_REPEAT   50


typedef struct {
  struct jpeg_error_mgr pub;
  jmp_buf setjmp_buffer;
} bench_error_mgr;

static void
bench_error_exit(j_common_ptr cinfo)
{
  bench_error_mgr *myerr = (bench_error_mgr *)cinfo->err;

  (*cinfo->err->output_message) (cinfo);
  longjmp(myerr->setjmp_buffer, 1);
}


/* One fully-decoded image, ready to hand to jpar_encode_strips_parallel(). */
typedef struct {
  JSAMPLE *data;                /* contiguous height*row_stride bytes */
  JSAMPROW *rows;                /* height row pointers into data */
  JDIMENSION width, height;
  int components;
  J_COLOR_SPACE color_space;
} loaded_image;

static void
free_loaded_image(loaded_image *img)
{
  free(img->data);
  free(img->rows);
  img->data = NULL;
  img->rows = NULL;
}


/* Loads an 8-bit JPEG file.  Returns FALSE (with img left zeroed) if the
 * file can't be decoded or isn't 8-bit. */
static boolean
load_jpeg(const char *path, loaded_image *img)
{
  struct jpeg_decompress_struct cinfo;
  bench_error_mgr jerr;
  FILE *f = fopen(path, "rb");
  JDIMENSION row_stride;

  memset(img, 0, sizeof(*img));
  if (!f)
    return FALSE;

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = bench_error_exit;

  if (setjmp(jerr.setjmp_buffer)) {
    jpeg_destroy_decompress(&cinfo);
    fclose(f);
    free_loaded_image(img);
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
    free_loaded_image(img);
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


/* Loads a BMP ('B') or PPM/PGM ('P') file by driving cjpeg's own existing
 * reader modules (the same ones cjpeg.c uses) directly, one scanline at a
 * time, copying each into our own persistent buffer -- the reader's
 * buffer is reused/overwritten on every get_pixel_rows() call, so it
 * can't be kept as-is the way jpar_encode_strip() needs its input. */
static boolean
load_via_cjpeg_reader(const char *path, char format, loaded_image *img)
{
  struct jpeg_compress_struct cinfo;
  bench_error_mgr jerr;
  cjpeg_source_ptr src;
  FILE *f = fopen(path, "rb");
  JDIMENSION row_stride, scanline;

  memset(img, 0, sizeof(*img));
  if (!f)
    return FALSE;

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = bench_error_exit;

  if (setjmp(jerr.setjmp_buffer)) {
    jpeg_destroy_compress(&cinfo);
    fclose(f);
    free_loaded_image(img);
    return FALSE;
  }

  jpeg_create_compress(&cinfo);
  cinfo.in_color_space = JCS_UNKNOWN;   /* let the reader pick */
  cinfo.data_precision = 8;

  src = (format == 'B') ? jinit_read_bmp(&cinfo, TRUE) : jinit_read_ppm(&cinfo);
  src->input_file = f;
  (*src->start_input) (&cinfo, src);

  /* BMP's use_inversion_array=TRUE path (needed to turn its bottom-up row
   * order into the top-down order everything else expects) requests a
   * virtual array via cinfo->mem, but that array's backing store isn't
   * actually allocated until realize_virt_arrays() runs -- normally done
   * for us inside jpeg_start_compress(), which this path never calls. */
  (*cinfo.mem->realize_virt_arrays) ((j_common_ptr)&cinfo);

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
    free_loaded_image(img);
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
    JDIMENSION num_rows = (*src->get_pixel_rows) (&cinfo, src);
    JDIMENSION i;

    for (i = 0; i < num_rows && scanline < cinfo.image_height; i++, scanline++)
      memcpy(img->rows[scanline], src->buffer[i], row_stride);
  }

  (*src->finish_input) (&cinfo, src);
  jpeg_destroy_compress(&cinfo);
  fclose(f);
  return TRUE;
}


/* Returns the max component v_samp_factor libjpeg picks by default for the
 * given input color space -- needed to compute MCU row height for strip
 * splitting.  (Mirrors the helper of the same purpose in jcparalleltest.c.) */
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
has_extension(const char *name, const char *ext)
{
  size_t name_len = strlen(name), ext_len = strlen(ext);

  if (ext_len > name_len)
    return FALSE;
  return strcasecmp(name + (name_len - ext_len), ext) == 0;
}


/* Loads `path` (dispatching on its extension) into *img.  Returns FALSE
 * (silently, no message) if the extension isn't one we handle. */
static boolean
load_image(const char *path, const char *name, loaded_image *img)
{
  if (has_extension(name, ".jpg") || has_extension(name, ".jpeg"))
    return load_jpeg(path, img);
  if (has_extension(name, ".bmp"))
    return load_via_cjpeg_reader(path, 'B', img);
  if (has_extension(name, ".ppm") || has_extension(name, ".pgm"))
    return load_via_cjpeg_reader(path, 'P', img);
  return FALSE;
}


static int
compare_names(const void *a, const void *b)
{
  return strcmp(*(const char * const *)a, *(const char * const *)b);
}


/* Lists `dir`'s regular files, sorted, into *names_out (caller frees each
 * entry and the array).  Returns the count, or -1 on failure to open. */
static int
list_dir_sorted(const char *dir, char ***names_out)
{
  DIR *d = opendir(dir);
  struct dirent *entry;
  char **names = NULL;
  int count = 0, capacity = 0;

  if (!d)
    return -1;

  while ((entry = readdir(d)) != NULL) {
    char path[4096];
    struct stat st;

    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
      continue;
    snprintf(path, sizeof(path), "%s/%s", dir, entry->d_name);
    if (stat(path, &st) != 0 || !S_ISREG(st.st_mode))
      continue;

    if (count == capacity) {
      capacity = capacity ? capacity * 2 : 16;
      names = (char **)realloc(names, sizeof(char *) * capacity);
    }
    names[count++] = strdup(entry->d_name);
  }
  closedir(d);

  qsort(names, count, sizeof(char *), compare_names);
  *names_out = names;
  return count;
}


static double
timespec_diff(struct timespec t0, struct timespec t1)
{
  return (double)(t1.tv_sec - t0.tv_sec) +
         (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
}


static void
usage(const char *progname)
{
  fprintf(stderr,
    "Usage: %s <dataset_dir> --threads N [--quality Q] [--repeat R]\n"
    "  --threads N   number of strips/OpenMP threads to encode with (required)\n"
    "  --quality Q   JPEG quality, 0-100 (default %d)\n"
    "  --repeat R    encode each image R times and report mean/min (default %d)\n",
    progname, DEFAULT_QUALITY, DEFAULT_REPEAT);
}


int
main(int argc, char **argv)
{
  const char *dataset_dir = NULL;
  int threads = -1, quality = DEFAULT_QUALITY, repeat = DEFAULT_REPEAT;
  char **names;
  int num_names, i;
  int images_reported = 0, total_encoded = 0;
  double total_wall_time = 0.0;
  jpar_strip *strips;
  unsigned char **jpeg_bufs;
  unsigned long *jpeg_sizes;

  for (i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      threads = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--quality") == 0 && i + 1 < argc) {
      quality = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
      repeat = atoi(argv[++i]);
    } else if (argv[i][0] != '-' && !dataset_dir) {
      dataset_dir = argv[i];
    } else {
      usage(argv[0]);
      return 2;
    }
  }

  if (!dataset_dir || threads < 1 || repeat < 1) {
    usage(argv[0]);
    return 2;
  }

  num_names = list_dir_sorted(dataset_dir, &names);
  if (num_names < 0) {
    fprintf(stderr, "%s: can't open directory %s\n", argv[0], dataset_dir);
    return 2;
  }

  strips = (jpar_strip *)malloc(sizeof(jpar_strip) * (size_t)threads);
  jpeg_bufs = (unsigned char **)malloc(sizeof(unsigned char *) * (size_t)threads);
  jpeg_sizes = (unsigned long *)malloc(sizeof(unsigned long) * (size_t)threads);
  if (!strips || !jpeg_bufs || !jpeg_sizes) {
    fprintf(stderr, "%s: out of memory\n", argv[0]);
    return 2;
  }

  printf("image,width,height,threads,repeats,mean_time_s,min_time_s\n");

  for (i = 0; i < num_names; i++) {
    char path[4096];
    loaded_image img;
    int max_v_samp_factor;
    int rep, num_ok = 0;
    double sum_time = 0.0, min_time = 0.0;

    snprintf(path, sizeof(path), "%s/%s", dataset_dir, names[i]);
    if (!load_image(path, names[i], &img))
      continue;

    max_v_samp_factor = get_max_v_samp_factor(img.color_space, img.components);

    for (rep = 0; rep < repeat; rep++) {
      struct timespec t0, t1;
      int num_strips = 0, s;
      boolean ok;

      clock_gettime(CLOCK_MONOTONIC, &t0);
      ok = jpar_encode_strips_parallel(img.rows, img.width, img.height,
                                        max_v_samp_factor, img.components,
                                        img.color_space, quality, threads,
                                        strips, jpeg_bufs, jpeg_sizes,
                                        &num_strips);
      clock_gettime(CLOCK_MONOTONIC, &t1);

      for (s = 0; s < num_strips; s++)
        if (jpeg_bufs[s])
          free(jpeg_bufs[s]);

      if (!ok) {
        fprintf(stderr, "warning: %s: encode failed on repeat %d\n",
                names[i], rep);
        continue;
      }

      {
        double dt = timespec_diff(t0, t1);

        sum_time += dt;
        if (num_ok == 0 || dt < min_time)
          min_time = dt;
        num_ok++;
      }
    }

    free_loaded_image(&img);

    if (num_ok == 0) {
      fprintf(stderr, "warning: %s: every repeat failed to encode, skipping\n",
              names[i]);
      continue;
    }

    printf("%s,%u,%u,%d,%d,%.6f,%.6f\n", names[i], (unsigned)img.width,
           (unsigned)img.height, threads, num_ok, sum_time / num_ok, min_time);

    images_reported++;
    total_encoded += num_ok;
    total_wall_time += sum_time;
  }

  for (i = 0; i < num_names; i++)
    free(names[i]);
  free(names);
  free(strips);
  free(jpeg_bufs);
  free(jpeg_sizes);

  printf("# TOTAL threads=%d images=%d repeats=%d encoded=%d wall_time=%.3fs\n",
         threads, images_reported, repeat, total_encoded, total_wall_time);

  if (images_reported == 0) {
    fprintf(stderr, "%s: no supported images found/decoded in %s\n", argv[0],
            dataset_dir);
    return 2;
  }
  return 0;
}
