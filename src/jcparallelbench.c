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
 * jpeg_compress_struct API is 8-bit only), BMP, PNG, and PPM/PGM.
 * BMP/PNG/PPM/PGM are decoded by reusing cjpeg's own existing, tested
 * reader modules (jinit_read_bmp()/jinit_read_png()/jinit_read_ppm() from
 * rdbmp.c/rdpng.c/rdppm.c, declared in cdjpeg.h) rather than
 * reimplementing those formats.  Any other extension is silently skipped,
 * so pointing this at a real-world photo folder that also contains stray
 * non-image files is safe.
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
#include <sys/stat.h>
#include <time.h>

#include "jcimageload.h"
#include "jcparallel.h"
#include "jpeglib.h"

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


static int compare_names(const void *a, const void *b) {
  return strcmp(*(const char *const *)a, *(const char *const *)b);
}

/* Lists `dir`'s regular files, sorted, into *names_out (caller frees each
 * entry and the array).  Returns the count, or -1 on failure to open. */
static int list_dir_sorted(const char *dir, char ***names_out) {
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

static double timespec_diff(struct timespec t0, struct timespec t1) {
  return (double)(t1.tv_sec - t0.tv_sec) +
         (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
}

static void usage(const char *progname) {
  fprintf(stderr,
          "Usage: %s <dataset_dir> --threads N [--quality Q] [--repeat R]\n"
          "  --threads N   number of strips/OpenMP threads to encode with "
          "(required)\n"
          "  --quality Q   JPEG quality, 0-100 (default %d)\n"
          "  --repeat R    encode each image R times and report mean/min "
          "(default %d)\n",
          progname, DEFAULT_QUALITY, DEFAULT_REPEAT);
}

int main(int argc, char **argv) {
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
  jpeg_bufs =
      (unsigned char **)malloc(sizeof(unsigned char *) * (size_t)threads);
  jpeg_sizes = (unsigned long *)malloc(sizeof(unsigned long) * (size_t)threads);
  if (!strips || !jpeg_bufs || !jpeg_sizes) {
    fprintf(stderr, "%s: out of memory\n", argv[0]);
    return 2;
  }

  printf("image,width,height,threads,repeats,mean_time_s,min_time_s\n");

  for (i = 0; i < num_names; i++) {
    char path[4096];
    jcil_loaded_image img;
    int max_v_samp_factor;
    int rep, num_ok = 0;
    double sum_time = 0.0, min_time = 0.0;

    snprintf(path, sizeof(path), "%s/%s", dataset_dir, names[i]);
    if (!jcil_load_image(path, &img)) {
      fprintf(stderr, "failed to load image %s\n", path);
      continue;
    }

    max_v_samp_factor = jcil_get_max_v_samp_factor(img.color_space, img.components);

    for (rep = 0; rep < repeat; rep++) {
      struct timespec t0, t1;
      int num_strips = 0, s;
      boolean ok;

      clock_gettime(CLOCK_MONOTONIC, &t0);
      ok = jpar_encode_strips_parallel(
          img.rows, img.width, img.height, max_v_samp_factor, img.components,
          img.color_space, quality, threads, strips, jpeg_bufs, jpeg_sizes,
          &num_strips);
      clock_gettime(CLOCK_MONOTONIC, &t1);

      for (s = 0; s < num_strips; s++)
        if (jpeg_bufs[s])
          free(jpeg_bufs[s]);

      if (!ok) {
        fprintf(stderr, "warning: %s: encode failed on repeat %d\n", names[i],
                rep);
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

    jcil_free_loaded_image(&img);

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
