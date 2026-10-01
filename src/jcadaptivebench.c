/*
 * jcadaptivebench.c
 *
 * Measurement driver for the adaptive-core encoder (jcadaptive.c), built to
 * answer two questions over a directory of real images:
 *
 *   1. Energy per image for three core-allocation policies -- always 2
 *      cores, always 4 cores, and adaptive (classifier picks 2 or 4).
 *   2. Classifier overhead -- how much of an adaptive encode's time goes to
 *      feature extraction (jcfeatures.c) plus inference (complexity_model.c)
 *      rather than to the encode itself.
 *
 * This tool does not read any power sensor itself.  It decodes each image
 * once, then runs each policy back-to-back in its own "phase", repeating
 * the policy's work until the phase has lasted at least --min-seconds, and
 * prints one CSV row per phase carrying the phase's start/end timestamps
 * (CLOCK_MONOTONIC, absolute seconds) and repeat count.  A separate power
 * logger (scripts/measure/power_logger.py) samples the board's power rail
 * against the same clock, and scripts/measure/analyze_energy.py integrates
 * those samples over each phase's [t_start_s, t_end_s] window and divides
 * by the repeat count to get energy per image.
 *
 * Why phases are stretched to --min-seconds rather than run once: a single
 * encode takes tens of milliseconds, while a Jetson Nano's on-board INA3221
 * power monitor only produces a fresh reading every few milliseconds at
 * best.  Integrating over one encode would rest on a handful of samples;
 * integrating over a couple of seconds of identical repeated work does not.
 *
 * Phases per image:
 *   classify   feature extraction + inference only, no encode
 *   always2    jpar_encode_strips_spliced() with 2 threads
 *   always4    jpar_encode_strips_spliced() with 4 threads
 *   adaptive   feature extraction + inference + encode at the predicted
 *              core count -- exactly what jcadaptive.c does per image,
 *              minus image decode and file output
 * plus an "idle" phase (the process just sleeps) at the start and end of
 * the run, giving the analysis a baseline power to subtract.
 *
 * Image decode and file I/O are never inside a phase, matching
 * jcparallelbench.c.  The three encode phases run in a rotating order from
 * one image to the next, so no policy always inherits the same thermal /
 * frequency-scaling state from the phase before it.
 *
 * feature_s/inference_s/encode_s are the wall-clock time spent in each
 * stage summed over the phase's repeats; they give the classifier-overhead
 * numbers directly and need no power data (so that part also works on a
 * machine with no power sensor).
 *
 * POSIX only (dirent.h, nanosleep) -- its target is the Jetson Nano.
 *
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 */

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "jcfeatures.h"
#include "jcimageload.h"
#include "jcmodel.h"
#include "jcparallel.h"
#include "jpeglib.h"

#define DEFAULT_QUALITY       85
#define DEFAULT_MIN_SECONDS   2.0
#define DEFAULT_MIN_REPS      3
#define DEFAULT_IDLE_SECONDS  5.0
#define DEFAULT_SETTLE_MS     200

/* Must match jcadaptive.c's policy. */
#define SIMPLE_THREADS   2
#define COMPLEX_THREADS  4

typedef enum {
  PHASE_CLASSIFY,
  PHASE_ALWAYS2,
  PHASE_ALWAYS4,
  PHASE_ADAPTIVE
} phase_kind;

static const char *const phase_names[] = {
  "classify", "always2", "always4", "adaptive"
};

typedef struct {
  int reps;
  int threads;              /* thread count used by the last repeat */
  double t_start, t_end;
  double feature_s, inference_s, encode_s;
  int predicted_label;      /* 0 = simple, 1 = complex, -1 = not classified */
  double p_complex;
  unsigned long jpeg_bytes;
} phase_result;


static double now_s(void) {
  struct timespec t;

  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

static void sleep_s(double seconds) {
  struct timespec ts;

  if (seconds <= 0.0)
    return;
  ts.tv_sec = (time_t)seconds;
  ts.tv_nsec = (long)((seconds - (double)ts.tv_sec) * 1e9);
  nanosleep(&ts, NULL);
}

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

static unsigned long file_size(const char *path) {
  struct stat st;

  if (stat(path, &st) != 0)
    return 0;
  return (unsigned long)st.st_size;
}

/* Repeats one policy's per-image work until the phase has lasted at least
 * min_seconds and min_reps repeats.  Returns FALSE if an encode fails. */
static boolean run_phase(phase_kind kind, const jcil_loaded_image *img,
                         unsigned long file_bytes, int max_v_samp_factor,
                         int quality, int feature_downsample,
                         double min_seconds, int min_reps, phase_result *res) {
  memset(res, 0, sizeof(*res));
  res->predicted_label = -1;
  res->threads = (kind == PHASE_ALWAYS2) ? 2 : (kind == PHASE_ALWAYS4) ? 4 : 1;

  res->t_start = now_s();
  do {
    double ta, tb, tc;

    if (kind == PHASE_CLASSIFY || kind == PHASE_ADAPTIVE) {
      double features[JCFEAT_NUM_FEATURES], proba[2];

      ta = now_s();
      jcfeat_extract_downsampled(img->rows, img->width, img->height,
                                 img->components, file_bytes,
                                 feature_downsample, features);
      tb = now_s();
      score(features, proba);
      tc = now_s();

      res->feature_s += tb - ta;
      res->inference_s += tc - tb;
      res->predicted_label = (proba[1] > proba[0]) ? 1 : 0;
      res->p_complex = proba[1];
      if (kind == PHASE_ADAPTIVE)
        res->threads =
            res->predicted_label == 0 ? SIMPLE_THREADS : COMPLEX_THREADS;
    }

    if (kind != PHASE_CLASSIFY) {
      unsigned char *jpeg_buf = NULL;
      unsigned long jpeg_size = 0;
      int num_strips = 0;
      boolean ok;

      ta = now_s();
      ok = jpar_encode_strips_spliced(
          img->rows, img->width, img->height, max_v_samp_factor,
          img->components, img->color_space, quality, res->threads, &jpeg_buf,
          &jpeg_size, &num_strips);
      tb = now_s();

      free(jpeg_buf);
      if (!ok)
        return FALSE;
      res->encode_s += tb - ta;
      res->jpeg_bytes = jpeg_size;
    }

    res->reps++;
    res->t_end = now_s();
  } while (res->t_end - res->t_start < min_seconds || res->reps < min_reps);

  return TRUE;
}

static void print_idle_row(double seconds) {
  double t0 = now_s(), t1;

  sleep_s(seconds);
  t1 = now_s();
  printf("-,0,0,idle,0,1,%.6f,%.6f,%.6f,0.000000,0.000000,0.000000,-1,"
         "0.0000,0\n", t0, t1, t1 - t0);
  fflush(stdout);
}

static void usage(const char *progname) {
  fprintf(stderr,
          "Usage: %s <dataset_dir> [--quality Q] [--min-seconds S]\n"
          "          [--min-reps R] [--idle-seconds I] [--settle-ms M] "
          "[--limit N]\n"
          "          [--feature-downsample D]\n"
          "  --quality Q       JPEG quality, 0-100 (default %d)\n"
          "  --min-seconds S   minimum duration of each phase (default %.1f)\n"
          "  --min-reps R      minimum repeats per phase (default %d)\n"
          "  --idle-seconds I  idle baseline at start and end of the run "
          "(default %.1f)\n"
          "  --settle-ms M     pause between phases (default %d)\n"
          "  --limit N         stop after N images (default: all)\n"
          "  --feature-downsample D\n"
          "                    extract classifier features from every D-th "
          "pixel of\n"
          "                    every D-th row (default %d, what the shipped "
          "model was\n"
          "                    trained on; other values only time the "
          "extraction)\n",
          progname, DEFAULT_QUALITY, DEFAULT_MIN_SECONDS, DEFAULT_MIN_REPS,
          DEFAULT_IDLE_SECONDS, DEFAULT_SETTLE_MS, JCFEAT_MODEL_DOWNSAMPLE);
}

int main(int argc, char **argv) {
  /* always2/always4/adaptive, rotated by image index (see file header). */
  static const phase_kind encode_phases[3] = {
    PHASE_ALWAYS2, PHASE_ALWAYS4, PHASE_ADAPTIVE
  };
  const char *dataset_dir = NULL;
  int quality = DEFAULT_QUALITY, min_reps = DEFAULT_MIN_REPS;
  int settle_ms = DEFAULT_SETTLE_MS, limit = -1;
  int feature_downsample = JCFEAT_MODEL_DOWNSAMPLE;
  double min_seconds = DEFAULT_MIN_SECONDS, idle_seconds = DEFAULT_IDLE_SECONDS;
  char **names;
  int num_names, i, images_reported = 0;

  for (i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--quality") == 0 && i + 1 < argc) {
      quality = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--min-seconds") == 0 && i + 1 < argc) {
      min_seconds = atof(argv[++i]);
    } else if (strcmp(argv[i], "--min-reps") == 0 && i + 1 < argc) {
      min_reps = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--idle-seconds") == 0 && i + 1 < argc) {
      idle_seconds = atof(argv[++i]);
    } else if (strcmp(argv[i], "--settle-ms") == 0 && i + 1 < argc) {
      settle_ms = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--limit") == 0 && i + 1 < argc) {
      limit = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--feature-downsample") == 0 && i + 1 < argc) {
      feature_downsample = atoi(argv[++i]);
    } else if (argv[i][0] != '-' && !dataset_dir) {
      dataset_dir = argv[i];
    } else {
      usage(argv[0]);
      return 2;
    }
  }

  if (!dataset_dir || min_reps < 1 || min_seconds < 0.0) {
    usage(argv[0]);
    return 2;
  }

  num_names = list_dir_sorted(dataset_dir, &names);
  if (num_names < 0) {
    fprintf(stderr, "%s: can't open directory %s\n", argv[0], dataset_dir);
    return 2;
  }

  printf("image,width,height,phase,threads,reps,t_start_s,t_end_s,elapsed_s,"
         "feature_s,inference_s,encode_s,predicted_label,p_complex,"
         "jpeg_bytes\n");

  print_idle_row(idle_seconds);

  for (i = 0; i < num_names; i++) {
    char path[4096];
    jcil_loaded_image img;
    phase_kind order[4];
    int max_v_samp_factor, p;

    if (limit >= 0 && images_reported >= limit)
      break;

    snprintf(path, sizeof(path), "%s/%s", dataset_dir, names[i]);
    if (!jcil_load_image(path, &img))
      continue;

    max_v_samp_factor =
        jcil_get_max_v_samp_factor(img.color_space, img.components);

    order[0] = PHASE_CLASSIFY;
    for (p = 0; p < 3; p++)
      order[1 + p] = encode_phases[(images_reported + p) % 3];

    for (p = 0; p < 4; p++) {
      phase_result res;

      sleep_s(settle_ms / 1000.0);
      if (!run_phase(order[p], &img, file_size(path), max_v_samp_factor,
                     quality, feature_downsample, min_seconds, min_reps,
                     &res)) {
        fprintf(stderr, "warning: %s: %s encode failed, skipping phase\n",
                names[i], phase_names[order[p]]);
        continue;
      }

      printf("%s,%u,%u,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.4f,%lu\n",
             names[i], (unsigned)img.width, (unsigned)img.height,
             phase_names[order[p]], res.threads, res.reps, res.t_start,
             res.t_end, res.t_end - res.t_start, res.feature_s,
             res.inference_s, res.encode_s, res.predicted_label,
             res.p_complex, res.jpeg_bytes);
      fflush(stdout);
    }

    jcil_free_loaded_image(&img);
    images_reported++;
    fprintf(stderr, "[%d] %s done\n", images_reported, names[i]);
  }

  sleep_s(settle_ms / 1000.0);
  print_idle_row(idle_seconds);

  for (i = 0; i < num_names; i++)
    free(names[i]);
  free(names);

  if (images_reported == 0) {
    fprintf(stderr, "%s: no supported images found/decoded in %s\n", argv[0],
            dataset_dir);
    return 2;
  }
  return 0;
}
