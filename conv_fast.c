#define _GNU_SOURCE
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#ifndef OUT_NCHW
#define OUT_NCHW 0
#endif
#ifndef UW
#define UW 12                      // pixels per register block (accumulators)
#endif
#ifndef WREG
#define WREG 0                     // weight vectors held in registers (0 = all 3 folded into the FMA as memory operands)
#endif

enum { N = 16, C = 3, H = 128, W = 128, K = 8, R = 3, S = 3, PAD = 1 };
enum { OH = H, OW = W };                                            // F2 with s1 p1 k3
enum { NBLK = OW / UW, TAIL = OW - NBLK * UW };                     // 12: 10 blocks + 8 tail

_Static_assert(K == 8, "vectorised over output channels: K must be 8");
_Static_assert(UW == 8 || UW == 12, "UW must be 8 or 12");
_Static_assert(TAIL == 0 || TAIL == 8, "tail block must be 8 pixels");

// ------------------------------------------------------------------ helpers
static void *big_alloc(size_t bytes) {                              // 2 MB aligned, THP hint
    const size_t A = (size_t)2 << 20;
    bytes = (bytes + A - 1) & ~(A - 1);
    void *p = aligned_alloc(A, bytes);
    if (!p) { perror("aligned_alloc"); exit(1); }
#ifdef MADV_HUGEPAGE
    madvise(p, bytes, MADV_HUGEPAGE);
#endif
    memset(p, 0, bytes);                                            // pre-fault outside timing
    return p;
}
static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static int load_bin(const char *path, float *dst, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 0; }
    size_t got = fread(dst, sizeof(float), n, f);
    fclose(f);
    if (got != n) { fprintf(stderr, "%s: expected %zu floats, got %zu\n", path, n, got); return 0; }
    return 1;
}

// ------------------------------------------------------------------ reference (your loop)
__attribute__((noinline))
static void conv2d_reference(const float *in, const float *wt, const float *bias, float *y) {
    for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++)
    for (int p = 0; p < OH; p++)
    for (int q = 0; q < OW; q++) {
        float sum = bias[k];
        for (int c = 0; c < C; c++)
        for (int r = 0; r < R; r++)
        for (int s = 0; s < S; s++) {
            int ih = p - PAD + r, iw = q - PAD + s;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                sum = fmaf(in[((n * C + c) * H + ih) * W + iw],
                           wt[((k * C + c) * R + r) * S + s], sum);
        }
        y[((n * K + k) * OH + p) * OW + q] = sum;
    }
}

// wt[k][c][r][s] -> wv[c][r][s][k] : 27 ymm = 864 B, stays in L0
static void pack_weights(const float *wt, float *wv) {
    for (int c = 0; c < C; c++) for (int r = 0; r < R; r++)
    for (int s = 0; s < S; s++) for (int k = 0; k < K; k++)
        wv[((c * R + r) * S + s) * K + k] = wt[((k * C + c) * R + r) * S + s];
}

// ------------------------------------------------------------------ NCHW store helpers
#if OUT_NCHW
static inline __attribute__((always_inline)) void transpose8(__m256 *v) {
    __m256 t0 = _mm256_unpacklo_ps(v[0], v[1]), t1 = _mm256_unpackhi_ps(v[0], v[1]);
    __m256 t2 = _mm256_unpacklo_ps(v[2], v[3]), t3 = _mm256_unpackhi_ps(v[2], v[3]);
    __m256 t4 = _mm256_unpacklo_ps(v[4], v[5]), t5 = _mm256_unpackhi_ps(v[4], v[5]);
    __m256 t6 = _mm256_unpacklo_ps(v[6], v[7]), t7 = _mm256_unpackhi_ps(v[6], v[7]);
    __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44), u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44), u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44), u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    __m256 u6 = _mm256_shuffle_ps(t5, t7, 0x44), u7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    v[0] = _mm256_permute2f128_ps(u0, u4, 0x20); v[1] = _mm256_permute2f128_ps(u1, u5, 0x20);
    v[2] = _mm256_permute2f128_ps(u2, u6, 0x20); v[3] = _mm256_permute2f128_ps(u3, u7, 0x20);
    v[4] = _mm256_permute2f128_ps(u0, u4, 0x31); v[5] = _mm256_permute2f128_ps(u1, u5, 0x31);
    v[6] = _mm256_permute2f128_ps(u2, u6, 0x31); v[7] = _mm256_permute2f128_ps(u3, u7, 0x31);
}
#endif

// ------------------------------------------------------------------ microkernel
// Output row p, pixels q0 .. q0+NW-1, all 8 k.  All arguments after q0 are compile-time.
//   RLO/RHI : kernel rows that exist (top row p=0 -> RLO=1, bottom row p=127 -> RHI=1)
//   LEFT    : q0 == 0     -> input column -1 is padding: never loaded, never multiplied
//   RIGHT   : q0+NW == 128 -> input column 128 is padding: same
static inline __attribute__((always_inline))
void ublock(const float *restrict img, const float *restrict wv, __m256 vb,
            float *restrict yn, int p, int q0,
            const int NW, const int RLO, const int RHI, const int LEFT, const int RIGHT) {
    __m256 acc[12];
#pragma GCC unroll 12
    for (int j = 0; j < NW; j++) acc[j] = vb;
    // Stop GCC from hoisting the 27 weight vectors out of the block loop: it has no 27
    // spare registers, so it would copy them to the stack and reload them (extra stores).
    // Reading them straight from wv (864 B, always in L0) is the same load, minus the copy.
    __asm__ volatile("" : "+r"(wv));

#pragma GCC unroll 3
    for (int c = 0; c < C; c++) {
#pragma GCC unroll 3
        for (int r = RLO; r <= RHI; r++) {
            const float *row = img + ((size_t)c * H + (p - PAD + r)) * W + (q0 - PAD);
            const float *w = wv + (size_t)((c * R + r) * S) * K;
#if WREG >= 1
            const __m256 w0 = _mm256_load_ps(w);
#define W0 w0
#else
#define W0 _mm256_load_ps(w)
#endif
#if WREG >= 2
            const __m256 w1 = _mm256_load_ps(w + K);
#define W1 w1
#else
#define W1 _mm256_load_ps(w + K)
#endif
#if WREG >= 3
            const __m256 w2 = _mm256_load_ps(w + 2 * K);
#define W2 w2
#else
#define W2 _mm256_load_ps(w + 2 * K)
#endif
#pragma GCC unroll 14
            for (int t = 0; t < NW + 2; t++) {                  // input column q0-1+t
                if (LEFT && t == 0) continue;
                if (RIGHT && t == NW + 1) continue;
                const __m256 x = _mm256_broadcast_ss(row + t);
                // pixel j receives s=0 at t=j, s=1 at t=j+1, s=2 at t=j+2 -> order s=0,1,2 kept
                if (t - 2 >= 0 && t - 2 < NW) acc[t - 2] = _mm256_fmadd_ps(x, W2, acc[t - 2]);
                if (t - 1 >= 0 && t - 1 < NW) acc[t - 1] = _mm256_fmadd_ps(x, W1, acc[t - 1]);
                if (t < NW)                   acc[t]     = _mm256_fmadd_ps(x, W0, acc[t]);
            }
#undef W0
#undef W1
#undef W2
        }
    }

#if OUT_NCHW
    float *o = yn + (size_t)p * OW + q0;                        // y[n][k][p][q]
    transpose8(acc);
    for (int k = 0; k < K; k++) _mm256_storeu_ps(o + (size_t)k * OH * OW, acc[k]);
    if (NW == 12) {                                             // pixels 8..11: two 4x4 transposes
        __m128 l0 = _mm256_castps256_ps128(acc[8]),  l1 = _mm256_castps256_ps128(acc[9]);
        __m128 l2 = _mm256_castps256_ps128(acc[10]), l3 = _mm256_castps256_ps128(acc[11]);
        __m128 h0 = _mm256_extractf128_ps(acc[8], 1),  h1 = _mm256_extractf128_ps(acc[9], 1);
        __m128 h2 = _mm256_extractf128_ps(acc[10], 1), h3 = _mm256_extractf128_ps(acc[11], 1);
        _MM_TRANSPOSE4_PS(l0, l1, l2, l3);
        _MM_TRANSPOSE4_PS(h0, h1, h2, h3);
        float *o8 = o + 8;
        _mm_storeu_ps(o8 + 0 * OH * OW, l0); _mm_storeu_ps(o8 + 1 * OH * OW, l1);
        _mm_storeu_ps(o8 + 2 * OH * OW, l2); _mm_storeu_ps(o8 + 3 * OH * OW, l3);
        _mm_storeu_ps(o8 + 4 * OH * OW, h0); _mm_storeu_ps(o8 + 5 * OH * OW, h1);
        _mm_storeu_ps(o8 + 6 * OH * OW, h2); _mm_storeu_ps(o8 + 7 * OH * OW, h3);
    }
#else
    float *o = yn + ((size_t)p * OW + q0) * K;                  // y[n][p][q][8k], contiguous run
#pragma GCC unroll 12
    for (int j = 0; j < NW; j++) _mm256_store_ps(o + j * K, acc[j]);
#endif
}

// One output row: left block, inner blocks, right block (tail).
#define ROW(RLO, RHI)                                                                         \
    do {                                                                                      \
        ublock(img, wv, vb, yn, p, 0, UW, RLO, RHI, 1, TAIL == 0 && NBLK == 1);               \
        for (int b = 1; b < (TAIL ? NBLK : NBLK - 1); b++)                                    \
            ublock(img, wv, vb, yn, p, b * UW, UW, RLO, RHI, 0, 0);                           \
        if (TAIL) ublock(img, wv, vb, yn, p, NBLK * UW, 8, RLO, RHI, 0, 1);                   \
        else      ublock(img, wv, vb, yn, p, (NBLK - 1) * UW, UW, RLO, RHI, 0, 1);            \
    } while (0)

static void conv_image(const float *restrict img, const float *restrict wv, __m256 vb,
                       float *restrict yn) {
    int p = 0;
    ROW(1, 2);                                                  // top row: kernel row 0 is padding
    for (p = 1; p < OH - 1; p++) ROW(0, 2);                     // 126 interior rows
    p = OH - 1;
    ROW(0, 1);                                                  // bottom row: kernel row 2 is padding
}

// ------------------------------------------------------------------ threads + timing
typedef struct {
    int tid, nthreads, cpu, reps;
    const float *in, *wv, *bias;
    float *y;
    pthread_barrier_t *bar;
} Job;

static void *worker(void *arg) {
    Job *j = arg;
    if (j->cpu >= 0) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(j->cpu, &set);
        if (pthread_setaffinity_np(pthread_self(), sizeof set, &set))
            fprintf(stderr, "warn: could not pin thread %d to cpu %d\n", j->tid, j->cpu);
    }
    const __m256 vb = _mm256_loadu_ps(j->bias);
    const int n0 = (N * j->tid) / j->nthreads, n1 = (N * (j->tid + 1)) / j->nthreads;
    for (int rep = 0; rep < j->reps; rep++) {
        pthread_barrier_wait(j->bar);
        for (int n = n0; n < n1; n++)
            conv_image(j->in + (size_t)n * C * H * W, j->wv, vb, j->y + (size_t)n * K * OH * OW);
        pthread_barrier_wait(j->bar);
    }
    return NULL;
}

static int read_pcores(int *cpus, int max) {
    FILE *f = fopen("/sys/devices/cpu_core/cpus", "r");
    int n = 0;
    if (f) {
        int a, b; char sep = 0;
        while (n < max && fscanf(f, "%d", &a) == 1) {
            b = a;
            if (fscanf(f, "%c", &sep) == 1 && sep == '-') {
                if (fscanf(f, "%d", &b) != 1) break;
                if (fscanf(f, "%c", &sep) != 1) sep = 0;
            }
            for (int c = a; c <= b && n < max; c++) cpus[n++] = c;
            if (sep != ',') break;
        }
        fclose(f);
    }
    return n;
}
static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    int T = 1, reps = 200, cpu_req = -1;
    double ghz = 3.7;
    const char *fin = NULL, *fw = NULL, *fb = NULL, *fout = NULL;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-t") && i + 1 < argc) T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-g") && i + 1 < argc) ghz = atof(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) cpu_req = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) fin = argv[++i];
        else if (!strcmp(argv[i], "-w") && i + 1 < argc) fw = argv[++i];
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) fb = argv[++i];
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) fout = argv[++i];
        else { fprintf(stderr, "usage: %s [-t threads] [-c cpu] [-r reps] [-g GHz] [-i in -w w -b b] [-o y.bin]\n", argv[0]); return 1; }
    }
    if (T < 1 || T > N) { fprintf(stderr, "threads must be 1..%d\n", N); return 1; }

    const size_t nin = (size_t)N * C * H * W, nw = K * C * R * S, ny = (size_t)N * K * OH * OW;
    float *in = big_alloc(sizeof(float) * nin), *y = big_alloc(sizeof(float) * ny);
    float *wt = aligned_alloc(64, 1024), *wv = aligned_alloc(64, 1024), *bias = aligned_alloc(64, 64);
    float *yref = malloc(sizeof(float) * ny);

    srand(12345);
    if (fin) { if (!load_bin(fin, in, nin)) return 1; }
    else for (size_t i = 0; i < nin; i++) in[i] = (float)rand() / RAND_MAX * 2.f - 1.f;
    if (fw)  { if (!load_bin(fw, wt, nw)) return 1; }
    else for (size_t i = 0; i < nw; i++) wt[i] = (float)rand() / RAND_MAX * 2.f - 1.f;
    if (fb)  { if (!load_bin(fb, bias, K)) return 1; }
    else for (int i = 0; i < K; i++) bias[i] = (float)rand() / RAND_MAX * 0.2f - 0.1f;
    pack_weights(wt, wv);

    int pc[64], npc = read_pcores(pc, 64);
    pthread_barrier_t bar;
    pthread_barrier_init(&bar, NULL, T + 1);
    pthread_t th[16]; Job jobs[16];
    for (int t = 0; t < T; t++) {
        int cpu = (cpu_req >= 0 && T == 1) ? cpu_req : (npc ? pc[t % npc] : t);
        jobs[t] = (Job){ .tid = t, .nthreads = T, .cpu = cpu, .reps = reps,
                         .in = in, .wv = wv, .bias = bias, .y = y, .bar = &bar };
        pthread_create(&th[t], NULL, worker, &jobs[t]);
    }
    double *ts = malloc(sizeof(double) * reps);
    for (int rep = 0; rep < reps; rep++) {
        double t0 = now_s();
        pthread_barrier_wait(&bar);
        pthread_barrier_wait(&bar);
        ts[rep] = now_s() - t0;
    }
    for (int t = 0; t < T; t++) pthread_join(th[t], NULL);
    pthread_barrier_destroy(&bar);

    // ---- correctness: bit-exact against the reference loop
    conv2d_reference(in, wt, bias, yref);
    size_t bad = 0, first = 0; double maxabs = 0;
    for (int n = 0; n < N; n++) for (int k = 0; k < K; k++)
    for (int p = 0; p < OH; p++) for (int q = 0; q < OW; q++) {
        size_t ir = (((size_t)n * K + k) * OH + p) * OW + q;
        size_t iy = OUT_NCHW ? ir : (((size_t)n * OH + p) * OW + q) * K + k;
        double d = fabs((double)y[iy] - yref[ir]);
        if (y[iy] != yref[ir]) { if (!bad) first = ir; bad++; }
        if (d > maxabs) maxabs = d;
    }

    int skip = reps / 10, m = reps - skip;
    qsort(ts + skip, m, sizeof(double), cmp_d);
    double tmin = ts[skip], tmed = ts[skip + m / 2];
    const double fma_instr = (double)N * C * (3.0 * OH - 2) * (3.0 * OW - 2);   // F11: 7,004,352
    const double floor_cyc = fma_instr / 2.0 / T;
    printf("conv_fast: N=%d C=%d %dx%d K=%d 3x3 s1 p1 | UW=%d accumulators | out=%s\n",
           N, C, H, W, K, UW, OUT_NCHW ? "NCHW" : "nChw8c");
    printf("threads=%d cpus:", T);
    for (int t = 0; t < T; t++) printf(" %d", jobs[t].cpu);
    printf("\n");
    printf("time   min %.3f ms   median %.3f ms\n", tmin * 1e3, tmed * 1e3);
    printf("cycles at %.2f GHz: median %.2f M   floor %.2f M   -> %.1f%% of FMA peak\n",
           ghz, tmed * ghz * 1e3, floor_cyc * 1e-6, 100.0 * floor_cyc / (tmed * ghz * 1e9));
    printf("vs oneDNN preferred 1.226 ms (at 3.7 GHz): %.2fx   vs oneDNN NCHW 3.669 ms: %.2fx  (valid only at -g 3.7 on the 270K Plus)\n",
           1.226e-3 / tmed, 3.669e-3 / tmed);
    if (!bad) printf("verdict BIT-EXACT vs reference\n");
    else printf("verdict MISMATCH: %zu elements, first at %zu, max abs err %.3g\n", bad, first, maxabs);

    if (fout) { FILE *f = fopen(fout, "wb"); if (f) { fwrite(y, sizeof(float), ny, f); fclose(f); } }
    return bad ? 2 : 0;
}
