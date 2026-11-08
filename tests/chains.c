// How many independent FMA chains does a kernel need to saturate the 2 FMA/cycle issue rate?
//
// This is the explanation for why a conv kernel lands at a given "% of peak": a 256-bit FMA has
// ~4 cycles latency but 2/cycle issue, so sustained FMAs/cycle = min(2, chains/latency).
//   chains=4 -> 1.0 FMA/cyc (50% of peak)
//   chains=6 -> 1.5 FMA/cyc (75%)   <-- ggml's conv block uses 6 t-accumulators
//   chains=8 -> 2.0 FMA/cyc (100%)
//
// All accumulators are EXPLICIT NAMED VARIABLES (never an array), which is the only way to be
// sure gcc keeps them in ymm registers; an earlier version of this probe used __m256 acc[N] and
// measured 1.9 GMAC/s because the array stayed in memory.
//
//   cc -O3 -mavx2 -mfma -fopenmp chains.c -o chains ; ./chains [threads]
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

#define ITERS 2000000
#define MULV _mm256_set1_ps(0.9999999f)
#define ADDV _mm256_set1_ps(1.0000001f)

static double bench(int nacc, int th);

#define BODY(N, ...)  /* N = accumulator chains */                              \
    double best = 1e9;                                                          \
    for (int rep = 0; rep < 3; rep++) {                                          \
        const double t0 = now_s();                                              \
        _Pragma("omp parallel num_threads(th)")                                  \
        {                                                                        \
            const __m256 m = MULV, c = ADDV;                                     \
            __VA_ARGS__;                                                         \
            for (long it = 0; it < ITERS; it++) {                                \
                X##N##_STEP                                                      \
            }                                                                    \
            __m256 s = X##N##_SUM;                                               \
            float tmp[8];                                                        \
            _mm256_storeu_ps(tmp, s);                                            \
            volatile float sink = tmp[0] + tmp[7];                               \
            (void) sink;                                                         \
        }                                                                        \
        const double dt = now_s() - t0;                                          \
        if (dt < best) best = dt;                                                \
    }                                                                            \
    return (double) N * ITERS * th / best / 1e9;   /* GFMA/s (GMAC/s) */

// 4 chains
#define X4_DECL __m256 a0 = c, a1 = c, a2 = c, a3 = c;
#define X4_STEP a0 = _mm256_fmadd_ps(a0, m, c); a1 = _mm256_fmadd_ps(a1, m, c); \
                a2 = _mm256_fmadd_ps(a2, m, c); a3 = _mm256_fmadd_ps(a3, m, c);
#define X4_SUM  _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3))
static double b4(int th) { BODY(4, X4_DECL) }

// 6 chains
#define X6_DECL __m256 a0 = c, a1 = c, a2 = c, a3 = c, a4 = c, a5 = c;
#define X6_STEP a0 = _mm256_fmadd_ps(a0, m, c); a1 = _mm256_fmadd_ps(a1, m, c); \
                a2 = _mm256_fmadd_ps(a2, m, c); a3 = _mm256_fmadd_ps(a3, m, c); \
                a4 = _mm256_fmadd_ps(a4, m, c); a5 = _mm256_fmadd_ps(a5, m, c);
#define X6_SUM  _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)), _mm256_add_ps(a4, a5))
static double b6(int th) { BODY(6, X6_DECL) }

// 8 chains
#define X8_DECL __m256 a0 = c, a1 = c, a2 = c, a3 = c, a4 = c, a5 = c, a6 = c, a7 = c;
#define X8_STEP a0 = _mm256_fmadd_ps(a0, m, c); a1 = _mm256_fmadd_ps(a1, m, c); \
                a2 = _mm256_fmadd_ps(a2, m, c); a3 = _mm256_fmadd_ps(a3, m, c); \
                a4 = _mm256_fmadd_ps(a4, m, c); a5 = _mm256_fmadd_ps(a5, m, c); \
                a6 = _mm256_fmadd_ps(a6, m, c); a7 = _mm256_fmadd_ps(a7, m, c);
#define X8_SUM  _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)), \
                              _mm256_add_ps(_mm256_add_ps(a4, a5), _mm256_add_ps(a6, a7)))
static double b8(int th) { BODY(8, X8_DECL) }

// 12 chains
#define X12_DECL __m256 a0 = c, a1 = c, a2 = c, a3 = c, a4 = c, a5 = c, a6 = c, a7 = c, \
                       a8 = c, a9 = c, aa = c, ab = c;
#define X12_STEP a0 = _mm256_fmadd_ps(a0, m, c); a1 = _mm256_fmadd_ps(a1, m, c); \
                 a2 = _mm256_fmadd_ps(a2, m, c); a3 = _mm256_fmadd_ps(a3, m, c); \
                 a4 = _mm256_fmadd_ps(a4, m, c); a5 = _mm256_fmadd_ps(a5, m, c); \
                 a6 = _mm256_fmadd_ps(a6, m, c); a7 = _mm256_fmadd_ps(a7, m, c); \
                 a8 = _mm256_fmadd_ps(a8, m, c); a9 = _mm256_fmadd_ps(a9, m, c); \
                 aa = _mm256_fmadd_ps(aa, m, c); ab = _mm256_fmadd_ps(ab, m, c);
#define X12_SUM  _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)), \
                               _mm256_add_ps(_mm256_add_ps(a4, a5), _mm256_add_ps(a6, a7)))
static double b12(int th) { BODY(12, X12_DECL) }

int main(int argc, char ** argv) {
    const int th = argc > 1 ? atoi(argv[1]) : 4;
    const double ghz = 2.62;   // measured earlier: 83.9 GFLOPS/core with 8 chains
    printf("threads=%d   chains | GMAC/s | FMA/cycle/core | %% of 2.0/cycle\n", th);
    struct { int n; double (*f)(int); } cases[] = { {4, b4}, {6, b6}, {8, b8}, {12, b12} };
    for (int i = 0; i < 4; i++) {
        const double g = cases[i].f(th);
        const double fpc = g * 1e9 / th / ghz / 1e9;   // GMAC/s -> FMA/cycle/core
        printf("             %2d      | %6.1f | %5.2f           | %5.1f%%\n",
               cases[i].n, g, fpc, fpc / 2.0 * 100.0);
    }
    return 0;
}
