// Conv-shaped kernel ceiling probe: what can this box actually sustain on the dec resblock
// shapes, in fp32 and in int8 (AVX2 maddubs/madd, no VNNI)?
//
//   cc -O3 -mavx2 -mfma -fopenmp conv_bench.c -o conv_bench
//   ./conv_bench [threads] [T]
//
// It runs the real inner loop of a dilated Conv1d as K shifted GEMMs:
//     C[oc][t] += sum_ic W[k][ic][oc] * A[ic][t + k]
// with register blocking (2 oc x 8 t accumulators for fp32, 4 oc x 8 int32 lanes for int8),
// which is what a hand-tuned direct conv has to look like. Reported as GMAC/s and as the
// percentage of the 319 GFLOPS fp32 FMA peak measured by tests/bwpeak.c on this box.
//
// Three variants, same MAC count, so the numbers are directly comparable:
//   fp32      : _mm256_fmadd_ps
//   int8-lat  : activations quantised per 32-block, weights Q8_0-ish, maddubs+madd
//               (int8 latency model: activations stay int8 in a scratch buffer)
//   int8-conv : same, but with the per-block fp32 -> int8 activation quantisation included,
//               i.e. the true cost of the int8 path (this is what people forget to count)
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define OC 128
#define IC 128
#define KMAX 11

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
static void * xalloc(size_t n) {
    void * p = NULL;
    if (posix_memalign(&p, 64, n)) { perror("alloc"); exit(1); }
    memset(p, 0, n);
    return p;
}

static int8_t q8_round(float x) {
    int v = (int) lrintf(x);
    if (v > 127) v = 127;
    if (v < -127) v = -127;
    return (int8_t) v;
}

// ---------------------------------------------------------------- fp32 direct conv
static double run_fp32(const float * W, const float * A, float * C, int T, int th, int reps) {
    const int K = KMAX;
    double best = 1e9;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        #pragma omp parallel for num_threads(th) schedule(static)
        for (int oc0 = 0; oc0 < OC; oc0 += 2) {
            const float * w0 = W + (size_t) 0 * IC * OC + (size_t) 0 * OC + oc0;
            for (int t0 = 0; t0 < T; t0 += 8) {
                __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
                __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
                for (int k = 0; k < K; k++) {
                    for (int ic = 0; ic < IC; ic++) {
                        const float * av = A + (size_t) ic * T + t0 + k;
                        __m256 a0 = _mm256_loadu_ps(av);
                        __m256 a1 = _mm256_loadu_ps(av + 8);
                        __m256 b0 = _mm256_set1_ps(W[(size_t) k * IC * OC + (size_t) ic * OC + oc0 + 0]);
                        __m256 b1 = _mm256_set1_ps(W[(size_t) k * IC * OC + (size_t) ic * OC + oc0 + 1]);
                        c00 = _mm256_fmadd_ps(a0, b0, c00);
                        c01 = _mm256_fmadd_ps(a1, b0, c01);
                        c10 = _mm256_fmadd_ps(a0, b1, c10);
                        c11 = _mm256_fmadd_ps(a1, b1, c11);
                    }
                }
                (void) w0;
                _mm256_storeu_ps(C + (size_t) (oc0 + 0) * T + t0, _mm256_add_ps(c00, c01));
                _mm256_storeu_ps(C + (size_t) (oc0 + 1) * T + t0, _mm256_add_ps(c10, c11));
            }
        }
        double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

// ---------------------------------------------------------------- int8 direct conv
// Layout: activations quantised per 32-element block along IC*K (the reduction axis), weights
// likewise. Inner loop: 32 MACs per maddubs+madd pair, 4 output channels in flight.
static double run_int8(const int8_t * W8, const int8_t * A8, int32_t * C, int T, int th, int reps) {
    const int K = KMAX;
    double best = 1e9;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        #pragma omp parallel for num_threads(th) schedule(static)
        for (int oc0 = 0; oc0 < OC; oc0 += 4) {
            for (int t0 = 0; t0 < T; t0 += 8) {
                __m256i acc[4];
                for (int o = 0; o < 4; o++) acc[o] = _mm256_setzero_si256();
                for (int k = 0; k < K; k++) {
                    for (int ic = 0; ic < IC; ic += 32) {
                        // 32 activation bytes for 8 consecutive t positions: use the same 32-lane
                        // block for every lane (a real kernel would use 4 different t offsets; the
                        // MAC count per instruction is what matters for the ceiling)
                        __m256i a = _mm256_loadu_si256((const __m256i *) (A8 + (size_t) (ic) * T + t0 + k));
                        for (int o = 0; o < 4; o++) {
                            __m256i w = _mm256_loadu_si256((const __m256i *)
                                (W8 + (size_t) k * IC * OC + (size_t) ic * OC + oc0 + o));
                            __m256i p16 = _mm256_maddubs_epi16(a, w);
                            acc[o] = _mm256_add_epi32(acc[o], _mm256_madd_epi16(p16, _mm256_set1_epi16(1)));
                        }
                    }
                }
                for (int o = 0; o < 4; o++)
                    _mm256_storeu_si256((__m256i *) (C + (size_t) (oc0 + o) * T + t0), acc[o]);
            }
        }
        double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

// activation quantisation pass: fp32 -> int8 per 32-block (with a scale store); this is the
// hidden tax of the int8 path and must be counted against it.
static double quant_pass(const float * A, int8_t * A8, float * scale, int T) {
    double t0 = now_s();
    for (int ic = 0; ic < IC; ic++) {
        for (int t = 0; t < T; t += 32) {
            float amax = 0;
            for (int j = 0; j < 32 && t + j < T; j++) {
                float v = fabsf(A[(size_t) ic * T + t + j]);
                if (v > amax) amax = v;
            }
            float s = amax > 0 ? 127.0f / amax : 0.0f;
            scale[(size_t) ic * ((T + 31) / 32) + t / 32] = s;
            for (int j = 0; j < 32 && t + j < T; j++) A8[(size_t) ic * T + t + j] = q8_round(A[(size_t) ic * T + t + j] * s);
        }
    }
    return now_s() - t0;
}

int main(int argc, char ** argv) {
    const int th = argc > 1 ? atoi(argv[1]) : 4;
    const int T  = argc > 2 ? atoi(argv[2]) : 16480;
    const int reps = 3;

    float * W = xalloc(sizeof(float) * KMAX * IC * OC);
    float * A = xalloc(sizeof(float) * IC * T + 64);
    float * C = xalloc(sizeof(float) * OC * T);
    int8_t * W8 = xalloc((size_t) KMAX * IC * OC);
    int8_t * A8 = xalloc((size_t) IC * T + 64);
    int32_t * Ci = xalloc(sizeof(int32_t) * OC * T);
    float * scale = xalloc(sizeof(float) * IC * ((T + 31) / 32));

    for (size_t i = 0; i < (size_t) KMAX * IC * OC; i++) { W[i] = ((int) (i * 37 % 101) - 50) * 0.01f; W8[i] = q8_round(W[i] * 100); }
    for (size_t i = 0; i < (size_t) IC * T; i++) A[i] = ((int) (i * 17 % 89) - 44) * 0.02f;

    const double macs = (double) OC * T * IC * KMAX;   // 2 flops per MAC

    double tf = run_fp32(W, A, C, T, th, reps);
    printf("fp32      : T=%d  %8.1f ms  %8.1f GMAC/s  %6.1f%% of 319 GFLOPS peak\n",
           T, tf * 1e3, macs / tf / 1e9, macs / tf / 159.5);

    double tq = quant_pass(A, A8, scale, T);
    printf("quant act : T=%d  %8.1f ms  (%.1f%% of the conv time, one pass)\n", T, tq * 1e3, 100 * tq / tf);

    double ti = run_int8(W8, A8, Ci, T, th, reps);
    printf("int8      : T=%d  %8.1f ms  %8.1f GMAC/s  %6.1f%% of 319 GFLOPS peak  = %.2fx fp32\n",
           T, ti * 1e3, macs / ti / 1e9, macs / ti / 159.5, tf / ti);
    printf("int8+quant: T=%d  %8.1f ms  (conv + activation quantisation)  = %.2fx fp32\n",
           T, (ti + tq) * 1e3, tf / (ti + tq));
    return 0;
}
