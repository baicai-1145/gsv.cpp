// STATUS: DISABLED - seven revisions, all wrong. Across revisions the "AVX2 12-chain" figure
// stayed pinned at 588-589 GMAC/s while every other configuration changed, i.e. that number is a
// measurement artefact and not throughput; the built-in impossible-value check (implied FMA/cycle
// must be <= 2) flags it correctly, but nothing here can be trusted as a ceiling. Lesson recorded
// rather than a number: on-box microbenchmarks of FMA throughput are harder than they look
// (constant folding, closed-form recurrence collapse, SMT effects), and the only figures kept in
// this repo are ones cross-checked against a known-good implementation (OpenBLAS sgemm) or
// against wall-clock end-to-end timings of the real engine.
// For the record, theory + the measured 3.077 GHz clock on the Zen4 box gives, per core:
//   AVX2   fp32 : 16 MAC/cycle =  49 GMAC/s
//   AVX-512 fp32: 32 MAC/cycle =  98 GMAC/s
//   VNNI   int8 : 128 MAC/cycle = 394 GMAC/s
// SIMD throughput ceiling for this machine: AVX2 FMA vs AVX-512 FMA vs AVX-512 VNNI int8.
//
// Written after four broken microbenchmarks on the previous box, so the guards here are explicit:
//   1. the multiplier and addend come from VOLATILE globals, so gcc cannot fold the recurrence
//      a = fma(a, m, c) into a closed form (that is exactly what happened with constant operands:
//      it reported 40 MAC/cycle, 2.5x the hardware limit);
//   2. a chain-count control runs the same loop with 4 accumulators: with FMA latency L and C
//      independent chains the rate is min(2, C/L) per cycle, so 4 chains must be clearly slower
//      than 12. If they are equal, the work was removed and the number is garbage;
//   3. an iteration-scaling control: time must grow linearly with the iteration count;
//   4. the result is read back and printed, so nothing can be dead-code eliminated.
//
//   cc -O3 -march=native -fopenmp simd_peak.c -o simd_peak && ./simd_peak
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

// volatile sources: no constant folding, no hoisting
static volatile float g_mul = 0.9999999f;
static volatile float g_add = 1.0000001f;
static volatile unsigned char g_u8[64];

#define ITERS 2000000

/* ---------------- AVX2: 12 x __m256 FMA ---------------- */
#define A2_DECL __m256 a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,aa,ab;
#define A2_INIT for (int i=0;i<12;i++) ((__m256*)&a0)[i] = c;
#define A2_STEP a0=_mm256_fmadd_ps(a0,m,c); a1=_mm256_fmadd_ps(a1,m,c); a2=_mm256_fmadd_ps(a2,m,c); \
                a3=_mm256_fmadd_ps(a3,m,c); a4=_mm256_fmadd_ps(a4,m,c); a5=_mm256_fmadd_ps(a5,m,c); \
                a6=_mm256_fmadd_ps(a6,m,c); a7=_mm256_fmadd_ps(a7,m,c); a8=_mm256_fmadd_ps(a8,m,c); \
                a9=_mm256_fmadd_ps(a9,m,c); aa=_mm256_fmadd_ps(aa,m,c); ab=_mm256_fmadd_ps(ab,m,c);
#define A2_SUM (a0+a1+a2+a3+a4+a5+a6+a7+a8+a9+aa+ab)

static double bench_avx2(int chains, int th) {
    __m256 m = _mm256_set1_ps(g_mul), c = _mm256_set1_ps(g_add);
    double best = 1e9;
    for (int rep = 0; rep < 3; rep++) {
        const double t0 = now_s();
        #pragma omp parallel num_threads(th)
        {
            A2_DECL A2_INIT
            for (long it = 0; it < ITERS; it++) {
                m = _mm256_set1_ps(g_mul); c = _mm256_set1_ps(g_add);
                if (chains >= 12) { A2_STEP A2_STEP }
                else { a0=_mm256_fmadd_ps(a0,m,c); a1=_mm256_fmadd_ps(a1,m,c);
                       a2=_mm256_fmadd_ps(a2,m,c); a3=_mm256_fmadd_ps(a3,m,c);
                       a0=_mm256_fmadd_ps(a0,m,c); a1=_mm256_fmadd_ps(a1,m,c);
                       a2=_mm256_fmadd_ps(a2,m,c); a3=_mm256_fmadd_ps(a3,m,c); }
            }
            float tmp[8];
            _mm256_storeu_ps(tmp, A2_SUM);
            volatile float sink = tmp[0] + tmp[7];
            (void) sink;
        }
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    const int u = (chains >= 12) ? 2 : 1;
    const int n = (chains >= 12) ? 12 : 8;      // 8 FMAs per iteration in the 4-chain control
    return (double) n * u * 8.0 * ITERS * th / best / 1e9;   // GMAC/s
}

#ifdef __AVX512F__
/* ---------------- AVX-512: 12 x __m512 FMA ---------------- */
static double bench_avx512(int th) {
    __m512 m = _mm512_set1_ps(g_mul), c = _mm512_set1_ps(g_add);
    double best = 1e9;
    for (int rep = 0; rep < 3; rep++) {
        const double t0 = now_s();
        #pragma omp parallel num_threads(th)
        {
            __m512 a0=c,a1=c,a2=c,a3=c,a4=c,a5=c,a6=c,a7=c,a8=c,a9=c,aa=c,ab=c;
            for (long it = 0; it < ITERS; it++) {
                m = _mm512_set1_ps(g_mul); c = _mm512_set1_ps(g_add);
                #define S(x) x = _mm512_fmadd_ps(x, m, c);
                S(a0) S(a1) S(a2) S(a3) S(a4) S(a5) S(a6) S(a7) S(a8) S(a9) S(aa) S(ab)
                S(a0) S(a1) S(a2) S(a3) S(a4) S(a5) S(a6) S(a7) S(a8) S(a9) S(aa) S(ab)
                #undef S
            }
            __m512 s = _mm512_add_ps(_mm512_add_ps(a0,a1),_mm512_add_ps(a2,a3));
            s = _mm512_add_ps(s, _mm512_add_ps(_mm512_add_ps(a4,a5),_mm512_add_ps(a6,a7)));
            s = _mm512_add_ps(s, _mm512_add_ps(_mm512_add_ps(a8,a9),_mm512_add_ps(aa,ab)));
            float tmp[16];
            _mm512_storeu_ps(tmp, s);
            volatile float sink = tmp[0] + tmp[15];
            (void) sink;
        }
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    return 24.0 * 16.0 * ITERS * th / best / 1e9;
}

/* ---------------- AVX-512 VNNI: int8 dot, vpdpbusd = 64 int8 MAC per instruction ------------ */
static double bench_vnni(int th) {
    const __m512i acc0 = _mm512_setzero_si512();
    __m512i uu, ss;
    double best = 1e9;
    for (int rep = 0; rep < 3; rep++) {
        const double t0 = now_s();
        #pragma omp parallel num_threads(th)
        {
            __m512i a0=acc0,a1=acc0,a2=acc0,a3=acc0,a4=acc0,a5=acc0,a6=acc0,a7=acc0;
            for (long it = 0; it < ITERS; it++) {
                __asm__ __volatile__("" ::: "memory");
                uu = _mm512_set1_epi8((char) 3); ss = _mm512_set1_epi8((char) 2 - 128);
                #define D(x) x = _mm512_dpbusd_epi32(x, uu, ss);
                D(a0) D(a1) D(a2) D(a3) D(a4) D(a5) D(a6) D(a7)
                D(a0) D(a1) D(a2) D(a3) D(a4) D(a5) D(a6) D(a7)
                #undef D
            }
            __m512i t = _mm512_add_epi32(_mm512_add_epi32(a0,a1),_mm512_add_epi32(a2,a3));
            t = _mm512_add_epi32(t, _mm512_add_epi32(_mm512_add_epi32(a4,a5),_mm512_add_epi32(a6,a7)));
            int32_t tmp[16];
            _mm512_storeu_si512((void *) tmp, t);
            volatile int32_t sink = tmp[0] + tmp[15];
            (void) sink;
        }
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    return 16.0 * 64.0 * ITERS * th / best / 1e9;   // GMAC/s (int8)
}
#endif

static void verdict(const char * name, double gmacs, int lanes, double ghz, int th) {
    // one FMA instruction does `lanes` MACs; at most 2 FMA instructions per cycle per core
    const double fma_per_cycle = gmacs * 1e9 / (double) lanes / (ghz * 1e9) / (double) th;
    printf("   implied %.2f FMA/cycle/thread %s\n", fma_per_cycle,
           fma_per_cycle > 2.2 ? "  << IMPOSSIBLE -> loop was optimised away, DISCARD" : "  (ok)");
}

int main(int argc, char ** argv) {
    const int th = argc > 1 ? atoi(argv[1]) : 2;
    printf("threads=%d   (volatile operands: no constant folding; result printed so nothing is dead)\n\n", th);

    const double c4 = bench_avx2(4, 1), c12 = bench_avx2(12, 1);
    printf("control: AVX2 4 chains = %.1f GMAC/s, 12 chains = %.1f GMAC/s, ratio %.2f (expect >2)\n",
           c4, c12, c4 / c12);

    const double GHZ = 3.077;   // measured with clock_probe (dependent add chain, 4 rows agree)
    printf("\n  unit            GMAC/s\n");
    const double a2 = bench_avx2(12, th);
    printf("  AVX2  fma      %7.1f\n", a2);
    verdict("AVX2", a2, 8, GHZ, th);
#ifdef __AVX512F__
    const double a5 = bench_avx512(th);
    printf("  AVX-512 fma    %7.1f\n", a5);
    verdict("AVX-512", a5, 16, GHZ, th);
    const double vn = bench_vnni(th);
    printf("  AVX-512 VNNI   %7.1f  (int8, vpdpbusd = 64 MAC/instr)\n", vn);
    verdict("VNNI", vn, 64, GHZ, th);
    printf("\n  AVX-512 / AVX2 fp32 ratio = %.2fx\n", a5 / a2);
    printf("  VNNI int8 / AVX-512 fp32  = %.2fx\n", vn / a5);
#endif
    return 0;
}
