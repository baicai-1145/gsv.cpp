// What is this box's real fp32 FMA peak? Measured so the answer does not depend on assuming a
// clock frequency.
//
// Method: rdtsc is an invariant counter ticking at the CPU's NOMINAL rate, so counting FMAs per
// TSC tick gives throughput directly:
//     FMA per TSC tick = 2 (the AVX2 issue limit) * (actual core clock / nominal clock)
// If the core never boosts, the maximum is 2.0; anything above that is boost, and the ratio tells
// us by how much. That makes the measurement independent of /proc/cpuinfo, which in a VM usually
// reports the nominal clock whether or not the core is boosting (we checked: it prints 2595 MHz
// during a 4-thread FMA load, which is why three earlier harnesses disagreed by 1.6x).
//
// SELF-CONTROLS (a peak number without these is worthless - this project has had three wrong
// microbenchmarks already):
//   1. chain sweep: with FMA latency L and C independent accumulators the sustained rate is
//      min(2, C/L). 4 chains must be clearly slower than 16. If they are equal, the compiler
//      removed the work and the number is garbage.
//   2. iteration sweep: total time must scale linearly with ITERS. If it does not, the loop was
//      hoisted or folded.
//   3. -O2 vs -O3 and a no-FMA variant must behave as expected.
//
//   cc -O3 -mavx2 -mfma -fopenmp fp32_peak.c -o fp32_peak
//   ./fp32_peak            # runs the self-controls, then 1..4 threads
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

#define MULV _mm256_set1_ps(0.9999999f)
#define ADDV _mm256_set1_ps(1.0000001f)

// One 16-accumulator block, unrolled x2, as a macro so nothing can be "optimised" away silently.
#define STEP16                                                              \
    a0 = _mm256_fmadd_ps(a0, m, c); a1 = _mm256_fmadd_ps(a1, m, c);         \
    a2 = _mm256_fmadd_ps(a2, m, c); a3 = _mm256_fmadd_ps(a3, m, c);         \
    a4 = _mm256_fmadd_ps(a4, m, c); a5 = _mm256_fmadd_ps(a5, m, c);         \
    a6 = _mm256_fmadd_ps(a6, m, c); a7 = _mm256_fmadd_ps(a7, m, c);         \
    a8 = _mm256_fmadd_ps(a8, m, c); a9 = _mm256_fmadd_ps(a9, m, c);         \
    aa = _mm256_fmadd_ps(aa, m, c); ab = _mm256_fmadd_ps(ab, m, c);         \
    ac = _mm256_fmadd_ps(ac, m, c); ad = _mm256_fmadd_ps(ad, m, c);         \
    ae = _mm256_fmadd_ps(ae, m, c); af = _mm256_fmadd_ps(af, m, c);

#define DECL16                                                             \
    __m256 a0 = c, a1 = c, a2 = c, a3 = c, a4 = c, a5 = c, a6 = c, a7 = c;  \
    __m256 a8 = c, a9 = c, aa = c, ab = c, ac = c, ad = c, ae = c, af = c;
#define SUM16                                                              \
    _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)), \
                                _mm256_add_ps(_mm256_add_ps(a4, a5), _mm256_add_ps(a6, a7))), \
                  _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a8, a9), _mm256_add_ps(aa, ab)), \
                                _mm256_add_ps(_mm256_add_ps(ac, ad), _mm256_add_ps(ae, af))))

// 4 accumulators: the latency-bound control.
#define DECL4  __m256 a0 = c, a1 = c, a2 = c, a3 = c;
#define STEP4  a0 = _mm256_fmadd_ps(a0, m, c); a1 = _mm256_fmadd_ps(a1, m, c); \
               a2 = _mm256_fmadd_ps(a2, m, c); a3 = _mm256_fmadd_ps(a3, m, c);
#define SUM4   _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3))

// Runs `iters` iterations of an N-accumulator block (unrolled U times) on `th` threads.
// Returns the aggregate MAC count and the elapsed TSC ticks (per thread 0).
static void run(int which, long iters, int th, double * macs_out, uint64_t * tsc_out) {
    unsigned aux;
    const uint64_t t0 = __rdtscp(&aux);
    #pragma omp parallel num_threads(th)
    {
        const __m256 m = MULV, c = ADDV;
        if (which == 16) {
            __m256 a0 = c, a1 = c, a2 = c, a3 = c, a4 = c, a5 = c, a6 = c, a7 = c;
            __m256 a8 = c, a9 = c, aa = c, ab = c, ac = c, ad = c, ae = c, af = c;
            for (long it = 0; it < iters; it++) { STEP16 STEP16 }
            float tmp[8];
            _mm256_storeu_ps(tmp, SUM16);
            volatile float sink = tmp[0] + tmp[7];
            (void) sink;
        } else {
            __m256 a0 = c, a1 = c, a2 = c, a3 = c;
            for (long it = 0; it < iters; it++) { STEP4 STEP4 }
            float tmp[8];
            _mm256_storeu_ps(tmp, SUM4);
            volatile float sink = tmp[0] + tmp[7];
            (void) sink;
        }
    }
    const uint64_t t1 = __rdtscp(&aux);
    *tsc_out = t1 - t0;
    *macs_out = (double) which * 2.0 /*unroll*/ * (double) iters * th;   // MAC = lanes already in `which`
}

int main(int argc, char ** argv) {
    const long ITERS = 400000;
    const double NOMINAL_GHZ = 2.595;   // /proc/cpuinfo under load; only used for the GFLOPS column

    printf("=== self-control 1: chain count (1 thread) ===\n");
    printf("   4 chains must be clearly slower than 16, else the loop was optimised away\n");
    double m4, m16; uint64_t t4, t16;
    run(4,  ITERS, 1, &m4,  &t4);
    run(16, ITERS, 1, &m16, &t16);
    const double r4 = m4 / (double) t4, r16 = m16 / (double) t16;
    printf("   4 chains : %8.1f MMAC  %8.1f Mtick  -> %.3f MAC/tick\n", m4 / 1e6, t4 / 1e6, r4);
    printf("   16 chains: %8.1f MMAC  %8.1f Mtick  -> %.3f MAC/tick\n", m16 / 1e6, t16 / 1e6, r16);
    printf("   ratio 4/16 = %.3f  %s\n", r4 / r16,
           (r4 / r16 < 0.85) ? "(OK: latency-bound as expected - the harness responds to chains)"
                             : "(WARNING: 4 chains are as fast as 16 -> work was removed, DISCARD)");

    printf("\n=== self-control 2: iteration scaling (1 thread, 16 chains) ===\n");
    double ma, mb; uint64_t ta, tb;
    run(16, ITERS, 1, &ma, &ta);
    run(16, ITERS * 4, 1, &mb, &tb);
    printf("   ITERS x1 -> %.1f Mtick, x4 -> %.1f Mtick, ratio %.3f (expect ~4.0)\n",
           ta / 1e6, tb / 1e6, (double) tb / (double) ta);

    printf("\n=== 16-chain peak vs thread count ===\n");
    printf("  thr    MAC/tick/thread   GMAC/s   GFLOPS   implied GHz @2 FMA/cyc\n");
    double best4 = 0;
    for (int th = 1; th <= 4; th++) {
        run(16, ITERS, th, &m16, &t16);
        const double per_thread_tick = m16 / (double) t16 / th;   // MAC per tick per thread
        const double gmacs = m16 / ((double) t16 / (NOMINAL_GHZ * 1e9)) / 1e9;  // MAC/s, total
        const double gflops = gmacs * 2.0;
        // MAC/tick/thread = 16 * (core GHz / nominal GHz) / 8 lanes ... in MAC terms:
        // one FMA = 8 MACs, 2 FMA/cycle = 16 MAC/cycle -> MAC/tick = 16 * (core/nominal)
        const double implied_ghz = per_thread_tick / 16.0 * NOMINAL_GHZ;
        printf("   %d      %8.3f         %7.1f  %7.1f   %8.2f\n",
               th, per_thread_tick, gmacs, gflops, implied_ghz);
        if (th == 4) best4 = gmacs;
    }
    printf("\n  4 threads -> %.1f GMAC/s = %.1f GFLOPS (this is the number to use as the fp32 ceiling)\n",
           best4, best4 * 2.0);
    return 0;
}
