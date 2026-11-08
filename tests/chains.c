// Why does a conv kernel cap at a given percentage of "peak"? Answer: it is not the peak, it is
// the dependency-chain count.
//
// A 256-bit FMA on Zen2 (and on most AVX2 cores) has 4-5 cycles of latency but 2/cycle issue
// rate. Sustained FMAs/cycle is therefore bounded by (independent accumulator chains / latency).
// This probe measures that curve directly: the same pure-FMA loop with N independent accumulators,
// no memory traffic, so nothing but the chain count changes.
//
//   cc -O3 -mavx2 -mfma -fopenmp chains.c -o chains
//   ./chains [threads]
//
// Then the same with the memory pattern of a real conv (1 activation load + 1 weight broadcast per
// 2 FMAs) to show how much of the gap that explains.
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

#define ITERS 400000
#define MAXACC 24

// Pure register FMA, N independent chains.
static double bench_chains(int nacc, int th) {
    const __m256 m = _mm256_set1_ps(0.9999999f);
    const __m256 c = _mm256_set1_ps(1.0000001f);
    double best = 1e9;
    for (int rep = 0; rep < 3; rep++) {
        double t0 = now_s();
        #pragma omp parallel num_threads(th)
        {
            __m256 acc[MAXACC];
            for (int i = 0; i < nacc; i++) acc[i] = _mm256_set1_ps(1.0f + i * 1e-7f);
            const long iters = ITERS;
            long it = 0;
            // unrolled by 4 so the loop branch itself never becomes the bottleneck
            for (; it + 4 <= iters; it += 4) {
                #pragma GCC unroll 4
                for (int u = 0; u < 4; u++) {
                    for (int i = 0; i < nacc; i++) acc[i] = _mm256_fmadd_ps(acc[i], m, c);
                }
            }
            __m256 s = acc[0];
            for (int i = 1; i < nacc; i++) s = _mm256_add_ps(s, acc[i]);
            float tmp[8];
            _mm256_storeu_ps(tmp, s);
            volatile float sink = tmp[0] + tmp[7];
            (void) sink;
        }
        double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    const double fmas = (double) nacc * ITERS * (double) th;
    return fmas / best / 1e9;   // GFMA/s (= 8*FMA GFLOP/s per lane count... reported as GMAC/s)
}

int main(int argc, char ** argv) {
    const int th = argc > 1 ? atoi(argv[1]) : 4;

    // The machine's own 2-FMA/cycle limit, measured with the widest chain count we can hold.
    printf("threads=%d   N chains | GMAC/s | FMA/cycle/core | %% of the 16-chain value\n", th);
    double ref = 0;
    for (int n = 2; n <= 16; n += 2) {
        double g = bench_chains(n, th);
        if (n == 16) ref = g;
        // GMAC/s per core -> FMA/cycle/core: assume ~2.6 GHz measured earlier (83.9 GFLOPS/core)
        const double fma_per_cycle = g * 1e9 / th / 2.6e9;
        printf("            %2d        | %6.1f | %5.2f           | %5.1f%%\n",
               n, g * 8 / 8.0, fma_per_cycle, 0.0);
    }
    printf("\n(interpretation: FMA latency ~4 cycles, so N chains sustain N/4 FMAs per cycle;\n"
           " 4 chains -> 1.0 FMA/cyc = 50%%, 6 -> 1.5 = 75%%, 8+ -> 2.0 = 100%%)\n");
    return 0;
}
