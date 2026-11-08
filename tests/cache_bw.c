// STATUS: DISABLED - reported 209 GB/s at EVERY footprint from 16 KiB to 64 MiB, with 800 GB/s at
// 4 threads. A flat curve across four orders of magnitude means it is not measuring memory at all
// (single-core DRAM read on this box is ~20 GB/s, measured by the previous revision of this file,
// which was noisy for the small sizes for a different reason: OpenMP fork/join overhead). Sixth
// broken microbenchmark in this project. DO NOT USE.
// Reliable alternative that is in the repo: numpy sum() bandwidth (see scripts/bench_mem.sh), or
// tests/bwpeak.c for the streaming triad/dot curves that docs/perf-tuning-4vcpu.md quotes.
// Cache knees and per-level read bandwidth, measured.
//
// Why this exists: a conv kernel re-reads its weight tile from L2 into L1 once per t-block, so what
// binds is the sustained L1<-L2 refill rate, not whether the tile fits in L2 (largest tile is 176 KiB
// against 4 MiB). In a VM, lscpu often reports the host cache geometry, so measure the knees.
//
// Simplest correct form, after four broken harnesses in this project:
//   - the pass count is computed OUTSIDE the parallel region from the total line count, so every
//     thread moves the same number of lines and bytes = lines * passes * 64 exactly;
//   - the repeat loop is INSIDE the parallel region, so a 256-line footprint is not fork/join.
//
//   cc -O2 -fopenmp cache_bw.c -o cache_bw && ./cache_bw
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

#define STRIDE_F 16          /* one float per 64-byte line */

/* Reads `lines` lines `passes` times, split across `th` threads. Returns GB/s. */
static double bw(const float * a, size_t lines, long passes, int th) {
    double best = 1e9;
    for (int rep = 0; rep < 3; rep++) {
        const double t0 = now_s();
        #pragma omp parallel num_threads(th)
        {
            const size_t chunk = (lines + (size_t) th - 1) / (size_t) th;
            const size_t lo = (size_t) omp_get_thread_num() * chunk;
            const size_t hi = (lo + chunk < lines) ? lo + chunk : lines;
            if (lo < hi) {
                float s = 0.f;
                const float * q = a + lo * (size_t) STRIDE_F;
                for (long p = 0; p < passes; p++)
                    for (size_t i = 0; i < hi - lo; i++) s += q[i * (size_t) STRIDE_F];
                volatile float sink = s;
                (void) sink;
            }
        }
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    return (double) lines * (double) passes * 64.0 / best / 1e9;
}

int main(void) {
    const char * names[] = {"  16 KiB", "  32 KiB (L1d)", "  64 KiB", " 256 KiB",
                            "   1 MiB", "   4 MiB", "  16 MiB", "  64 MiB"};
    const size_t bytes[] = {16u << 10, 32u << 10, 64u << 10, 256u << 10,
                            1u << 20, 4u << 20, 16u << 20, 64u << 20};
    const size_t MAXB = 64u << 20;
    float * a = (float *) aligned_alloc(64, MAXB);
    for (size_t i = 0; i < MAXB / sizeof(float); i++) a[i] = 1.0f;

    printf("footprint       1 thr GB/s   4 thr GB/s   4t/1t\n");
    for (int i = 0; i < 8; i++) {
        const size_t lines = (bytes[i] / sizeof(float)) / STRIDE_F;
        /* aim for ~20-60 ms per measurement at either thread count */
        const long passes = (long) (4.0e7 / (double) lines) + 4;
        const double g1 = bw(a, lines, passes, 1);
        const double g4 = bw(a, lines, passes, 4);
        printf("%-14s %8.1f      %8.1f     %5.2fx\n", names[i], g1, g4, g4 / g1);
    }
    printf("\nL1d knee expected between 32 and 64 KiB, L2 between 4 and 16 MiB, DRAM beyond.\n");
    free(a);
    return 0;
}
