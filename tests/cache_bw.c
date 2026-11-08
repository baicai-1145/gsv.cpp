// Where are the cache knees on this box, and how much read bandwidth does each level actually give?
//
// A conv kernel's weight tile is re-read from L2 into L1 once per t-block, so what matters is not
// whether the tile fits in L2 (everything here does - the biggest tile is 176 KiB against 4 MiB)
// but how many GB/s of L1<-L2 traffic the core can sustain. This probe measures achieved read
// bandwidth against footprint size, so the L1/L2/L3/DRAM knees are measured rather than assumed
// (in a VM, lscpu's numbers are frequently the host's, not the guest's).
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

// Read-only sum over `n` floats touched at 64-byte stride (one float per cache line, so we measure
// bandwidth without counting the other 15 floats of the line we did not ask for).
static double read_pass(const float * a, size_t n, int stride_f, int th, double * gbs_out) {
    const size_t lines = n / stride_f;
    double best = 1e9;
    for (int rep = 0; rep < 3; rep++) {
        double t0 = now_s();
        float acc = 0.f;
        #pragma omp parallel num_threads(th) reduction(+:acc)
        {
            const size_t chunk = lines / (size_t) th + 1;
            const size_t t = (size_t) omp_get_thread_num();
            const size_t lo = t * chunk, hi = (lo + chunk < lines) ? lo + chunk : lines;
            float s = 0.f;
            for (size_t i = lo; i < hi; i++) s += a[i * (size_t) stride_f];
            acc += s;
        }
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
        volatile float sink = acc;   // keep the loads alive
        (void) sink;
    }
    *gbs_out = (double) lines * 64.0 / best / 1e9;
    return best;
}

int main(void) {
    struct { const char * name; size_t bytes; } cases[] = {
        {"  16 KiB (L1)",       16u << 10},
        {"  32 KiB (L1 cap)",   32u << 10},
        {"  64 KiB (>L1)",      64u << 10},
        {" 256 KiB (L2)",      256u << 10},
        {"   1 MiB (L2)",        1u << 20},
        {"   4 MiB (L2 cap)",    4u << 20},
        {"  16 MiB (L3 cap)",   16u << 20},
        {"  64 MiB (DRAM)",     64u << 20},
    };
    const int ncase = (int) (sizeof(cases) / sizeof(cases[0]));
    const size_t MAXB = 64u << 20;
    float * a = (float *) aligned_alloc(64, MAXB + 64);
    for (size_t i = 0; i < MAXB / sizeof(float); i++) a[i] = 1.0f;

    printf("footprint           1 thr GB/s   4 thr GB/s   4thr/1thr\n");
    for (int i = 0; i < ncase; i++) {
        const size_t n = cases[i].bytes / sizeof(float);
        double g1 = 0, g4 = 0;
        read_pass(a, n, 16, 1, &g1);
        read_pass(a, n, 16, 4, &g4);
        printf("%-18s %8.1f     %8.1f      %5.2fx\n", cases[i].name, g1, g4, g4 / g1);
    }
    printf("\n(the knee at 32 KiB is L1d; the plateau after ~16 MiB is DRAM. What a conv pays for is\n"
           " the L1<-L2 refill rate, i.e. the GB/s in the L2-sized rows.)\n");
    free(a);
    return 0;
}
