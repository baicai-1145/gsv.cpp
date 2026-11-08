// What is the real core clock on this box? /proc/cpuinfo reports 2595 MHz even under a 4-thread
// FMA load, yet OpenBLAS sgemm reaches 95.3 GFLOPS single-thread, which at 2595 MHz would be 114%
// of the AVX2 fp32 limit (32 flops/cycle). So the reported clock is nominal, not actual.
//
// Method: a serial integer add chain. On Zen2, `add` has 1-cycle latency, so N dependent adds take
// exactly N core cycles. Timing the chain with a wall clock therefore measures the core clock
// directly, with no assumptions:
//     clock = N_adds / wall_seconds
// The chain is written in inline asm so no compiler transformation can break the dependency, and
// it is unrolled 8x per loop iteration (8 dependent adds + 1 branch, the branch overlaps).
//
//   cc -O2 clock_probe.c -o clock_probe && ./clock_probe
#include <stdio.h>
#include <stdint.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

// 8 dependent adds per iteration, all on the same register.
static inline void chain_8(uint64_t * x, long iters) {
    __asm__ __volatile__(
        "1:\n\t"
        "add $1, %[x]\n\t"
        "add $1, %[x]\n\t"
        "add $1, %[x]\n\t"
        "add $1, %[x]\n\t"
        "add $1, %[x]\n\t"
        "add $1, %[x]\n\t"
        "add $1, %[x]\n\t"
        "add $1, %[x]\n\t"
        "dec %[n]\n\t"
        "jnz 1b\n\t"
        : [x] "+r"(*x), [n] "+r"(iters)
        :
        : "cc");
}

int main(void) {
    printf("dependent integer add chain: 1 add = 1 cycle on Zen2, so N/T = core clock\n");
    printf("iters        adds        wall ms     GHz\n");
    for (int e = 0; e < 4; e++) {
        const long iters = 50000000L << e;      // 50M .. 400M iterations (x8 adds each)
        uint64_t x = 1;
        long n = iters;
        const double t0 = now_s();
        chain_8(&x, n);
        const double t1 = now_s();
        const double adds = 8.0 * (double) iters;
        const double ghz = adds / (t1 - t0) / 1e9;
        printf("%-12ld %-11.0f %-11.2f %.3f   (x=%llu)\n",
               iters, adds, (t1 - t0) * 1e3, ghz, (unsigned long long) x);
    }
    printf("\nIf the last rows agree to within ~1%%, that is the sustained core clock.\n"
           "Multiply by 32 for the single-core fp32 FLOPS ceiling (2 FMA/cycle x 8 lanes x 2),\n"
           "and by 128 for 4 cores.\n");
    return 0;
}
