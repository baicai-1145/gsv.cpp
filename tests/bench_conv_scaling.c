// dec conv 的 4 线程伸缩损失诊断台（单文件，pthreads，无 ggml 依赖）。
//
// 问题：同一个 resblock conv，单线程 ~48.7 GMAC/s（=16 MAC/cycle，FMA 峰值），4 线程只有
// 103 GMAC/s（25.8/线程，53% 伸缩）。要回答的是：损失来自
//   (a) 每个线程各流一份权重（t 切分下每个线程都要过全部 oc 块），
//   (b) X/O 的 DRAM 流量，
//   (c) 每核 L2(512K) 装不下权重切片导致的 L3 往返，还是
//   (d) fork-join / 线程唤醒（单 conv 只有一个 barrier，理论上可忽略）。
//
// 做法：复刻 ggml conv_direct_1d 的 phase1/phase2（与 tests/bench_conv_epi.c 同一份内核
// 代码，已被证明与 ggml 逐位相同），然后：
//   * --threads N            1..4，测 per-thread GMAC/s 曲线（关键图）
//   * --split t|oc           t 切分（现状，权重每线程各一份）vs oc 切分（权重不重复、
//                           但 X 每线程各一份）—— 直接对比“哪种重复更贵”
//   * --cold                 每 rep 前用 64 MB 缓冲冲掉 L3，测冷流量；默认是热 L3
//   * 形状 archetype         全 L2 驻留 / 权重大于 L2 / X 主导，用来分离 L2 容量效应
//   * 每线程统计            自己的耗时/tile 数/字节数 ⇒ 每线程 GMAC/s、线程间不均衡、
//                           以及达到的 GB/s（和 37.1 GB/s 的 DRAM roofline 比）
//   * --bw                  同线程数下的纯读带宽参考（同 footprint），用来判断是不是带宽墙
//
// 规矩：threads>1 必须 GSV_BOX_FREE=1（这台机器共享，4 线程任务落进别人的测量窗口会把
// 两边样本一起废掉）；1 线程不需要。
//
//   GSV_BOX_FREE=1 ./scaling --threads 4 --shape real1        # 约 1 s
//   ./scaling --threads 1 --shape real1
//   GSV_BOX_FREE=1 ./scaling --threads 4 --split oc --shape real1
#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}
static void * xalloc(size_t n) {
    void * p = NULL;
    if (posix_memalign(&p, 64, n ? n : 64) != 0) { fprintf(stderr, "oom\n"); exit(1); }
    memset(p, 0, n);
    return p;
}
static void fill_rand(float * p, size_t n, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        p[i] = ((float) (s & 0xffff) / 32768.0f - 1.0f);
    }
}

// ---------------------------------------------------------------- 内核（与 bench_conv_epi 同源）
#define TB 6

struct conv {
    long IC, OC, K, dil, T, OL, pad;
    long OCp, KP, L;
    float * Wt, * bp, * Xp, * out;
    const float * w, * x, * b, * r;
};

static inline __m256 vleaky(__m256 v, float slope) {
    const __m256 neg = _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_LT_OQ);
    return _mm256_blendv_ps(v, _mm256_mul_ps(v, _mm256_set1_ps(slope)), neg);
}
static inline void transpose6x8(__m256 s0, __m256 s1, __m256 s2, __m256 s3, __m256 s4, __m256 s5,
                                __m256 r[8]) {
    const __m256 z = _mm256_setzero_ps();
    const __m256 u0 = _mm256_unpacklo_ps(s0, s1), u1 = _mm256_unpackhi_ps(s0, s1);
    const __m256 u2 = _mm256_unpacklo_ps(s2, s3), u3 = _mm256_unpackhi_ps(s2, s3);
    const __m256 u4 = _mm256_unpacklo_ps(s4, s5), u5 = _mm256_unpackhi_ps(s4, s5);
    const __m256d v0 = _mm256_unpacklo_pd(_mm256_castps_pd(u0), _mm256_castps_pd(u2));
    const __m256d v1 = _mm256_unpackhi_pd(_mm256_castps_pd(u0), _mm256_castps_pd(u2));
    const __m256d v2 = _mm256_unpacklo_pd(_mm256_castps_pd(u1), _mm256_castps_pd(u3));
    const __m256d v3 = _mm256_unpackhi_pd(_mm256_castps_pd(u1), _mm256_castps_pd(u3));
    const __m256d v4 = _mm256_unpacklo_pd(_mm256_castps_pd(u4), _mm256_castps_pd(z));
    const __m256d v5 = _mm256_unpackhi_pd(_mm256_castps_pd(u4), _mm256_castps_pd(z));
    const __m256d v6 = _mm256_unpacklo_pd(_mm256_castps_pd(u5), _mm256_castps_pd(z));
    const __m256d v7 = _mm256_unpackhi_pd(_mm256_castps_pd(u5), _mm256_castps_pd(z));
    const __m256 c0 = _mm256_castpd_ps(v0), c1 = _mm256_castpd_ps(v1);
    const __m256 c2 = _mm256_castpd_ps(v2), c3 = _mm256_castpd_ps(v3);
    const __m256 c4 = _mm256_castpd_ps(v4), c5 = _mm256_castpd_ps(v5);
    const __m256 c6 = _mm256_castpd_ps(v6), c7 = _mm256_castpd_ps(v7);
    r[0] = _mm256_permute2f128_ps(c0, c4, 0x20);
    r[4] = _mm256_permute2f128_ps(c0, c4, 0x31);
    r[1] = _mm256_permute2f128_ps(c1, c5, 0x20);
    r[5] = _mm256_permute2f128_ps(c1, c5, 0x31);
    r[2] = _mm256_permute2f128_ps(c2, c6, 0x20);
    r[6] = _mm256_permute2f128_ps(c2, c6, 0x31);
    r[3] = _mm256_permute2f128_ps(c3, c7, 0x20);
    r[7] = _mm256_permute2f128_ps(c3, c7, 0x31);
}
static inline void store6(float * o, __m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    _mm_storeu_ps(o, lo);
    _mm_storel_epi64((__m128i *) (o + 4), _mm_castps_si128(hi));
}

struct tstat { double secs; long tiles, wbytes, xbytes, obytes; };

// t 块范围 [tb0,tb1) × oc 块范围 [ob2lo,ob2hi)（ggml 的 TSB/OSB 分块原样保留）
static void compute_range(const struct conv * c, int64_t tb0, int64_t tb1, int64_t ob2lo,
                          int64_t ob2hi, struct tstat * st) {
    const long IC = c->IC, OC = c->OC, K = c->K, dil = c->dil, OL = c->OL, OCp = c->OCp, L = c->L;
    const float * Wt = c->Wt, * Xp = c->Xp, * bp = c->bp;
    float * o_data = c->out;
    const int64_t n_tb = (OL + TB - 1) / TB;
    const int64_t TSB0 = 65536 / (TB * IC * 4 + 1);
    const int64_t TSBc = TSB0 < 1 ? 1 : (TSB0 > 32 ? 32 : TSB0);

    const double t_start = now_s();
    for (int64_t ts = 0; ts < n_tb; ts += TSBc) {
        const int64_t te = (ts + TSBc < n_tb) ? ts + TSBc : n_tb;
        const int64_t a = ts > tb0 ? ts : tb0;
        const int64_t b = te < tb1 ? te : tb1;
        if (a >= b) continue;
        for (int64_t ob2 = ob2lo; ob2 < ob2hi; ++ob2) {
            const int64_t oc0 = ob2 * 16;
            const float * Wb = Wt + oc0;
            for (int64_t tb = a; tb < b; ++tb) {
                const int64_t t0 = tb * TB;
                if (t0 + TB > OL) {   // 尾巴：标量
                    for (int64_t t = t0; t < OL; ++t)
                        for (int64_t oc = oc0; oc < (oc0 + 16 < OC ? oc0 + 16 : OC); ++oc) {
                            float acc = bp[oc];
                            for (int64_t ic = 0; ic < IC; ++ic)
                                for (int64_t kw = 0; kw < K; ++kw)
                                    acc += Wt[(ic * K + kw) * OCp + oc] * Xp[ic * L + t + kw * dil];
                            o_data[oc * OL + t] = acc;
                        }
                    continue;
                }
                __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
                __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
                __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
                __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
                __m256 a40 = _mm256_setzero_ps(), a41 = _mm256_setzero_ps();
                __m256 a50 = _mm256_setzero_ps(), a51 = _mm256_setzero_ps();
                for (int64_t ic = 0; ic < IC; ++ic) {
                    const float * xp = Xp + ic * L + t0;
                    const float * wp = Wb + ic * K * OCp;
                    for (int64_t kw = 0; kw < K; ++kw, xp += dil, wp += OCp) {
                        const __m256 w0 = _mm256_loadu_ps(wp);
                        const __m256 w1 = _mm256_loadu_ps(wp + 8);
                        a00 = _mm256_fmadd_ps(_mm256_broadcast_ss(xp + 0), w0, a00);
                        a01 = _mm256_fmadd_ps(_mm256_broadcast_ss(xp + 0), w1, a01);
                        a10 = _mm256_fmadd_ps(_mm256_broadcast_ss(xp + 1), w0, a10);
                        a11 = _mm256_fmadd_ps(_mm256_broadcast_ss(xp + 1), w1, a11);
                        a20 = _mm256_fmadd_ps(_mm256_broadcast_ss(xp + 2), w0, a20);
                        a21 = _mm256_fmadd_ps(_mm256_broadcast_ss(xp + 2), w1, a21);
                        a30 = _mm256_fmadd_ps(_mm256_broadcast_ss(xp + 3), w0, a30);
                        a31 = _mm256_fmadd_ps(_mm256_broadcast_ss(xp + 3), w1, a31);
                        a40 = _mm256_fmadd_ps(_mm256_broadcast_ss(xp + 4), w0, a40);
                        a41 = _mm256_fmadd_ps(_mm256_broadcast_ss(xp + 4), w1, a41);
                        a50 = _mm256_fmadd_ps(_mm256_broadcast_ss(xp + 5), w0, a50);
                        a51 = _mm256_fmadd_ps(_mm256_broadcast_ss(xp + 5), w1, a51);
                    }
                }
                const int64_t nvalid = (OC - oc0 < 16) ? (OC - oc0) : 16;
                __m256 s0[2] = { a00, a01 }, s1[2] = { a10, a11 }, s2[2] = { a20, a21 };
                __m256 s3[2] = { a30, a31 }, s4[2] = { a40, a41 }, s5[2] = { a50, a51 };
                const __m256 bv0 = _mm256_loadu_ps(bp + oc0), bv1 = _mm256_loadu_ps(bp + oc0 + 8);
                for (int h = 0; h < 2; ++h) {
                    const __m256 bv = h ? bv1 : bv0;
                    s0[h] = _mm256_add_ps(s0[h], bv); s1[h] = _mm256_add_ps(s1[h], bv);
                    s2[h] = _mm256_add_ps(s2[h], bv); s3[h] = _mm256_add_ps(s3[h], bv);
                    s4[h] = _mm256_add_ps(s4[h], bv); s5[h] = _mm256_add_ps(s5[h], bv);
                }
                __m256 t0v[8], t1v[8];
                transpose6x8(s0[0], s1[0], s2[0], s3[0], s4[0], s5[0], t0v);
                transpose6x8(s0[1], s1[1], s2[1], s3[1], s4[1], s5[1], t1v);
                for (int64_t j = 0; j < nvalid; ++j) {
                    const int64_t oc = oc0 + j;
                    store6(o_data + oc * OL + t0, j < 8 ? t0v[j] : t1v[j - 8]);
                }
                st->tiles++;
                st->wbytes += (long) (IC * K * 16 * 4);          // 每个 tile 重读 16 oc 的权重切片
                st->xbytes += (long) (IC * (TB + (K - 1) * dil) * 4);  // 该 tile 覆盖的 X 行 + halo
                st->obytes += (long) (nvalid * TB * 4);
            }
        }
    }
    st->secs += now_s() - t_start;
}

// ---------------------------------------------------------------- phase 1（打包，串行）
static void pack(struct conv * c) {
    const long OCp = c->OCp, IC = c->IC, K = c->K, OC = c->OC;
    for (long oc = 0; oc < OCp; ++oc)
        for (long ic = 0; ic < IC; ++ic) {
            for (long kw = 0; kw < K; ++kw)
                c->Wt[(ic * K + kw) * OCp + oc] = (oc < OC) ? c->w[((oc * IC) + ic) * K + kw] : 0.0f;
            if (oc < OC) c->bp[oc] = c->b ? c->b[oc] : 0.0f;
        }
    for (long ic = 0; ic < IC; ++ic) {
        float * row = c->Xp + ic * c->L;
        memset(row, 0, (size_t) c->pad * sizeof(float));
        memcpy(row + c->pad, c->x + ic * c->T, (size_t) c->T * sizeof(float));
        memset(row + c->pad + c->T, 0, (size_t) c->pad * sizeof(float));
    }
}

// ---------------------------------------------------------------- 线程
struct job {
    const struct conv * c;
    int64_t tb0, tb1, ob2lo, ob2hi;
    struct tstat st;
    int id;
};

static void * worker(void * arg) {
    struct job * j = (struct job *) arg;
    memset(&j->st, 0, sizeof j->st);
    compute_range(j->c, j->tb0, j->tb1, j->ob2lo, j->ob2hi, &j->st);
    return NULL;
}

struct shape { const char * name; long IC, OC, K, dil, T; const char * note; };

static void prepare(struct conv * c, const struct shape * s) {
    memset(c, 0, sizeof *c);
    c->IC = s->IC; c->OC = s->OC; c->K = s->K; c->dil = s->dil; c->T = s->T;
    c->pad = (s->K - 1) / 2 * s->dil;
    c->OL = s->T;
    c->OCp = ((s->OC + 15) / 16) * 16;
    c->KP = s->IC * s->K;
    c->L = s->T + 2 * c->pad;
    const size_t nW = (size_t) s->K * s->IC * s->OC, nX = (size_t) s->IC * c->L;
    float * W = xalloc(nW * 4), * X = xalloc(nX * 4), * B = xalloc((size_t) c->OCp * 4);
    fill_rand(W, nW, 1234u); fill_rand(X, nX, 5678u);
    for (long i = 0; i < c->OCp; i++) B[i] = 0.01f * (float) i;
    c->w = W; c->x = X; c->b = B;
    c->Wt = xalloc((size_t) (c->KP * c->OCp) * 4);
    c->bp = xalloc((size_t) c->OCp * 4);
    c->Xp = xalloc((size_t) c->IC * c->L * 4);
    c->out = xalloc((size_t) c->OL * c->OCp * 4);
}

struct bwjob { const float * buf; size_t n; double secs; };
static void * bw_worker(void * arg) {
    struct bwjob * j = (struct bwjob *) arg;
    size_t n = j->n & ~(size_t) 31;
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    const double t0 = now_s();
    for (size_t i = 0; i < n; i += 32) {
        s0 = _mm256_add_ps(s0, _mm256_loadu_ps(j->buf + i));
        s1 = _mm256_add_ps(s1, _mm256_loadu_ps(j->buf + i + 8));
        s2 = _mm256_add_ps(s2, _mm256_loadu_ps(j->buf + i + 16));
        s3 = _mm256_add_ps(s3, _mm256_loadu_ps(j->buf + i + 24));
    }
    j->secs = now_s() - t0;
    const __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    float sink[8];
    _mm256_storeu_ps(sink, s);
    if (sink[0] == 12345.0f) printf(".");
    return NULL;
}

struct runres {
    double best;
    double thr_secs[8];
    long thr_tiles[8], thr_wb[8], thr_xb[8];
};

static struct runres run(struct conv * c, int nt, const char * split, int reps, float * flushbuf) {
    struct runres r;
    memset(&r, 0, sizeof r);
    r.best = 1e30;
    const int64_t n_tb = (c->OL + TB - 1) / TB;
    const int64_t n_ob2 = c->OCp / 16;
    for (int rep = 0; rep < reps; rep++) {
        if (flushbuf) for (size_t z = 0; z < ((size_t) 16 << 20); z += 8) flushbuf[z] = (float) z;
        struct job jobs[8];
        pthread_t th[8];
        for (int i = 0; i < nt; i++) {
            jobs[i].c = c;
            jobs[i].id = i;
            memset(&jobs[i].st, 0, sizeof jobs[i].st);
            if (!strcmp(split, "oc")) {
                const int64_t per = (n_ob2 + nt - 1) / nt;
                jobs[i].ob2lo = i * per;
                jobs[i].ob2hi = (jobs[i].ob2lo + per > n_ob2) ? n_ob2 : jobs[i].ob2lo + per;
                jobs[i].tb0 = 0;
                jobs[i].tb1 = n_tb;
            } else {
                const int64_t per = (n_tb + nt - 1) / nt;
                jobs[i].tb0 = i * per;
                jobs[i].tb1 = (jobs[i].tb0 + per > n_tb) ? n_tb : jobs[i].tb0 + per;
                jobs[i].ob2lo = 0;
                jobs[i].ob2hi = n_ob2;
            }
        }
        const double t0 = now_s();
        for (int i = 0; i < nt; i++) pthread_create(&th[i], NULL, worker, &jobs[i]);
        for (int i = 0; i < nt; i++) pthread_join(th[i], NULL);
        const double dt = now_s() - t0;
        if (dt < r.best) {
            r.best = dt;
            for (int i = 0; i < nt; i++) {
                r.thr_secs[i] = jobs[i].st.secs;
                r.thr_tiles[i] = jobs[i].st.tiles;
                r.thr_wb[i] = jobs[i].st.wbytes;
                r.thr_xb[i] = jobs[i].st.xbytes;
            }
        }
    }
    return r;
}

int main(int argc, char ** argv) {
    int threads = 1, reps = 3, cold = 0, bw = 0;
    const char * split = "t";
    const char * only = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--split") && i + 1 < argc) split = argv[++i];
        else if (!strcmp(argv[i], "--shape") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--cold")) cold = 1;
        else if (!strcmp(argv[i], "--bw")) bw = 1;
    }
    if (threads > 1 && !getenv("GSV_BOX_FREE")) {
        fprintf(stderr, "refusing a %d-thread run: ask the lead for \"box is free\", "
                        "then re-run with GSV_BOX_FREE=1 (1-thread runs do not need it).\n", threads);
        return 2;
    }
    if (threads < 1 || threads > 8) { fprintf(stderr, "threads out of range\n"); return 2; }

    // archetype：把“容量/流量”从“结构”里分离出来
    const struct shape sh[] = {
        { "real1",  128, 128, 11, 5,  4120, "dec stage1 K=11（最大项，41% MAC）" },
        { "real2",   64,  64,  7, 3,  8240, "dec stage2 K=7" },
        { "real4",   16,  16,  3, 1, 32960, "dec stage4 K=3（尾形，epilogue 已修）" },
        { "l2all",  128, 128, 11, 5,  1024, "Wt 721K + X 512K：全 L2 驻留（容量不是问题）" },
        { "wbig",   256, 256, 11, 5,   512, "Wt 2.9M > L2：权重主导" },
        { "xbig",    16,  16,  3, 1, 65536, "Wt 3K, X 4.2M：激活主导" },
    };
    const int nsh = (int) (sizeof sh / sizeof sh[0]);

    printf("threads=%d split=%s%s%s reps=%d\n", threads, split, cold ? " cold" : " hot",
           bw ? " +bw" : "", reps);
    printf("%-6s %-46s %6s %9s %9s %8s %8s %7s %8s %8s\n",
           "shape", "note", "tiles", "aggGMAC/s", "perThr", "scale", "imbal", "GB/s/thr", "Wt/thr", "X/thr");
    printf("%-6s %-46s %6s %9s %9s %8s %8s %7s %8s %8s\n", "", "", "", "", "GMAC/s", "vs 1t",
           "max/min", "GB/s", "MB", "MB");

    float * flushbuf = cold ? (float *) xalloc((size_t) 64 << 20) : NULL;

    for (int si = 0; si < nsh; si++) {
        if (only && strcmp(only, sh[si].name)) continue;
        struct conv c;
        prepare(&c, &sh[si]);
        pack(&c);
        const double macs = (double) c.OL * c.IC * c.OC * c.K;

        // 同一进程内总是先量一次单线程作基准（real1 只要 ~15 ms），伸缩比才不依赖上一条命令
        struct runres r1 = run(&c, 1, "t", reps > 3 ? 3 : reps, NULL);
        struct runres rn = run(&c, threads, split, reps, flushbuf);

        const double agg1 = macs / r1.best * 1e-9;
        const double agg = macs / rn.best * 1e-9;
        double smax = 0, smin = 1e30, gb_sum = 0;
        long tiles = 0, wb = 0, xb = 0;
        for (int i = 0; i < threads; i++) {
            if (rn.thr_secs[i] <= 0) continue;
            if (rn.thr_secs[i] > smax) smax = rn.thr_secs[i];
            if (rn.thr_secs[i] < smin) smin = rn.thr_secs[i];
            gb_sum += (double) (rn.thr_wb[i] + rn.thr_xb[i]) / rn.thr_secs[i] * 1e-9;
            tiles += rn.thr_tiles[i]; wb += rn.thr_wb[i]; xb += rn.thr_xb[i];
        }
        printf("%-6s %-46s %6ld %9.2f %9.2f %7.2fx %8.2f %7.1f %8.1f %8.1f\n",
               sh[si].name, sh[si].note, tiles, agg, agg / threads, agg / agg1,
               smax / (smin > 0 ? smin : 1), gb_sum / threads,
               (double) wb / threads * 1e-6, (double) xb / threads * 1e-6);
        printf("       └ 单线程基准 %.2f GMAC/s (= %.2f/线程)；本行 %d 线程 %s 切分%s\n",
               agg1, agg1, threads, split, cold ? "，冷 L3" : "，热 L3");

        if (bw) {
            struct bwjob bj[8];
            pthread_t bth[8];
            for (int i = 0; i < threads; i++) {
                bj[i].buf = c.Wt;
                bj[i].n = (size_t) (c.KP * c.OCp) / (size_t) threads;
                bj[i].secs = 0;
            }
            for (int i = 0; i < threads; i++) pthread_create(&bth[i], NULL, bw_worker, &bj[i]);
            for (int i = 0; i < threads; i++) pthread_join(bth[i], NULL);
            double mx = 1e30;
            for (int i = 0; i < threads; i++) if (bj[i].secs > 0 && bj[i].secs < mx) mx = bj[i].secs;
            const double bytes = (double) c.KP * c.OCp * 4;
            printf("       纯读带宽参考: %.1f GB/s (%.1f MB, %d 线程共享同一 buffer) ← DRAM roofline 37.1\n",
                   bytes / mx * 1e-9, bytes * 1e-6, threads);
        }
        free((void *) c.w); free((void *) c.x); free((void *) c.b);
        free(c.Wt); free(c.bp); free(c.Xp); free(c.out);
    }
    free(flushbuf);
    return 0;
}
