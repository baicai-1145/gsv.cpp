// conv_direct_1d CPU 内核的“尾形 epilogue”基准/验证台（单线程，无 ggml 依赖）。
//
// 为什么要有它：dec 的 stage3/4（IC=32/16）只跑到 65%/49% 的 FMA 峰值，而 stage1
// （IC=128, K=11）有 73%。热点内循环本身是 12 条 FMA / 2 次权重装载 / 6 次广播，
// 在 Zen2 上正好是 FMA 端口打满（16 MAC/cycle），所以差距只可能在 epilogue：
//   现在每个 6t×16oc 瓦片要走 tmp[6][8]：12 次向量存 + 96 次标量读（32B store →
//   4B load 的 store-forwarding 有惩罚）+ 96 次标量存（store 口 1/cycle ⇒ 96 cycle）。
//   对 IC=16/K=3 来说主循环才 ~288 cycle，epilogue 直接吃掉 1/3~1/2 的时间。
//
// 本文件逐行复刻 patches/0001 里 ggml_compute_forward_conv_direct_1d 的 phase1/phase2
// （同样的 Wt/Xp 打包、同样的 TB=6 / OCB=16 / TSB/OSB 分块），只把 epilogue 换成不同
// 变体，并用 **逐位一致** 校验（变体输出必须和 variant0 memcmp 相同）保证移植安全。
//
//   ./bench_conv_epi                 # 跑 dec 的 5 个 stage 形状 × K=3/11 × 变体 0..3
//   ./bench_conv_epi --quick         # 只跑 2 个形状（<2 s）
//   ./bench_conv_epi --verify        # 只做逐位校验，不测时间
//
// 单线程。不做多线程计时（共享机器）。
#define _POSIX_C_SOURCE 199309L
#include <immintrin.h>
#include <math.h>
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

// ---------------------------------------------------------------- 单线程 FMA 峰值
static double fma_peak_gflops(int inner) {
    __m256 a0 = _mm256_set1_ps(1.0f), a1 = _mm256_set1_ps(1.1f);
    __m256 a2 = _mm256_set1_ps(1.2f), a3 = _mm256_set1_ps(1.3f);
    __m256 a4 = _mm256_set1_ps(1.4f), a5 = _mm256_set1_ps(1.5f);
    __m256 a6 = _mm256_set1_ps(1.6f), a7 = _mm256_set1_ps(1.7f);
    const __m256 b = _mm256_set1_ps(1.000001f), c = _mm256_set1_ps(1e-7f);
    const int reps = 200, n = 1000;
    double best = 1e30;
    for (int r = 0; r < reps / 20; r++) {
        double t0 = now_s();
        for (int i = 0; i < n; i++) {
            for (int u = 0; u < inner; u++) {
                a0 = _mm256_fmadd_ps(a0, b, c); a1 = _mm256_fmadd_ps(a1, b, c);
                a2 = _mm256_fmadd_ps(a2, b, c); a3 = _mm256_fmadd_ps(a3, b, c);
                a4 = _mm256_fmadd_ps(a4, b, c); a5 = _mm256_fmadd_ps(a5, b, c);
                a6 = _mm256_fmadd_ps(a6, b, c); a7 = _mm256_fmadd_ps(a7, b, c);
            }
        }
        double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    double flops = (double) reps / 20 * n * inner * 8 * 8 * 2;
    __m256 s = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)),
                             _mm256_add_ps(_mm256_add_ps(a4, a5), _mm256_add_ps(a6, a7)));
    float sink[8]; _mm256_storeu_ps(sink, s);
    if (sink[0] == 12345.0f) printf(".");
    return flops / best * 1e-9;
}

// ---------------------------------------------------------------- 6×8 转置
// 6 个“t 行”向量（每个 8 个 oc lane）→ 8 个寄存器，r[j] 的低 6 lane = oc j 的 t0..t5，
// lane 6/7 为 0（所以后面只能存 16B+8B 或 maskstore，不能整 32B 存，否则踩掉下一瓦片）。
static inline void transpose6x8(__m256 s0, __m256 s1, __m256 s2, __m256 s3,
                                __m256 s4, __m256 s5, __m256 r[8]) {
    const __m256 z = _mm256_setzero_ps();
    const __m256 u0 = _mm256_unpacklo_ps(s0, s1), u1 = _mm256_unpackhi_ps(s0, s1);
    const __m256 u2 = _mm256_unpacklo_ps(s2, s3), u3 = _mm256_unpackhi_ps(s2, s3);
    const __m256 u4 = _mm256_unpacklo_ps(s4, s5), u5 = _mm256_unpackhi_ps(s4, s5);
    const __m256d zd = _mm256_castps_pd(z);
    const __m256d v0 = _mm256_unpacklo_pd(_mm256_castps_pd(u0), _mm256_castps_pd(u2));
    const __m256d v1 = _mm256_unpackhi_pd(_mm256_castps_pd(u0), _mm256_castps_pd(u2));
    const __m256d v2 = _mm256_unpacklo_pd(_mm256_castps_pd(u1), _mm256_castps_pd(u3));
    const __m256d v3 = _mm256_unpackhi_pd(_mm256_castps_pd(u1), _mm256_castps_pd(u3));
    const __m256d v4 = _mm256_unpacklo_pd(_mm256_castps_pd(u4), zd);
    const __m256d v5 = _mm256_unpackhi_pd(_mm256_castps_pd(u4), zd);
    const __m256d v6 = _mm256_unpacklo_pd(_mm256_castps_pd(u5), zd);
    const __m256d v7 = _mm256_unpackhi_pd(_mm256_castps_pd(u5), zd);
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

static inline __m256 vleaky(__m256 v, float slope) {
    const __m256 neg = _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_LT_OQ);
    return _mm256_blendv_ps(v, _mm256_mul_ps(v, _mm256_set1_ps(slope)), neg);
}


static inline __m128 vleaky128(__m128 v, float slope) {
    const __m128 neg = _mm_cmp_ps(v, _mm_setzero_ps(), _CMP_LT_OQ);
    return _mm_blendv_ps(v, _mm_mul_ps(v, _mm_set1_ps(slope)), neg);
}
// 转置之后每个寄存器 = 某个 oc 的 t0..t5（lane 6/7 无意义），与输出/res 的布局一致，
// 所以 res 的加法和 leaky 都在这一步做，用 16B + 8B 两半（绝不整 32B 存/读，否则越过行尾）
static inline void epi_emit(float * o, __m256 v, const float * r, int has_res, int has_act, float slope) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    if (has_res) {
        lo = _mm_add_ps(lo, _mm_loadu_ps(r));
        hi = _mm_add_ps(hi, _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *) (r + 4))));
    }
    if (has_act) { lo = vleaky128(lo, slope); hi = vleaky128(hi, slope); }
    _mm_storeu_ps(o, lo);
    _mm_storel_epi64((__m128i *) (o + 4), _mm_castps_si128(hi));
}

// ---------------------------------------------------------------- 复刻的 conv 内核
struct conv {
    long IC, OC, K, dil, T, OL, pad;
    long OCp, KP, L;
    int  has_bias, has_act, has_res;
    float slope;
    float *Wt, *bp, *Xp;            // phase1 产物（[KP][OCp], [OCp], [IC][L]）
    const float *w, *x, *b, *r;     // 模型张量
};

// phase 1：打包 W^T 列（oc 连续）、bias、zero-padded X —— 与补丁逐行一致
static void pack(struct conv * c) {
    const long OCp = c->OCp, IC = c->IC, K = c->K, OC = c->OC;
    for (long oc = 0; oc < OCp; ++oc)
        for (long ic = 0; ic < IC; ++ic) {
            for (long kw = 0; kw < K; ++kw) {
                const float v = (oc < OC) ? c->w[((oc * IC) + ic) * K + kw] : 0.0f;
                c->Wt[(ic * K + kw) * OCp + oc] = v;
            }
            if (oc < OC) c->bp[oc] = c->b ? c->b[oc] : 0.0f;
        }
    for (long ic = 0; ic < IC; ++ic) {
        float * row = c->Xp + ic * c->L;
        memset(row, 0, (size_t) c->pad * sizeof(float));
        memcpy(row + c->pad, c->x + ic * c->T, (size_t) c->T * sizeof(float));
        memset(row + c->pad + c->T, 0, (size_t) c->pad * sizeof(float));
    }
}

// phase 2（单线程版，含 TSB/OSB 分块）
static void compute(struct conv * c, float * o_data, int variant) {
    const long IC = c->IC, OC = c->OC, K = c->K, dil = c->dil, OL = c->OL;
    const long OCp = c->OCp, KP = c->KP, L = c->L;
    const float * Wt = c->Wt, * bp = c->bp, * Xp = c->Xp;
    const int has_bias = c->has_bias, has_act = c->has_act, has_res = c->has_res;
    const float slope = c->slope;
    const int64_t TB = 6;
    const int64_t n_tb = (OL + TB - 1) / TB;
    const int64_t n_ob = OCp / 16;
    const int64_t TSB = (int64_t) (65536 / (TB * IC * 4 + 1));
    const int64_t TSBc = TSB < 1 ? 1 : (TSB > 32 ? 32 : TSB);
    const int64_t OSB0 = (int64_t) (131072 / (16 * KP * 4 + 1));
    const int64_t OSB = OSB0 < 1 ? 1 : (OSB0 > n_ob ? n_ob : OSB0);
    const __m256i mask6 = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0);

    for (int64_t ts = 0; ts < n_tb; ts += TSBc) {
        const int64_t te = (ts + TSBc < n_tb) ? ts + TSBc : n_tb;
        for (int64_t ob = 0; ob < n_ob; ob += OSB) {
            const int64_t oe = (ob + OSB < n_ob) ? ob + OSB : n_ob;
            for (int64_t ob2 = ob; ob2 < oe; ++ob2) {
                const int64_t oc0 = ob2 * 16;
                const float * Wb = Wt + oc0;
                for (int64_t tb = ts; tb < te; ++tb) {
                    const int64_t t0 = tb * TB;
                    if (t0 + TB > OL) {   // 尾巴：标量（每次调用最多一个）
                        for (int64_t t = t0; t < OL; ++t)
                            for (int64_t oc = oc0; oc < (oc0 + 16 < OC ? oc0 + 16 : OC); ++oc) {
                                float acc = has_bias ? bp[oc] : 0.0f;
                                for (int64_t ic = 0; ic < IC; ++ic)
                                    for (int64_t kw = 0; kw < K; ++kw)
                                        acc += Wt[(ic * K + kw) * OCp + oc] * Xp[ic * L + t + kw * dil];
                                if (has_res) acc += c->r[oc * OL + t];
                                if (has_act) acc = (acc > 0.f ? acc : 0.f) + slope * (acc < 0.f ? acc : 0.f);
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
                        const float * xp = Xp + ic * L + t0;   // 注意：Xp 的 pad 已在索引里
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

                    // ---------------- epilogue 变体 ----------------
                    const int64_t nvalid = (OC - oc0 < 16) ? (OC - oc0) : 16;
                    if (variant == 0) {
                        // V0 = 现状：每个 oc 反复把 6 个向量存进 tmp 再标量读回来
                        float tmp[TB][8];
                        for (int64_t j = 0; j < 16; ++j) {
                            const int64_t oc = oc0 + j;
                            if (oc >= OC) break;
                            const int hi = j >= 8 ? 1 : 0;
                            const int ch = (int) (j & 7);
                            const float bv = has_bias ? bp[oc] : 0.0f;
                            _mm256_storeu_ps(&tmp[0][0], hi ? a01 : a00);
                            _mm256_storeu_ps(&tmp[1][0], hi ? a11 : a10);
                            _mm256_storeu_ps(&tmp[2][0], hi ? a21 : a20);
                            _mm256_storeu_ps(&tmp[3][0], hi ? a31 : a30);
                            _mm256_storeu_ps(&tmp[4][0], hi ? a41 : a40);
                            _mm256_storeu_ps(&tmp[5][0], hi ? a51 : a50);
                            for (int64_t i = 0; i < TB; ++i) {
                                float v = bv + tmp[i][ch];
                                if (has_res) v += c->r[oc * OL + t0 + i];
                                if (has_act) v = (v > 0.f ? v : 0.f) + slope * (v < 0.f ? v : 0.f);
                                o_data[oc * OL + t0 + i] = v;
                            }
                        }
                    } else if (variant == 4) {
                        // V4 = 上界诊断：主循环 + 最小落盘量，不做 epilogue 算术/转置。
                        // 不是合法卷积，只用来回答“epilogue 还有多少油水”。
                        float tmp[TB][16];
                        _mm256_storeu_ps(&tmp[0][0], a00); _mm256_storeu_ps(&tmp[0][8], a01);
                        _mm256_storeu_ps(&tmp[1][0], a10); _mm256_storeu_ps(&tmp[1][8], a11);
                        _mm256_storeu_ps(&tmp[2][0], a20); _mm256_storeu_ps(&tmp[2][8], a21);
                        _mm256_storeu_ps(&tmp[3][0], a30); _mm256_storeu_ps(&tmp[3][8], a31);
                        _mm256_storeu_ps(&tmp[4][0], a40); _mm256_storeu_ps(&tmp[4][8], a41);
                        _mm256_storeu_ps(&tmp[5][0], a50); _mm256_storeu_ps(&tmp[5][8], a51);
                        for (int64_t j = 0; j < nvalid; ++j) {
                            float * o = o_data + (oc0 + j) * OL + t0;
                            for (int64_t i = 0; i < TB; ++i) o[i] = tmp[i][j];
                        }
                    } else if (variant == 1) {
                        // V1 = 最小改动：12 次向量存一次做完（不是每 oc 重存 6 次），
                        //      算术全部在标量侧
                        float tmp[TB][16];
                        _mm256_storeu_ps(&tmp[0][0],        a00); _mm256_storeu_ps(&tmp[0][8], a01);
                        _mm256_storeu_ps(&tmp[1][0],        a10); _mm256_storeu_ps(&tmp[1][8], a11);
                        _mm256_storeu_ps(&tmp[2][0],        a20); _mm256_storeu_ps(&tmp[2][8], a21);
                        _mm256_storeu_ps(&tmp[3][0],        a30); _mm256_storeu_ps(&tmp[3][8], a31);
                        _mm256_storeu_ps(&tmp[4][0],        a40); _mm256_storeu_ps(&tmp[4][8], a41);
                        _mm256_storeu_ps(&tmp[5][0],        a50); _mm256_storeu_ps(&tmp[5][8], a51);
                        for (int64_t j = 0; j < nvalid; ++j) {
                            const int64_t oc = oc0 + j;
                            const float bv = has_bias ? bp[oc] : 0.0f;
                            float * o = o_data + oc * OL + t0;
                            const float * r = c->r + oc * OL + t0;
                            for (int64_t i = 0; i < TB; ++i) {
                                float v = bv + tmp[i][j];
                                if (has_res) v += r[i];
                                if (has_act) v = (v > 0.f ? v : 0.f) + slope * (v < 0.f ? v : 0.f);
                                o[i] = v;
                            }
                        }
                    } else {
                        // V2：bias(向量, lane=oc) → 6×8 转置 → 转置后按 128 位半边加 res / leaky
                        // → 16B+8B 两次写。转置后 lane = t，正好和输出/res 的行布局一致。
                        __m256 s0[2] = { a00, a01 }, s1[2] = { a10, a11 }, s2[2] = { a20, a21 };
                        __m256 s3[2] = { a30, a31 }, s4[2] = { a40, a41 }, s5[2] = { a50, a51 };
                        __m256 bias[2] = { _mm256_loadu_ps(bp + oc0), _mm256_loadu_ps(bp + oc0 + 8) };
                        if (!has_bias) { bias[0] = _mm256_setzero_ps(); bias[1] = _mm256_setzero_ps(); }
                        const int act_late = has_act && has_res;   // 顺序必须是 bias+acc → +res → act
                        for (int h = 0; h < 2; ++h) {
                            s0[h] = _mm256_add_ps(s0[h], bias[h]);
                            s1[h] = _mm256_add_ps(s1[h], bias[h]);
                            s2[h] = _mm256_add_ps(s2[h], bias[h]);
                            s3[h] = _mm256_add_ps(s3[h], bias[h]);
                            s4[h] = _mm256_add_ps(s4[h], bias[h]);
                            s5[h] = _mm256_add_ps(s5[h], bias[h]);
                            if (has_act && !act_late) {
                                s0[h] = vleaky(s0[h], slope); s1[h] = vleaky(s1[h], slope);
                                s2[h] = vleaky(s2[h], slope); s3[h] = vleaky(s3[h], slope);
                                s4[h] = vleaky(s4[h], slope); s5[h] = vleaky(s5[h], slope);
                            }
                        }
                        __m256 t0v[8], t1v[8];
                        transpose6x8(s0[0], s1[0], s2[0], s3[0], s4[0], s5[0], t0v);
                        transpose6x8(s0[1], s1[1], s2[1], s3[1], s4[1], s5[1], t1v);
                        for (int64_t j = 0; j < 8; ++j) {
                            if (j < nvalid)
                                epi_emit(o_data + (oc0 + j) * OL + t0, t0v[j],
                                         c->r + (oc0 + j) * OL + t0, has_res, has_act && act_late, slope);
                            if (j + 8 < nvalid)
                                epi_emit(o_data + (oc0 + j + 8) * OL + t0, t1v[j],
                                         c->r + (oc0 + j + 8) * OL + t0, has_res, has_act && act_late, slope);
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- 驱动
struct shape { const char * name; long IC, OC, K, dil, T; };

static void run_one(const struct shape * s, int has_act, int has_res, int verify_only,
                    double peak) {
    struct conv c;
    memset(&c, 0, sizeof c);
    const long dil = has_res ? 1 : s->dil;        // conv2 的 dil 恒为 1
    const long pad = (s->K - 1) / 2 * dil;         // 输出长度 = 输入长度
    c.IC = s->IC; c.OC = s->OC; c.K = s->K; c.dil = dil; c.T = s->T;
    c.pad = pad;
    c.OL = s->T;
    c.OCp = ((s->OC + 15) / 16) * 16;
    c.KP = s->IC * s->K;
    c.L = s->T + 2 * pad;
    c.has_bias = 1; c.has_act = has_act; c.has_res = has_res; c.slope = 0.1f;
    c.Wt = xalloc((size_t) (c.KP * c.OCp + c.OCp + c.IC * c.L) * sizeof(float));
    c.bp = c.Wt + c.KP * c.OCp;
    c.Xp = c.bp + c.OCp;
    float * W = xalloc((size_t) c.KP * c.OCp * sizeof(float));
    float * X = xalloc((size_t) c.IC * c.L * sizeof(float));
    float * B = xalloc((size_t) c.OCp * sizeof(float));
    float * R = xalloc((size_t) c.OL * c.OCp * sizeof(float));
    float * O = xalloc((size_t) c.OL * c.OCp * sizeof(float));
    for (long i = 0; i < c.KP * c.OCp; i++) W[i] = (float) ((i * 1103515245u + 12345u) % 2000) / 1000.0f - 1.0f;
    for (long i = 0; i < c.IC * c.L; i++)     X[i] = (float) ((i * 22695477u + 1u) % 2000) / 1000.0f - 1.0f;
    for (long i = 0; i < c.OCp; i++)          B[i] = 0.01f * (float) i;
    for (long i = 0; i < c.OL * c.OCp; i++)   R[i] = (float) ((i * 2654435761u) % 977) / 977.0f - 0.5f;
    c.w = W; c.x = X; c.b = B; c.r = R;

    double tp = now_s();
    pack(&c);
    tp = now_s() - tp;

    const double macs = (double) c.OL * c.IC * c.OC * c.K;
    if (verify_only) {
        compute(&c, O, 0);
        float * ref = xalloc((size_t) c.OL * c.OCp * sizeof(float));
        memcpy(ref, O, (size_t) c.OL * c.OCp * sizeof(float));
        for (int v = 1; v <= 3; v++) {
            memset(O, 0, (size_t) c.OL * c.OCp * sizeof(float));
            compute(&c, O, v);
            const size_t n = (size_t) c.OL * c.OCp;
            size_t bad = n;
            for (size_t i = 0; i < n; i++) if (memcmp(&ref[i], &O[i], 4)) { bad = i; break; }
            printf("    %-38s act=%d res=%d  V%d vs V0: %s", s->name, has_act, has_res, v,
                   bad == n ? "bit-identical" : "*** MISMATCH ***");
            if (bad != n) {
                const size_t oc = bad / (size_t) c.OL, t = bad % (size_t) c.OL;
                printf("  first diff at oc=%zu t=%zu: ref=%.9g got=%.9g (n_diff_rows oc0=%zu t0=%zu)",
                       oc, t, ref[bad], O[bad], oc, t - t % 6);
                size_t nbad = 0;
                for (size_t i = 0; i < n; i++) if (memcmp(&ref[i], &O[i], 4)) nbad++;
                printf("  n_diff=%zu/%zu", nbad, n);
            }
            printf("\n");
            if (bad != n) exit(2);
        }
        free(ref); free(O); free(W); free(X); free(B); free(R); free(c.Wt);
        return;
    }

    double best[5] = { 1e30, 1e30, 1e30, 1e30, 1e30 };
    for (int rep = 0; rep < 3; rep++) {
        for (int v = 0; v <= 4; v++) {
            memset(O, 0, (size_t) c.OL * c.OCp * sizeof(float));
            const double t0 = now_s();
            compute(&c, O, v);
            const double dt = now_s() - t0;
            if (dt < best[v]) best[v] = dt;
        }
    }
    printf("  %-10s IC=%-3ld OC=%-3ld K=%-2ld dil=%ld T=%-6ld | act=%d res=%d | MAC=%.2f GMAC\n",
           s->name, s->IC, s->OC, s->K, s->dil, s->T, has_act, has_res, macs * 1e-9);
    for (int v = 0; v <= 4; v++) {
        const double gflops = macs * 2.0 / best[v] * 1e-9;
        printf("    V%d %-22s %7.2f ms  %6.2f GMAC/s  %5.1f%% peak  (pack %.2f ms, %.1f%%)\n",
               v, v == 0 ? "current tmp[6][8]" : v == 1 ? "store-once+scalar"
                  : v == 2 ? "transpose+2 stores" : v == 3 ? "(unused)" : "UPPER BOUND (no epi)",
               best[v] * 1e3, macs / best[v] * 1e-9, gflops / peak * 100.0,
               tp * 1e3, tp / best[v] * 100.0);
    }
    free(O);
    free(W); free(X); free(B); free(R); free(c.Wt);
}

int main(int argc, char ** argv) {
    int quick = 0, verify_only = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--quick")) quick = 1;
        if (!strcmp(argv[i], "--verify")) verify_only = 1;
    }
    const double peak = fma_peak_gflops(64);
    printf("1-thread AVX2 FMA peak (measured in-process): %.1f GFLOPS\n", peak);

    // dec 的 5 个 stage × 3 个 pair 的 K（每 block 的 pair 是 K=3/7/11，dil=1/3/5）。
    // T 由 MAC 反推：stage0 = 17.0 GMAC = 2060 × 256² × 126（每 block Σ2K = 42，3 block）
    // 后一级 T ×2、C ÷2，所以 MAC 逐级减半。
    const struct shape shapes[] = {
        { "stage0", 256, 256,  3, 1,  2060 },
        { "stage0", 256, 256,  7, 3,  2060 },
        { "stage0", 256, 256, 11, 5,  2060 },
        { "stage1", 128, 128,  3, 1,  4120 },
        { "stage1", 128, 128,  7, 3,  4120 },
        { "stage1", 128, 128, 11, 5,  4120 },
        { "stage2",  64,  64,  3, 1,  8240 },
        { "stage2",  64,  64,  7, 3,  8240 },
        { "stage2",  64,  64, 11, 5,  8240 },
        { "stage3",  32,  32,  3, 1, 16480 },
        { "stage3",  32,  32,  7, 3, 16480 },
        { "stage3",  32,  32, 11, 5, 16480 },
        { "stage4",  16,  16,  3, 1, 32960 },
        { "stage4",  16,  16,  7, 3, 32960 },
        { "stage4",  16,  16, 11, 5, 32960 },
    };
    if (verify_only) {
        // 小形状：本地（仿真）就能跑完，但把代码路径都盖到：
        //   T 非 6 的倍数/整倍数、OC 不是 16 的倍数（部分 oc 块）、OCp>OC、尾瓦片标量路径
        const struct shape vsh[] = {
            { "small/tail",  16, 16,  3, 1,  37 },
            { "small/exact", 16, 16,  3, 1,  12 },
            { "small/oc24",   8, 24,  7, 3,  13 },
            { "small/k11",   32, 32, 11, 5,  61 },
        };
        printf("bit-exactness of epilogue variants vs the current one:\n");
        for (size_t i = 0; i < sizeof vsh / sizeof vsh[0]; i++) {
            run_one(&vsh[i], 1, 0, 1, peak);
            run_one(&vsh[i], 0, 1, 1, peak);
        }
        return 0;
    }
    const int n = quick ? 2 : (int) (sizeof shapes / sizeof shapes[0]);
    for (int i = 0; i < n; i++) {
        run_one(&shapes[i], 1, 0, 0, peak);   // conv1：bias + leaky
        run_one(&shapes[i], 0, 1, 0, peak);   // conv2：bias + residual
    }
    return 0;
}
