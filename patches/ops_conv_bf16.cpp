// bf16 weights + fp32 accumulation for the direct 1D conv (vdpbf16ps): 32 MAC per instruction against
// ~8 for the f32 kernel. Measured 1.68x on the dec stage (4.12 s fixture: 2571.5 -> 1529.0 ms,
// threads=2), numerics cos 0.999917 -> 0.999769 (both operands bf16, every product pair accumulated in
// fp32).
//
// Why its own translation unit with a target pragma instead of building ggml-cpu with AVX-512: the AR
// decode is 2.2x slower when the whole library is AVX-512, and the dec stage is only ~57% of the
// pipeline, so a global AVX-512 build is a net loss. One function with the right ISA is the whole win.
//
// Loop order matters more than the arithmetic here: with oc outer, every oc block re-reads the entire
// im2col buffer from DRAM (K times the activation, 371 MB for the largest dec layer). Putting the t block
// outer keeps one activation tile in L1 and streams the weight tile from L2.

#include "ops.h"
#include "ggml-cpu-impl.h"
#include <algorithm>
#include <cmath>
#include <algorithm>
#include <cstdint>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))

#pragma GCC push_options
#pragma GCC target("avx512f,avx512vl,avx512bf16,f16c,fma")
#include <immintrin.h>
#include <x86intrin.h>
#include <cstdio>
#include <cstdlib>

// GSV_DEC_T=1: phase split via rdtsc (TSC counts reference cycles, so these are wall-clock shares).
static unsigned long long g_dec_t[3] = {0, 0, 0};
static unsigned long long g_dec_n = 0;
unsigned long long g_dec_fma = 0;
static inline void gsv_dec_acc(int slot, unsigned long long d) { g_dec_t[slot] += d; }
struct gsv_dec_rep {
    ~gsv_dec_rep() {
        if (!getenv("GSV_DEC_T") || g_dec_n == 0) return;
        const double tot = (double) (g_dec_t[0] + g_dec_t[1] + g_dec_t[2]);
        fprintf(stderr, "[dec-t] layers=%llu pack=%.1f%% im2col=%.1f%% kernel=%.1f%%"
                        "  fma/layer=%.2f M  kernel_eff=%.1f%%\n",
                g_dec_n, 100.0 * g_dec_t[0] / tot, 100.0 * g_dec_t[1] / tot,
                100.0 * g_dec_t[2] / tot,
                g_dec_fma / (double) g_dec_n / 1e6,
                100.0 * (g_dec_fma / (double) g_dec_n) * 2.0 / (g_dec_t[2] / (double) g_dec_n));
    }
};
static gsv_dec_rep g_dec_rep_inst;
struct gsv_dec_kernel_scope {
    unsigned long long t;
    explicit gsv_dec_kernel_scope(unsigned long long t0) : t(t0) {}
    ~gsv_dec_kernel_scope() { gsv_dec_acc(2, __rdtsc() - t); }
};

static inline uint32_t ggml_f32_to_bf16_bits(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    if ((u & 0x7f800000u) != 0x7f800000u) {
        u += 0x7fffu + ((u >> 16) & 1u);
        u &= 0xffff0000u;
    }
    return u >> 16;
}


bool gsv_conv_direct_1d_bf16(const gsv_conv_bf16_args & args) {
    ggml_threadpool * tpp = args.tp;   // ggml_barrier 要 threadpool 指针
    // 原块体原样搬过来, 只把要用的量落成局部名, 块内代码一行未改
    void * wdata = args.wdata;
    const ggml_tensor * w = args.w, * x = args.x, * bias = args.bias, * res = args.res;
    ggml_tensor * dst = args.dst;
    const int64_t K = args.K, IC = args.IC, OC = args.OC, T = args.T, OL = args.OL;
    const int64_t pad = args.pad, dil = args.dil;
    const float slope = args.slope, in_scale = args.in_scale, in_slope = args.in_slope;
    const bool has_bias = args.bias != nullptr;
    const bool has_res = args.has_res != 0, has_act = args.has_act != 0, has_in = args.has_in != 0;
    (void) slope; (void) in_scale; (void) in_slope; (void) has_act; (void) has_in; (void) has_res;

        const int64_t KP  = IC * K;               // reduction length
        const int64_t KPP = (KP + 1) / 2;         // bf16 pairs
        const int64_t OCp = ((OC + 15) / 16) * 16;
        constexpr int64_t TBZ = 8;                 // micro-kernel tap count (compile-time)
        const int64_t OLp = ((OL + 31) / 32) * 32; // padded tap length (>= OL + TBZ)

        char * wbase = (char *) wdata;
        uint32_t * Wb = (uint32_t *) wbase;                  // [KPP][OCp] pairs
        float    * bp = (float *) (Wb + KPP * OCp);          // [OCp]
        uint32_t * Xb = (uint32_t *) (bp + OCp);             // [OL][KPP] pairs

        const ggml_bf16_t * wd = (const ggml_bf16_t *) w->data;
        const float * x_data = (const float *) x->data;
        const float * b_data = has_bias ? (const float *) bias->data : nullptr;
        const float * r_data = has_res  ? (const float *) res->data  : nullptr;
        float * o_data = (float *) dst->data;

        const int ith = args.ith;
        const int nth = args.nth;

        const unsigned long long _gt0 = __rdtsc();
        // phase 1a: pack weights along the reduction axis; threads split oc columns
        {
            const int64_t per = (OC + nth - 1) / nth;
            const int64_t o0 = std::min<int64_t>(ith * per, OC);
            const int64_t o1 = std::min<int64_t>(o0 + per, OC);
            for (int64_t oc = o0; oc < o1; ++oc) {
                // w is [K, IC, OC]: element (kw, ic, oc) == kp + KP*oc,  kp = ic*K + kw
                const ggml_bf16_t * wcol = wd + oc * KP;
                for (int64_t kp = 0; kp < KP; kp += 2) {
                    const uint32_t lo = (uint32_t) wcol[kp].bits;
                    const uint32_t hi = (kp + 1 < KP) ? (uint32_t) wcol[kp + 1].bits : 0u;
                    Wb[(kp >> 1) * OCp + oc] = lo | (hi << 16);
                }
            }
            const int64_t perb = (OCp + nth - 1) / nth;
            const int64_t b0 = std::min<int64_t>(ith * perb, OCp);
            const int64_t b1 = std::min<int64_t>(b0 + perb, OCp);
            for (int64_t oc = b0; oc < b1; ++oc) bp[oc] = (oc < OC && b_data) ? b_data[oc] : 0.0f;
        }

        { const unsigned long long _gt1 = __rdtsc(); gsv_dec_acc(0, _gt1 - _gt0); }
        const unsigned long long _gt2 = __rdtsc();
        ++g_dec_n;
        // phase 1b: im2col into bf16 pairs, laid out as Xb[kp2][t] so that a 32-bit pair read
        // for consecutive t is contiguous. Pairing is along the reduction axis (kp, kp+1), so the
        // two streams are each contiguous in t and the whole row converts 8 taps at a time:
        //   vcvtne2ps2bf16 -> unpacklo/hi_epi16 -> two 128-bit stores
        // measured before this rewrite: the scalar version cost 4.4 s for the dec stage, i.e. 78% of
        // the bf16 conv time; the dot loop alone is 2.03x faster than the f32 path.
        {
            const int64_t perk = (KPP + nth - 1) / nth;
            const int64_t k0 = std::min<int64_t>(ith * perk, KPP);
            const int64_t k1 = std::min<int64_t>(k0 + perk, KPP);

            auto xform = [&](__m256 v) -> __m256 {
                if (in_scale != 1.0f) v = _mm256_mul_ps(v, _mm256_set1_ps(in_scale));
                if (in_slope != 0.0f) {
                    const __m256 pos = _mm256_max_ps(v, _mm256_setzero_ps());
                    const __m256 neg = _mm256_min_ps(v, _mm256_setzero_ps());
                    v = _mm256_fmadd_ps(_mm256_set1_ps(in_slope), neg, pos);
                }
                return v;
            };

            for (int64_t kp2 = k0; kp2 < k1; ++kp2) {
                uint32_t * row = Xb + kp2 * OLp;
                const int64_t kp_a = 2 * kp2;
                const int64_t kp_b = std::min<int64_t>(2 * kp2 + 1, KP - 1);
                const bool odd_tail = (kp_b == kp_a);   // odd KP: pad the pair with zero, never duplicate
                const float * xa = x_data + (kp_a / K) * T;
                const float * xb = x_data + (kp_b / K) * T;
                const int64_t off_a = (kp_a % K) * dil - pad;
                const int64_t off_b = (kp_b % K) * dil - pad;

                // interior: both taps in range, 8 t at a time
                const int64_t tlo = std::max<int64_t>(0, odd_tail ? -off_a : std::max(-off_a, -off_b));
                const int64_t thi = std::min<int64_t>(OL, odd_tail ? (T - off_a) : std::min(T - off_a, T - off_b));

                // zero only the parts the interior loop will not write. Zeroing the whole row first and
                // then overwriting the middle doubled the write traffic of im2col, which the phase
                // timing put at 17% of the dec stage.
                for (int64_t t = 0; t < tlo; t += 8) {
                    _mm256_storeu_si256((__m256i *) (row + t), _mm256_setzero_si256());
                }
                for (int64_t t = thi; t < OLp; t += 8) {
                    _mm256_storeu_si256((__m256i *) (row + t), _mm256_setzero_si256());
                }
                int64_t t = tlo;
                for (; t + 8 <= thi; t += 8) {
                    const __m256 va = xform(_mm256_loadu_ps(xa + t + off_a));
                    const __m256 vb = odd_tail ? _mm256_setzero_ps() : xform(_mm256_loadu_ps(xb + t + off_b));
                    const __m128i ba = (__m128i) _mm256_cvtneps_pbh(va);
                    const __m128i bb = (__m128i) _mm256_cvtneps_pbh(vb);
                    const __m128i lo = _mm_unpacklo_epi16(ba, bb);   // {a0,b0,a1,b1,a2,b2,a3,b3}
                    const __m128i hi = _mm_unpackhi_epi16(ba, bb);
                    _mm_storeu_si128((__m128i *) (row + t),     lo);
                    _mm_storeu_si128((__m128i *) (row + t + 4), hi);
                }
                // scalar helpers for the edges (same transform as xform, one element at a time)
                auto xf1 = [&](float v) -> float {
                    if (in_scale != 1.0f) v *= in_scale;
                    if (in_slope != 0.0f) v = (v > 0.f ? v : 0.f) + in_slope * (v < 0.f ? v : 0.f);
                    return v;
                };
                // scalar remainder of the interior, then the two edge ranges (one tap out of range)
                for (; t < thi; ++t) {
                    const uint32_t lo = ggml_f32_to_bf16_bits(xf1(xa[t + off_a]));
                    const uint32_t hi = ggml_f32_to_bf16_bits(odd_tail ? 0.0f : xf1(xb[t + off_b]));
                    row[t] = lo | (hi << 16);
                }
                for (int64_t e = 0; e < OL; ++e) {
                    if (e >= tlo && e < thi) continue;
                    const int64_t ia = e + off_a, ib = e + off_b;
                    float va = (ia >= 0 && ia < T) ? xa[ia] : 0.0f;
                    float vb = (!odd_tail && ib >= 0 && ib < T) ? xb[ib] : 0.0f;
                    if (has_in) {
                        if (in_scale != 1.0f) { va *= in_scale; vb *= in_scale; }
                        if (in_slope != 0.0f) {
                            va = (va > 0.f ? va : 0.f) + in_slope * (va < 0.f ? va : 0.f);
                            vb = (vb > 0.f ? vb : 0.f) + in_slope * (vb < 0.f ? vb : 0.f);
                        }
                    }
                    row[e] = ggml_f32_to_bf16_bits(va) | (ggml_f32_to_bf16_bits(vb) << 16);
                }
            }
        }

        ggml_barrier(tpp);

        // phase 2: 6 t x 32 oc micro-kernel, 32 MAC per vdpbf16ps.
        // Two 16-lane accumulator sets share one broadcast of the activation pair, because perf
        // showed the six vpbroadcastd dominating (66% of the kernel samples, vdpbf16ps only 8%):
        // one broadcast now feeds 2 dpbf16ps instead of 1. Tap count is a compile-time constant so
        // every accumulator stays in a register (a runtime trip count made gcc emit a switch).
        const int64_t n_tb = (OL + TBZ - 1) / TBZ;
        const int64_t pert = (n_tb + nth - 1) / nth;
        const int64_t tb0  = std::min<int64_t>(ith * pert, n_tb);
        const int64_t tb1  = std::min<int64_t>(tb0 + pert, n_tb);

        auto store_block = [&](const float tmp[TBZ][16], int64_t ocb, int64_t tstart, int64_t tn) {
            for (int64_t j = 0; j < 16; ++j) {
                const int64_t oc = ocb + j;
                if (oc >= OC) break;
                const float bv = bp[oc];
                float * orow = o_data + oc * OL + tstart;
                for (int64_t i = 0; i < tn; ++i) {
                    float v = bv + tmp[i][j];
                    if (has_res) v = v + r_data[oc * OL + tstart + i];
                    if (has_act) v = slope < 0.f ? tanhf(v)
                                      : (v > 0.f ? v : 0.f) + slope * (v < 0.f ? v : 0.f);
                    orow[i] = v;
                }
            }
        };

        // 8 taps x 32 oc per pass.
        // The bottleneck is the size of the Xb activation tile, NT * KPP * 4 bytes: for the largest dec
        // layer KPP = 128*11/2 = 704, so 12 taps is 33.8 KB - past the 32 KB L1 - and every t block
        // re-fetches it from L2 (perf put 60% of the samples on the vpbroadcastd loads that hit it).
        // 8 taps is 22.0 KB.
        // One batch of 8 broadcasts feeds 16 dpbf16ps, so the activation reads per MAC are
        // amortized over 2 oc blocks as well.
        // Register budget: 16 accumulators + 2 weights + 1 broadcast = 19 of 32 zmm.
        extern unsigned long long g_dec_fma;
        g_dec_fma += (unsigned long long) ((OCp / 32) * KPP * (tb1 - tb0) * 16);
        { const unsigned long long _gt3 = __rdtsc(); gsv_dec_acc(1, _gt3 - _gt2); }
        gsv_dec_kernel_scope _gks(__rdtsc());
        for (int64_t tb = tb0; tb < tb1; ++tb) {
            const int64_t tstart = tb * TBZ;
            const int64_t tn = std::min<int64_t>(TBZ, OL - tstart);
            for (int64_t oc0 = 0; oc0 < OCp; oc0 += 32) {
                const bool has1 = oc0 + 16 < OCp;
                __m512 v0_0 = _mm512_setzero_ps();
                __m512 v0_1 = _mm512_setzero_ps();
                __m512 v0_2 = _mm512_setzero_ps();
                __m512 v0_3 = _mm512_setzero_ps();
                __m512 v0_4 = _mm512_setzero_ps();
                __m512 v0_5 = _mm512_setzero_ps();
                __m512 v0_6 = _mm512_setzero_ps();
                __m512 v0_7 = _mm512_setzero_ps();
                __m512 v1_0 = _mm512_setzero_ps();
                __m512 v1_1 = _mm512_setzero_ps();
                __m512 v1_2 = _mm512_setzero_ps();
                __m512 v1_3 = _mm512_setzero_ps();
                __m512 v1_4 = _mm512_setzero_ps();
                __m512 v1_5 = _mm512_setzero_ps();
                __m512 v1_6 = _mm512_setzero_ps();
                __m512 v1_7 = _mm512_setzero_ps();
                for (int64_t kp2 = 0; kp2 < KPP; ++kp2) {
                    const uint32_t * wp = Wb + kp2 * OCp + oc0;
                    const uint32_t * xp = Xb + kp2 * OLp + tstart;
                    __m512bh xb;
                    const __m512bh w0 = m512bh(_mm512_loadu_si512((const void *) (wp)));
                    const __m512bh w1 = has1
                        ? m512bh(_mm512_loadu_si512((const void *) (wp + 16)))
                        : w0;
                    if (kp2 + 4 < KPP) {
                        const uint32_t * xa = Xb + (kp2 + 4) * OLp + tstart;
                        _mm_prefetch((const char *) xa,          _MM_HINT_T0);
                        _mm_prefetch((const char *) (xa + 8),    _MM_HINT_T0);
                        _mm_prefetch((const char *) (xa + 16),   _MM_HINT_T0);
                        _mm_prefetch((const char *) (xa + 24),   _MM_HINT_T0);
                    }
                    xb = m512bh(_mm512_set1_epi32((int) xp[0]));
                    v0_0 = _mm512_dpbf16_ps(v0_0, xb, w0);
                    v1_0 = _mm512_dpbf16_ps(v1_0, xb, w1);
                    xb = m512bh(_mm512_set1_epi32((int) xp[1]));
                    v0_1 = _mm512_dpbf16_ps(v0_1, xb, w0);
                    v1_1 = _mm512_dpbf16_ps(v1_1, xb, w1);
                    xb = m512bh(_mm512_set1_epi32((int) xp[2]));
                    v0_2 = _mm512_dpbf16_ps(v0_2, xb, w0);
                    v1_2 = _mm512_dpbf16_ps(v1_2, xb, w1);
                    xb = m512bh(_mm512_set1_epi32((int) xp[3]));
                    v0_3 = _mm512_dpbf16_ps(v0_3, xb, w0);
                    v1_3 = _mm512_dpbf16_ps(v1_3, xb, w1);
                    xb = m512bh(_mm512_set1_epi32((int) xp[4]));
                    v0_4 = _mm512_dpbf16_ps(v0_4, xb, w0);
                    v1_4 = _mm512_dpbf16_ps(v1_4, xb, w1);
                    xb = m512bh(_mm512_set1_epi32((int) xp[5]));
                    v0_5 = _mm512_dpbf16_ps(v0_5, xb, w0);
                    v1_5 = _mm512_dpbf16_ps(v1_5, xb, w1);
                    xb = m512bh(_mm512_set1_epi32((int) xp[6]));
                    v0_6 = _mm512_dpbf16_ps(v0_6, xb, w0);
                    v1_6 = _mm512_dpbf16_ps(v1_6, xb, w1);
                    xb = m512bh(_mm512_set1_epi32((int) xp[7]));
                    v0_7 = _mm512_dpbf16_ps(v0_7, xb, w0);
                    v1_7 = _mm512_dpbf16_ps(v1_7, xb, w1);
                }
                alignas(64) float t0[TBZ][16];
                _mm512_store_ps(t0[0], v0_0);
                _mm512_store_ps(t0[1], v0_1);
                _mm512_store_ps(t0[2], v0_2);
                _mm512_store_ps(t0[3], v0_3);
                _mm512_store_ps(t0[4], v0_4);
                _mm512_store_ps(t0[5], v0_5);
                _mm512_store_ps(t0[6], v0_6);
                _mm512_store_ps(t0[7], v0_7);
                store_block(t0, oc0, tstart, tn);
                if (has1) {
                    alignas(64) float t1[TBZ][16];
                _mm512_store_ps(t1[0], v1_0);
                _mm512_store_ps(t1[1], v1_1);
                _mm512_store_ps(t1[2], v1_2);
                _mm512_store_ps(t1[3], v1_3);
                _mm512_store_ps(t1[4], v1_4);
                _mm512_store_ps(t1[5], v1_5);
                _mm512_store_ps(t1[6], v1_6);
                _mm512_store_ps(t1[7], v1_7);
                    store_block(t1, oc0 + 16, tstart, tn);
                }
            }
        }
    return true;
}

#pragma GCC pop_options

bool gsv_conv_bf16_available(void) {
    return __builtin_cpu_supports("avx512bf16") && __builtin_cpu_supports("avx512f");
}

#else

bool gsv_conv_direct_1d_bf16(const gsv_conv_bf16_args &) { return false; }
bool gsv_conv_bf16_available(void) { return false; }

#endif
