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
        constexpr int64_t TBZ = 12;               // micro-kernel tap count (compile-time)
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

                // zero the whole row: covers t >= OL, the padded t tail and out-of-range taps
                for (int64_t t = 0; t < OLp; t += 16) {
                    _mm256_storeu_si256((__m256i *) (row + t), _mm256_setzero_si256());
                    _mm256_storeu_si256((__m256i *) (row + t + 8), _mm256_setzero_si256());
                }

                // interior: both taps in range, 8 t at a time
                const int64_t tlo = std::max<int64_t>(0, odd_tail ? -off_a : std::max(-off_a, -off_b));
                const int64_t thi = std::min<int64_t>(OL, odd_tail ? (T - off_a) : std::min(T - off_a, T - off_b));
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

        auto store_block = [&](const __m512 & a0, const __m512 & a1, const __m512 & a2,
                               const __m512 & a3, const __m512 & a4, const __m512 & a5,
                               int64_t ocb, int64_t tstart, int64_t tn) {
            alignas(64) float tmp[TBZ][16];
            _mm512_store_ps(tmp[0], a0); _mm512_store_ps(tmp[1], a1);
            _mm512_store_ps(tmp[2], a2); _mm512_store_ps(tmp[3], a3);
            _mm512_store_ps(tmp[4], a4); _mm512_store_ps(tmp[5], a5);
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

        // 12 taps x 16 oc: twelve independent accumulators give a 12-cycle reuse distance against a
        // ~4-6 cycle vdpbf16ps latency, so the single dp port stays busy. With 6 taps the reuse
        // distance equalled the latency and the port idled (measured IPC 2.08 against 2.96 for f32).
        // Loop order matters more than the micro-kernel here: with oc outer, every oc block re-reads
        // the whole im2col buffer from DRAM (KPP*OLp*4 bytes, i.e. K times the activation, once per
        // block - 371 MB for the largest dec layer against 67 MB for the f32 kernel which needs no
        // im2col at all). With the t block outer, one 12*KPP*4 = 34 KB Xb tile stays in L1 and it is
        // the 360 KB weight tile that streams, from L2 instead of DRAM.
        for (int64_t tb = tb0; tb < tb1; ++tb) {
            const int64_t tstart = tb * TBZ;
            for (int64_t oc0 = 0; oc0 < OCp; oc0 += 16) {
                __m512 a0 = _mm512_setzero_ps();
                __m512 a1 = _mm512_setzero_ps();
                __m512 a2 = _mm512_setzero_ps();
                __m512 a3 = _mm512_setzero_ps();
                __m512 a4 = _mm512_setzero_ps();
                __m512 a5 = _mm512_setzero_ps();
                __m512 a6 = _mm512_setzero_ps();
                __m512 a7 = _mm512_setzero_ps();
                __m512 a8 = _mm512_setzero_ps();
                __m512 a9 = _mm512_setzero_ps();
                __m512 a10 = _mm512_setzero_ps();
                __m512 a11 = _mm512_setzero_ps();
                for (int64_t kp2 = 0; kp2 < KPP; ++kp2) {
                    const __m512bh wv0 = m512bh(_mm512_loadu_si512((const void *) (Wb + kp2 * OCp + oc0)));
                    const uint32_t * xp = Xb + kp2 * OLp + tstart;
                    a0 = _mm512_dpbf16_ps(a0, m512bh(_mm512_set1_epi32((int) xp[0])), wv0);
                    a1 = _mm512_dpbf16_ps(a1, m512bh(_mm512_set1_epi32((int) xp[1])), wv0);
                    a2 = _mm512_dpbf16_ps(a2, m512bh(_mm512_set1_epi32((int) xp[2])), wv0);
                    a3 = _mm512_dpbf16_ps(a3, m512bh(_mm512_set1_epi32((int) xp[3])), wv0);
                    a4 = _mm512_dpbf16_ps(a4, m512bh(_mm512_set1_epi32((int) xp[4])), wv0);
                    a5 = _mm512_dpbf16_ps(a5, m512bh(_mm512_set1_epi32((int) xp[5])), wv0);
                    a6 = _mm512_dpbf16_ps(a6, m512bh(_mm512_set1_epi32((int) xp[6])), wv0);
                    a7 = _mm512_dpbf16_ps(a7, m512bh(_mm512_set1_epi32((int) xp[7])), wv0);
                    a8 = _mm512_dpbf16_ps(a8, m512bh(_mm512_set1_epi32((int) xp[8])), wv0);
                    a9 = _mm512_dpbf16_ps(a9, m512bh(_mm512_set1_epi32((int) xp[9])), wv0);
                    a10 = _mm512_dpbf16_ps(a10, m512bh(_mm512_set1_epi32((int) xp[10])), wv0);
                    a11 = _mm512_dpbf16_ps(a11, m512bh(_mm512_set1_epi32((int) xp[11])), wv0);
                }
                alignas(64) float tmp[TBZ][16];
                _mm512_store_ps(tmp[0], a0);
                _mm512_store_ps(tmp[1], a1);
                _mm512_store_ps(tmp[2], a2);
                _mm512_store_ps(tmp[3], a3);
                _mm512_store_ps(tmp[4], a4);
                _mm512_store_ps(tmp[5], a5);
                _mm512_store_ps(tmp[6], a6);
                _mm512_store_ps(tmp[7], a7);
                _mm512_store_ps(tmp[8], a8);
                _mm512_store_ps(tmp[9], a9);
                _mm512_store_ps(tmp[10], a10);
                _mm512_store_ps(tmp[11], a11);
                const int64_t tn = std::min<int64_t>(TBZ, OL - tstart);
                for (int64_t j = 0; j < 16; ++j) {
                    const int64_t oc = oc0 + j;
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
