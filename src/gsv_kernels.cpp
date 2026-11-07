// v2 SoVITS dec ResBlock1 主循环的 AVX2 实现。设计取舍见 gsv_kernels.h 顶部注释。
//
// 关键点：
//   * 累加器取向：lane = 时间点（8 个 t），寄存器 = 输出通道（OB=4）。
//     于是 epilogue（bias / leaky / +residual）全是向量运算，直接 storeu 到
//     [T, C] 布局，没有任何转置或逐元素标量写。
//   * 一次算 16 个 t（两份 8 lane）以摊平权重广播：每 (ic,m) 2 次 x 装填 +
//     4 次广播 + 8 次 FMA = 6 loads / 8 FMA（3 cycle vs 4 cycle）⇒ FMA 受限。
//   * 权重按 [K][IC][OC]（oc 最快）重排，一个 oc 块的权重只有 K*IC*16 B。
//   * conv1 的 leaky(0.1) 输入变换在装填时做、输出 leaky 在 epilogue 里做，
//     所以 conv2 的输入不需要任何变换；conv2 的 residual 也从寄存器里加完再写。
//   * 一个线程包一段连续时间行的**整条** ResBlock 链（3 块 × 3 对），
//     stage 内没有屏障；块边界需要的 halo 行由各线程冗余重算（<3%，stage0 ~17%）。
#include "gsv_kernels.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#if defined(__AVX2__)

#include <immintrin.h>

namespace {

constexpr int OB = 4;      // 一次处理的输出通道数（寄存器 = OB 个通道 × 2 个 8-lane）
constexpr int NT = 16;     // 一次处理的输出行数（2 × 8）

// leaky_relu(v, 0.1)：与 ggml 的 epilogue 同口径（负值乘 0.1，正值原样，零原样）
inline __m256 vleaky(__m256 v) {
    const __m256 neg = _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_LT_OQ);
    return _mm256_blendv_ps(v, _mm256_mul_ps(v, _mm256_set1_ps(0.1f)), neg);
}

// 一个 16 行 × OB 通道的输出瓦片。orow/yr 用的是缓冲行号（全局行 - phys_lo）。
//   IN_ACT  : 装填 x 时做 leaky（conv1）
//   OUT_ACT : 写 y 前做 leaky（conv1）
//   HAS_RES : 写 y 前加 residual（conv2）
template <bool IN_ACT, bool OUT_ACT, bool HAS_RES>
inline void conv_tile(const float * __restrict x, int ldx,
                      const float * __restrict w, const float * __restrict b,
                      float * __restrict y, int ldy, const float * __restrict res,
                      int C, int K, int dil, int xrow, int orow) {
    __m256 a0[OB], a1[OB];
    for (int j = 0; j < OB; j++) {
        const __m256 bv = _mm256_set1_ps(b[j]);
        a0[j] = bv;
        a1[j] = bv;
    }
    for (int ic = 0; ic < C; ic++) {
        const float * xr = x + (size_t) ic * ldx + xrow;
        const float * wp = w + (size_t) ic * C;
        for (int m = 0; m < K; m++) {
            __m256 xv0 = _mm256_loadu_ps(xr);
            __m256 xv1 = _mm256_loadu_ps(xr + 8);
            if (IN_ACT) { xv0 = vleaky(xv0); xv1 = vleaky(xv1); }
            for (int j = 0; j < OB; j++) {
                const __m256 wv = _mm256_broadcast_ss(wp + j);
                a0[j] = _mm256_fmadd_ps(xv0, wv, a0[j]);
                a1[j] = _mm256_fmadd_ps(xv1, wv, a1[j]);
            }
            xr += dil;
            wp += (size_t) C * C;
        }
    }
    for (int j = 0; j < OB; j++) {
        __m256 v0 = a0[j], v1 = a1[j];
        if (OUT_ACT) { v0 = vleaky(v0); v1 = vleaky(v1); }
        if (HAS_RES) {
            const size_t o = (size_t) orow + (size_t) j * ldy;
            v0 = _mm256_add_ps(v0, _mm256_loadu_ps(res + o));
            v1 = _mm256_add_ps(v1, _mm256_loadu_ps(res + o + 8));
        }
        const size_t o = (size_t) orow + (size_t) j * ldy;
        _mm256_storeu_ps(y + o, v0);
        _mm256_storeu_ps(y + o + 8, v1);
    }
}

// 输出全局行 [glo, ghi)，写进 y（缓冲行 = 全局行 - phys_lo）。末尾不足 16 行的
// 瓦片照样跑满 16 行（写到 slack 区，调用方保证缓冲两侧各有 ≥ NT+EXT 行 slack），
// 这样任何范围都不需要标量尾巴。
//
// 分块顺序（很关键）：oc-super 外层 → t-chunk 内层 → 瓦片。若拿 t 当最外层，
// 每个 16 行瓦片都要扫一遍整个权重矩阵（stage1 K=11 就是 721 KB），实测直接从
// DRAM 拉 741 MB/conv，比计算本身还慢。现在 oc-super=32 通道的权重（≤ 180 KB）
// 在整个 t-chunk（256 行 = 128 KB 激活）里常驻 L2，权重只扫一遍，激活重读
// C/32 次，两边合计 ~10 MB/conv 量级。
template <bool IN_ACT, bool OUT_ACT, bool HAS_RES>
void conv_range(const float * x, int ldx, const float * w, const float * b,
                float * y, int ldy, const float * res,
                int C, int K, int dil, int pad, int glo, int ghi, int phys_lo) {
    constexpr int OC_SUPER = 32;
    constexpr int TCHUNK = 256;
    const int lo = glo - phys_lo, hi = ghi - phys_lo;
    for (int ocs = 0; ocs < C; ocs += OC_SUPER) {
        const int oce = std::min(C, ocs + OC_SUPER);
        for (int tc = lo; tc < hi; tc += TCHUNK) {
            const int tce = std::min(hi, tc + TCHUNK);
            for (int t = tc; t < tce; t += NT) {
                const int xrow = t - pad;
                for (int oc0 = ocs; oc0 < oce; oc0 += OB) {
                    conv_tile<IN_ACT, OUT_ACT, HAS_RES>(
                            x, ldx, w + (size_t) oc0, b + oc0,
                            y + (size_t) oc0 * ldy, ldy,
                            (res && HAS_RES) ? res + (size_t) oc0 * ldy : nullptr,
                            C, K, dil, xrow, t);
                }
            }
        }
    }
}

// out[a..b) += x（严格按边界，不写 slack）
void add_rows(float * out, const float * x, int ld, int xoff, int a, int b, int T, int C) {
    const int n = b - a;
    for (int c = 0; c < C; c++) {
        float * d = out + (size_t) a + (size_t) T * c;
        const float * s = x + (size_t) xoff + (size_t) ld * c;
        int i = 0;
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(d + i, _mm256_add_ps(_mm256_loadu_ps(d + i), _mm256_loadu_ps(s + i)));
        for (; i < n; i++) d[i] += s[i];
    }
}

void zero_rows(float * out, int ld, int a, int b, int C) {
    for (int c = 0; c < C; c++) std::memset(out + (size_t) a + (size_t) ld * c, 0, (size_t) (b - a) * 4);
}

// 全局边界处中间张量必须补 0：把缓冲行 [r0, r1) 里落在全局 [0, T) *之外* 的部分清 0。
//
// 为什么这一步不能省：conv 的零填充指的是「输入张量在张量边界外为 0」。
//   * conv2 的 m 抽头越界时，参考实现取的是 y[越界] = 0；
//   * 下一对的 conv1 抽头越界时，取的是 x[越界] = 0。
// 而 halo 重算会把 y/x 的越界行“算出来”（= leaky(bias + conv) ≠ 0），
// 直接当抽头用就和参考实现不一样（小 K 时差 1~4%，K=11 时更大）。
// 边界只在首/末线程出现，代价是 O(widen × C) 个 0，可忽略。
void zero_outside(float * buf, int ld, int rows, int r0, int r1, int phys_lo, int T, int C) {
    int b0 = std::max(r0, 0), b1 = std::min(r1, rows);
    if (b0 >= b1) return;
    const int v0 = std::max(0, -phys_lo);          // 全局行 0 对应的缓冲行
    const int v1 = std::min(T - phys_lo, rows);    // 全局行 T 对应的缓冲行
    if (v0 > b0) zero_rows(buf, ld, b0, std::min(v0, b1), C);
    if (v1 < b1) zero_rows(buf, ld, std::max(v1, b0), b1, C);
}

// 一个线程：负责最终输出的全局行 [a, b)
void stage_worker(const gsv_rb_stage & st, const float * in, float * out,
                  int a, int b, int widen, int ext) {
    const int C = st.C, T = st.T, NP = st.n_pairs, NB = st.n_blocks;

    // 缓冲区几何：[a-H, b+H)，全局行之外补 0；H 里含 EXT（瓦片尾巴多读）和 NT（多发写）
    const int H = widen + ext + NT;
    const int phys_lo = a - H;
    const int rows = (b - a) + 2 * H;
    const int ld = rows;

    std::vector<float> xb((size_t) rows * C), yb((size_t) rows * C);
    const int g0 = std::max(0, phys_lo), g1 = std::min(T, phys_lo + rows);

    zero_rows(out, T, a, b, C);
    for (int blk = 0; blk < NB; blk++) {
        std::memset(xb.data(), 0, xb.size() * 4);
        for (int c = 0; c < C; c++)
            std::memcpy(&xb[(size_t) (g0 - phys_lo) + (size_t) ld * c],
                        in + (size_t) g0 + (size_t) T * c, (size_t) (g1 - g0) * 4);

        const gsv_rb_pair * pr = st.pairs + (size_t) blk * NP;
        for (int p = 0; p < NP; p++) {
            // 本对的输出行范围：后面的对还需要更宽的输入，所以这里要放宽
            int lo = a, hi = b;
            for (int q = p + 1; q < NP; q++) {
                lo -= pr[q].pad1 + pr[q].pad2;
                hi += pr[q].pad1 + pr[q].pad2;
            }
            const int ylo = lo - pr[p].pad2, yhi = hi + pr[p].pad2;
            conv_range<true, true, false>(xb.data(), ld, pr[p].w1, pr[p].b1, yb.data(), ld, nullptr,
                                          C, pr[p].K, pr[p].dil, pr[p].pad1, ylo, yhi, phys_lo);
            // conv2 当抽头用得越界 y 行为 0（不是把越界行也算出来的值）
            zero_outside(yb.data(), ld, rows, ylo - phys_lo, yhi - phys_lo, phys_lo, T, C);
            conv_range<false, false, true>(yb.data(), ld, pr[p].w2, pr[p].b2, xb.data(), ld, xb.data(),
                                           C, pr[p].K, 1, pr[p].pad2, lo, hi, phys_lo);
            // 同理：下一对的 conv1 把越界 x 行当 0 用
            zero_outside(xb.data(), ld, rows, lo - phys_lo, hi - phys_lo, phys_lo, T, C);
        }
        add_rows(out, xb.data(), ld, a - phys_lo, a, b, T, C);
    }
}

}  // namespace

void gsv_v2_repack_w(const float * src, float * dst, int K, int IC, int OC) {
    // src: flat = k + ic*K + oc*K*IC ;  dst: flat = m*(IC*OC) + ic*OC + oc
    for (int oc = 0; oc < OC; oc++)
        for (int ic = 0; ic < IC; ic++)
            for (int k = 0; k < K; k++)
                dst[(size_t) k * IC * OC + (size_t) ic * OC + oc] = src[(size_t) k + (size_t) ic * K + (size_t) oc * K * IC];
}

void gsv_v2_dec_resblocks(const gsv_rb_stage & st, const float * in, float * out, int n_threads) {
    if (st.T <= 0 || st.C <= 0 || !in || !out) return;

    int widen = 0, ext = 0;
    for (int blk = 0; blk < st.n_blocks; blk++) {
        int w = 0;
        for (int p = 0; p < st.n_pairs; p++) {
            const gsv_rb_pair & pr = st.pairs[(size_t) blk * st.n_pairs + p];
            w += pr.pad1 + pr.pad2;
            ext = std::max(ext, (pr.K - 1) * pr.dil);
        }
        widen = std::max(widen, w);
    }

    int nt = n_threads > 0 ? n_threads : 1;
    nt = std::min(nt, 16);
    // 每线程至少 4 个瓦片，否则 halo 冗余重算不划算
    while (nt > 1 && st.T / nt < 4 * NT) nt--;
    if (nt <= 1) {
        stage_worker(st, in, out, 0, st.T, widen, ext);
        return;
    }

    const int chunk = ((st.T + nt - 1) / nt + NT - 1) / NT * NT;
    std::vector<std::thread> th;
    th.reserve((size_t) nt);
    for (int p = 0; p < nt; p++) {
        const int a = p * chunk;
        const int b = std::min(st.T, a + chunk);
        if (a >= b) break;
        th.emplace_back(stage_worker, std::cref(st), in, out, a, b, widen, ext);
    }
    for (auto & t : th) t.join();
}

#else  // 非 AVX2 目标：保持可编译（标量参考实现，性能无关）

void gsv_v2_repack_w(const float * src, float * dst, int K, int IC, int OC) {
    for (int oc = 0; oc < OC; oc++)
        for (int ic = 0; ic < IC; ic++)
            for (int k = 0; k < K; k++)
                dst[(size_t) k * IC * OC + (size_t) ic * OC + oc] = src[(size_t) k + (size_t) ic * K + (size_t) oc * K * IC];
}

void gsv_v2_dec_resblocks(const gsv_rb_stage & st, const float * in, float * out, int n_threads) {
    (void) n_threads;
    // 标量参考实现：只保证语义正确，性能无关（非 x86 目标本来也不跑这条路径）
    const int C = st.C, T = st.T;
    std::vector<float> x((size_t) T * C), y((size_t) T * C);
    std::fill(out, out + (size_t) T * C, 0.0f);
    auto tap = [&](const float * buf, int oc, int row) {
        return (row < 0 || row >= T) ? 0.0f : buf[(size_t) row + (size_t) T * oc];
    };
    for (int blk = 0; blk < st.n_blocks; blk++) {
        std::copy(in, in + (size_t) T * C, x.begin());
        for (int p = 0; p < st.n_pairs; p++) {
            const gsv_rb_pair & pr = st.pairs[(size_t) blk * st.n_pairs + p];
            for (int o = 0; o < C; o++)
                for (int t = 0; t < T; t++) {
                    float acc = pr.b1[o];
                    for (int ic = 0; ic < C; ic++)
                        for (int m = 0; m < pr.K; m++) {
                            float v = tap(x.data(), ic, t + m * pr.dil - pr.pad1);
                            if (v < 0) v *= 0.1f;
                            acc += v * pr.w1[((size_t) m * C + ic) * C + o];
                        }
                    y[(size_t) t + (size_t) T * o] = (acc < 0) ? acc * 0.1f : acc;
                }
            for (int o = 0; o < C; o++)
                for (int t = 0; t < T; t++) {
                    float acc = pr.b2[o];
                    for (int ic = 0; ic < C; ic++)
                        for (int m = 0; m < pr.K; m++)
                            acc += tap(y.data(), ic, t + m - pr.pad2) * pr.w2[((size_t) m * C + ic) * C + o];
                    x[(size_t) t + (size_t) T * o] = acc + x[(size_t) t + (size_t) T * o];
                }
        }
        for (size_t i = 0; i < (size_t) T * C; i++) out[i] += x[i];
    }
}

#endif
