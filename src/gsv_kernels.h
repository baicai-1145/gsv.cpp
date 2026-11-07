// v2 SoVITS dec 里 ResBlock1 主循环的 AVX2 实现（F32，无量化）。
//
// 背景：ggml 的 ggml_conv_direct_1d（patches/0001）用 6 时间点 x 16 输出通道的
// 寄存器分块直接卷积，内层循环本身能到 FMA 峰值的 ~95%，但**epilogue 是逐元素
// 标量**（把 12 个 ymm 累加器经 tmp[6][8] 转置后一元素一元素地写），当 IC*K 不大时
// （tail stage 的 16/32 通道、K=3）epilogue 与主循环同量级，实测只有 25%~56% 峰值。
//
// 这里的取向是「时间点在向量维、输出通道在标量维」：
//     acc[c] = Σ_{ic,m} x[t..t+7, ic] · w[m, ic, c]        （c 走寄存器，t 走 lane）
// 于是 epilogue 天然是向量化的：bias / leaky / residual 全在 ymm 里算完，直接
// _mm256_storeu_ps 落到 [T, C] 布局（t 是 lane 维 ⇒ 连续），不需要任何转置。
// 每 (ic,m) 只要 1 次 x 装填 + OB 次权重广播 + OB*NT 次 FMA：
//     OB=4、NT=2（一次 16 个 t）⇒ 6 loads / 8 FMA = 3 cycle vs 4 cycle ⇒ FMA 受限，
// 理论上可以贴着峰值跑；权重按 [K][IC][OC]（oc 最快）重排，一个 oc 块
// （4 通道）的权重 = K*IC*16 B，stage1 只要 22.5 KB ⇒ L1 常驻。
//
// 线程划分：(输出通道块, 时间块) 的瓦片按「每个线程一段连续时间」切开，线程内把
// 一个 ResBlock1 的 conv1 → leaky → conv2 → +res 直接在寄存器/本地缓冲里串起来，
// stage 内没有任何 per-node 屏障；块边界需要的 halo 行由各线程**冗余重算**
// （最多 ~90 行 / chunk，< 3%），因此每个输出行只属于一个线程 ⇒ 同线程数下逐位可复现。
#ifndef GSV_KERNELS_H
#define GSV_KERNELS_H

#include <cstddef>

// 一对 conv（ResBlock1 的一次 xt = c2(leaky(c1(leaky(x)))); x += xt）
struct gsv_rb_pair {
    const float * w1 = nullptr;   // [K][C][C]，oc 最快（见 gsv_v2_repack_w）
    const float * b1 = nullptr;   // [C]
    const float * w2 = nullptr;   // [K][C][C]
    const float * b2 = nullptr;   // [C]
    int K = 3;                    // 卷积核长（奇数）
    int dil = 1;                  // conv1 的膨胀；conv2 恒为 1
    int pad1 = 1;                 // conv1 的 pad = (K/2)*dil
    int pad2 = 1;                 // conv2 的 pad = K/2
};

// 一个 stage 的 ResBlock1 组（n_blocks 个块，每块 n_pairs 对 conv）
struct gsv_rb_stage {
    int T = 0;                    // 采样点数
    int C = 0;                    // 通道数
    int n_blocks = 3;
    int n_pairs = 3;
    const gsv_rb_pair * pairs = nullptr;   // [n_blocks * n_pairs]，按 (块, 对) 排
};

// out[t, c] = Σ_blk ResBlock1_blk(in)[t, c]（in/out 都是 ggml 的 [T, C] t 主序）
// 不做 1/n_res 缩放、不做 leaky —— 与图路径完全一致：这两步由图里下一级 conv 的
// in_scale / in_slope 负责。n_threads 只是线程数，输出与线程数无关（逐位一致）。
void gsv_v2_dec_resblocks(const gsv_rb_stage & st, const float * in, float * out, int n_threads);

// ggml 的 [K, IC, OC]（oc 最慢，stride = K*IC）→ [K][IC][OC]（oc 最快）
void gsv_v2_repack_w(const float * src, float * dst, int K, int IC, int OC);

#endif  // GSV_KERNELS_H
