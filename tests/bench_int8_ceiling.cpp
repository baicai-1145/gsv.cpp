// Track B（int8）上限测量：在 dec resblock 的真实形状上，用 **ggml 自己的** q8_0
// GEMM 内核量 int8 的天花板，而不是手搓 int8 conv（手搓的输了 ggml 3x）。
//
// 为什么要先量上限：fp32 单线程这些 conv 已经在峰值 76-90%，但 4 线程 dec 只有
// ~25.5 GMAC/s/线程 —— 损失来自每线程都要流一遍 2.9 MB 的 Wt 以及 Xp/O 的 DRAM 流量。
// int8 正好打这个：权重字节减半、每字节 MAC 翻倍。但如果连“把 im2col/量化摊掉之后
// 的天花板”都不到 1.3x，那就不值得做。
//
// 每个形状测四件事（同一份数据、同一个后端、同样的线程数）：
//   1. fp32  GEMM          A_f32(Kdim,M) x W_f32(Kdim,N)      —— ggml 的 f32 mul_mat
//   2. q8_0  GEMM          A_q8 (Kdim,M) x W_q8 (Kdim,N)      —— vec_dot_q8_0_q8_0
//   3. 激活量化 pass       ggml_quantize_chunk(A_f32 -> A_q8) —— 这就是“必须融合进
//                          epilogue 否则吃光收益”的那笔成本
//   4. fp32  conv           ggml_conv_direct_1d_fused          —— 现在的实际路径
//
// 所有样本都打印进程 %CPU（4 线程干净跑应 ≈370-400%），低的就是被抢占的样本。
//
//   ./bench_int8_ceiling --threads 4 --reps 3
//   ./bench_int8_ceiling --check          # 形状/步长自检（对朴素参考 + q8 vs f32）
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <sys/resource.h>
#include <time.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

// 该进程至今的平均 CPU%（4 线程满载 ≈ 400%）
static double cpu_pct(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    const double cpu = (double) ru.ru_utime.tv_sec + (double) ru.ru_stime.tv_sec +
                       1e-6 * ((double) ru.ru_utime.tv_usec + (double) ru.ru_stime.tv_usec);
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const double wall = (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
    return wall > 0 ? 100.0 * cpu / wall : 0.0;
}

// 2D GEMM：A(Kdim,M) x W(Kdim,N) -> (N,M)。fp32 与 q8_0 走同一个 mul_mat。
static bool run_gemm(ggml_backend_t be, ggml_type ta, ggml_type tw, int64_t Kdim, int64_t M,
                     int64_t N, const std::vector<float> * A_f32, const std::vector<float> * W_f32,
                     const std::vector<char> * A_q8, const std::vector<char> * W_q8,
                     std::vector<float> * out, double * best, int reps) {
    ggml_init_params ip = { 8u << 20, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tW = ggml_new_tensor_2d(ctx, tw, Kdim, N);
    ggml_tensor * tA = ggml_new_tensor_2d(ctx, ta, Kdim, M);
    ggml_tensor * C = ggml_mul_mat(ctx, tW, tA);
    ggml_set_output(C);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, C);
    ggml_gallocr_t all = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(all, gf)) { printf("  alloc failed\n"); return false; }
    if (ta == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tA, A_f32->data(), 0, A_f32->size() * 4);
        ggml_backend_tensor_set(tW, W_f32->data(), 0, W_f32->size() * 4);
    } else {
        ggml_backend_tensor_set(tA, A_q8->data(), 0, A_q8->size());
        ggml_backend_tensor_set(tW, W_q8->data(), 0, W_q8->size());
    }
    *best = 1e30;
    for (int r = 0; r < reps; r++) {
        const double t0 = now_s();
        const ggml_status st = ggml_backend_graph_compute(be, gf);
        const double dt = now_s() - t0;
        if (st != GGML_STATUS_SUCCESS) { printf("  compute failed\n"); return false; }
        if (dt < *best) *best = dt;
    }
    if (out) {
        out->assign((size_t) ggml_nelements(C), 0.0f);
        ggml_backend_tensor_get(C, out->data(), 0, out->size() * 4);
    }
    ggml_gallocr_free(all);
    ggml_free(ctx);
    return true;
}

static void fill_rand(std::vector<float> & v, uint32_t seed, float scale) {
    uint32_t s = seed;
    for (size_t i = 0; i < v.size(); i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        v[i] = scale * ((float) (s & 0xffff) / 32768.0f - 1.0f);
    }
}

// ---------------------------------------------------------------- 自检
// (a) fp32 GEMM 对朴素三重循环（小形状）——验证 A/W 的行主序与 Kdim 布局假设
// (b) q8_0 GEMM 对 fp32 GEMM（量化误差量级）——验证 q8 路径真的在做同一件事
static int self_check(ggml_backend_t be) {
    const int64_t Kdim = 96, M = 8, N = 32;
    std::vector<float> A((size_t) Kdim * M), W((size_t) Kdim * N), Cgot;
    fill_rand(A, 12345u, 0.5f);
    fill_rand(W, 999u, 0.5f);
    double dt = 0;
    if (!run_gemm(be, GGML_TYPE_F32, GGML_TYPE_F32, Kdim, M, N, &A, &W, nullptr, nullptr, &Cgot, &dt, 1))
        return 1;
    double maxd = 0, maxv = 0;
    for (int64_t n = 0; n < N; n++)
        for (int64_t m = 0; m < M; m++) {
            double acc = 0;
            for (int64_t k = 0; k < Kdim; k++) acc += (double) W[(size_t) (k + Kdim * n)] * (double) A[(size_t) (k + Kdim * m)];
            const double got = Cgot[(size_t) (n + N * m)];
            maxd = std::max(maxd, std::fabs(acc - got));
            maxv = std::max(maxv, std::fabs(acc));
        }
    printf("  check A: fp32 GEMM(Kdim=%lld,M=%lld,N=%lld) vs scalar  max|diff|=%.3e (max|ref|=%.3f) %s\n",
           (long long) Kdim, (long long) M, (long long) N, maxd, maxv, maxd < 1e-4 * (maxv + 1) ? "OK" : "FAIL");
    const bool ok_a = maxd < 1e-4 * (maxv + 1);

    // (b) q8_0 vs fp32
    const int64_t QK = 32;
    std::vector<char> Aq(ggml_row_size(GGML_TYPE_Q8_0, Kdim) * M), Wq(ggml_row_size(GGML_TYPE_Q8_0, Kdim) * N);
    ggml_quantize_init(GGML_TYPE_Q8_0);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, A.data(), Aq.data(), 0, M, Kdim, nullptr);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, W.data(), Wq.data(), 0, N, Kdim, nullptr);
    std::vector<float> Cq;
    if (!run_gemm(be, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, Kdim, M, N, nullptr, nullptr, &Aq, &Wq, &Cq, &dt, 1))
        return 1;
    double qd = 0;
    for (size_t i = 0; i < Cq.size(); i++) qd = std::max(qd, (double) std::fabs(Cq[i] - Cgot[i]));
    printf("  check B: q8_0 GEMM vs fp32 GEMM  max|diff|=%.3e  (量化误差量级，相对 %.2e) %s\n",
           qd, qd / (maxv + 1e-9), qd < 3e-2 * (maxv + 1) ? "OK" : "FAIL");
    printf("  (Kdim=%lld 不是 32 的倍数时 %s)\n", (long long) Kdim, (Kdim % QK) ? "会被 q8 路径拒绝" : "对齐 OK");
    return (ok_a && qd < 3e-2 * (maxv + 1)) ? 0 : 1;
}

struct shape { const char * name; int64_t T, C, K; };

int main(int argc, char ** argv) {
    int threads = 4, reps = 3, quick = 0, check = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--check")) check = 1;
        if (!strcmp(argv[i], "--quick")) quick = 1;
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    }
    // lead 的规矩：4 线程运行必须先拿到显式的“box is free”。没有这个环境变量就拒绝，
    // 免得又落进别人已开始的 A/B 窗口（两边样本一起废掉）。
    if (threads > 1 && !getenv("GSV_BOX_FREE")) {
        fprintf(stderr,
                "refusing a %d-thread run: this box is shared. Ask the lead for \"box is free\",\n"
                "then re-run with GSV_BOX_FREE=1 (1-thread runs do not need it).\n", threads);
        return 2;
    }
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, threads);
    if (check) {
        printf("shape/stride self-check (backend: %s, threads=%d)\n", ggml_backend_name(be), threads);
        const int rc = self_check(be);
        ggml_backend_free(be);
        return rc;
    }

    // dec 的 5 个 stage（T 由 MAC 反推），每个取 K=3/7/11 三个 tap 数
    // im2col：把已经量化的激活 Xq[IC][T]（int8）展开成 A[(k,ic)][t]（int8）。
    // 权重按收缩维连续重排是加载时一次性的，这里只算激活这一侧——int8 走 GEMM
    // 绕不开的一笔，必须和 quant / GEMM 一起看才知道净收益。
    auto time_im2col = [&](const std::vector<char> & Xq, std::vector<char> & Aq, int64_t T, int64_t IC,
                           int64_t K, int64_t Kdim, int reps_im) -> double {
        double best_im = 1e30;
        for (int r = 0; r < reps_im; r++) {
            const double t0 = now_s();
            for (int64_t t = 0; t < T; t++) {
                char * dst = Aq.data() + (size_t) t * Kdim;
                for (int64_t k = 0; k < K; k++) {
                    const int64_t src_t = t + k;         // dil=1 的示意（真实 dil 只是换偏移）
                    for (int64_t ic = 0; ic < IC; ic++)
                        dst[k * IC + ic] = Xq[(size_t) (ic * T + (src_t < T ? src_t : T - 1))];
                }
                for (int64_t z = K * IC; z < Kdim; z++) dst[z] = 0;   // 32 对齐的 padding
            }
            const double d = now_s() - t0;
            if (d < best_im) best_im = d;
        }
        return best_im;
    };

    const shape sh[] = {
        { "stage0", 2060, 256, 3 }, { "stage0", 2060, 256, 7 }, { "stage0", 2060, 256, 11 },
        { "stage1", 4120, 128, 3 }, { "stage1", 4120, 128, 7 }, { "stage1", 4120, 128, 11 },
        { "stage2", 8240,  64, 3 }, { "stage2", 8240,  64, 7 }, { "stage2", 8240,  64, 11 },
        { "stage3", 16480, 32, 3 }, { "stage3", 16480, 32, 7 }, { "stage3", 16480, 32, 11 },
        { "stage4", 32960, 16, 3 }, { "stage4", 32960, 16, 7 }, { "stage4", 32960, 16, 11 },
    };
    const int nsh = (int) (sizeof sh / sizeof sh[0]);
    const int lim = quick ? 3 : nsh;

    printf("threads=%d reps=%d backend=%s\n", threads, reps, ggml_backend_name(be));
    printf("%-23s %9s %6s %6s | %8s %8s | %8s %8s | %7s %6s | %8s | %6s %6s\n",
           "shape", "useful", "Kdim", "pad%", "fp32GEM", "q8GEM", "fp32/1thr", "q8/1thr",
           "quant", "im2col", "conv", "q8/conv", "e2e/conv");
    printf("%-23s %9s %6s %6s | %8s %8s | %8s %8s | %7s %6s | %8s | %6s %6s\n",
           "", "GMAC", "", "", "GMAC/s", "GMAC/s", "GMAC/s", "GMAC/s", "quant", "im2col", "conv", "q8/conv", "e2e/conv");

    ggml_quantize_init(GGML_TYPE_Q8_0);
    double t_fp32 = 0, t_q8 = 0, t_conv = 0, t_quant = 0, t_im2col = 0, m_useful = 0;
    for (int i = 0; i < lim; i++) {
        const int64_t T = sh[i].T, C = sh[i].C, K = sh[i].K;
        const int64_t Kdim0 = K * C;
        const int64_t Kdim = (Kdim0 + 31) / 32 * 32;      // q8_0 的块对齐
        const double useful = (double) T * C * C * K;      // 有用 MAC
        const double padded = (double) T * C * Kdim;       // 打满后的 MAC（含 padding 浪费）

        std::vector<float> A((size_t) Kdim * T, 0.0f), W((size_t) Kdim * C, 0.0f);
        fill_rand(A, 1234u + i, 0.5f);
        fill_rand(W, 5678u + i, 0.2f);
        std::vector<char> Aq(ggml_row_size(GGML_TYPE_Q8_0, Kdim) * T);
        std::vector<char> Wq(ggml_row_size(GGML_TYPE_Q8_0, Kdim) * C);

        double dt = 0;
        // 激活量化 pass 的成本（必须融合掉的那笔）
        double best_q = 1e30;
        for (int r = 0; r < reps; r++) {
            const double t0 = now_s();
            ggml_quantize_chunk(GGML_TYPE_Q8_0, A.data(), Aq.data(), 0, T, Kdim, nullptr);
            const double d = now_s() - t0;
            if (d < best_q) best_q = d;
        }
        ggml_quantize_chunk(GGML_TYPE_Q8_0, W.data(), Wq.data(), 0, C, Kdim, nullptr);
        // int8 走 GEMM 的端到端：im2col(int8) + q8_0 GEMM；激活量化按“已融合”=0 计
        std::vector<char> Xq((size_t) C * T);           // 已量化的激活 [IC][T]
        for (size_t z = 0; z < Xq.size(); z++) Xq[z] = (char) ((z * 31u) & 0x7f);
        std::vector<char> Aim(Aq.size());
        const double im2col_s = time_im2col(Xq, Aim, T, C, K, Kdim, reps);

        double bf = 0, bq = 0;
        run_gemm(be, GGML_TYPE_F32, GGML_TYPE_F32, Kdim, T, C, &A, &W, nullptr, nullptr, nullptr, &bf, reps);
        run_gemm(be, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, Kdim, T, C, nullptr, nullptr, &Aq, &Wq, nullptr, &bq, reps);

        // 现在的实际路径：fp32 直接卷积（同一形状）
        double bc = 1e30;
        {
            ggml_init_params ip = { 8u << 20, nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * tw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, C, C);
            ggml_tensor * tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, C);
            ggml_tensor * tb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
            ggml_tensor * y = ggml_conv_direct_1d_fused(ctx, tw, tx, tb, nullptr, (int) (K / 2), 1, 0.1f, 1.0f, 0.0f);
            ggml_set_output(y);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, y);
            ggml_gallocr_t all = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
            if (ggml_gallocr_alloc_graph(all, gf)) {
                std::vector<float> wf((size_t) K * C * C), xf((size_t) T * C), biasv((size_t) C, 0.0f);
                fill_rand(wf, 42u + i, 0.2f);
                fill_rand(xf, 43u + i, 0.5f);
                ggml_backend_tensor_set(tw, wf.data(), 0, wf.size() * 4);
                ggml_backend_tensor_set(tx, xf.data(), 0, xf.size() * 4);
                ggml_backend_tensor_set(tb, biasv.data(), 0, biasv.size() * 4);
                for (int r = 0; r < reps; r++) {
                    const double t0 = now_s();
                    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) break;
                    const double d = now_s() - t0;
                    if (d < bc) bc = d;
                }
            }
            ggml_gallocr_free(all);
            ggml_free(ctx);
        }

        char nm[64];
        snprintf(nm, sizeof nm, "%s C=%lld K=%lld", sh[i].name, (long long) C, (long long) K);
        const double padpct = 100.0 * (double) (Kdim - Kdim0) / (double) Kdim0;
        printf("%-23s %9.3f %6lld %5.1f%% | %8.2f %8.2f | %8.2f %8.2f | %7.2f %6.2f | %8.2f | %6.2fx %6.2fx\n",
               nm, useful * 1e-9, (long long) Kdim, padpct,
               padded / bf * 1e-9, padded / bq * 1e-9,
               padded / bf * 1e-9 / threads, padded / bq * 1e-9 / threads,
               best_q * 1e3, im2col_s * 1e3, useful / bc * 1e-9,
               (useful / bc) / (padded / bq),                        // GEMM 单独 vs conv
               (useful / bc) / (padded / (bq + im2col_s)));          // 端到端 vs conv
        t_im2col += im2col_s;
        t_fp32 += bf; t_q8 += bq; t_conv += bc; t_quant += best_q; m_useful += useful;
    }
    printf("\n4 线程总览（只统计本表形状）: \n");
    printf("  fp32 GEMM      %8.2f GMAC/s  (= %.1f GMAC/s/线程)  %6.2f%% CPU\n", m_useful / t_fp32 * 1e-9, m_useful / t_fp32 * 1e-9 / threads, cpu_pct());
    printf("  q8_0 GEMM      %8.2f GMAC/s  (= %.1f GMAC/s/线程)  比 fp32 快 %.2fx\n",
           m_useful / t_q8 * 1e-9, m_useful / t_q8 * 1e-9 / threads, t_fp32 / t_q8);
    printf("  q8_0 + 独立量化 pass: %8.2f GMAC/s（量化 %.1f ms，占 GEMM 的 %.1f%%）\n",
           m_useful / (t_q8 + t_quant) * 1e-9, t_quant * 1e3, 100.0 * t_quant / t_q8);
    printf("  fp32 直接卷积  %8.2f GMAC/s  (= %.1f GMAC/s/线程)  ← 现在的路径\n",
           m_useful / t_conv * 1e-9, m_useful / t_conv * 1e-9 / threads);
    printf("  int8 端到端（im2col %.1f ms + GEMM %.1f ms / 形状合计）= %.2fx 当前卷积\n",
           t_im2col * 1e3, t_q8 * 1e3, (m_useful / t_conv) / (m_useful / (t_q8 + t_im2col)));
    printf("  （若激活量化没融合，再叠加 %.1f ms：%.2fx）\n",
           t_quant * 1e3, (m_useful / t_conv) / (m_useful / (t_q8 + t_im2col + t_quant)));
    printf("  ⇒ GEMM 单独 %.2fx；融合量化+im2col 后 %.2fx 当前卷积\n",
           (m_useful / t_conv) / (m_useful / t_q8), (m_useful / t_conv) / (m_useful / (t_q8 + t_im2col)));
    ggml_backend_free(be);
    return 0;
}
