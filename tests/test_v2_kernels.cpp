// gsv_v2_dec_resblocks（AVX2 ResBlock1 内核）的正确性自检：
//   1) 与朴素参考实现对比（验证折叠、halo、pad、leaky、residual 语义）
//   2) 线程数 1/2/3/4/5/7 的输出必须**逐位相同**（每行只属于一个线程，
//      halo 行由各线程冗余重算，重算的累加顺序固定 ⇒ 与切分无关）
//   3) 非 16 倍数的时间长度（尾巴瓦片写进 slack）也必须正确
//
//   ./test_v2_kernels                  # 全量（含 1..7 线程不变性）
//   GSV_TEST_THREADS=1 ./test_v2_kernels   # 只跑单线程（共享机器上友好）
//
// 不做任何计时（共享机器上计时毫无意义）。
//
// 非 x86 开发机上也能跑 *AVX2 路径*：把 tests/avx2_emu 加进 include path，
// 并用 -D__AVX2__ 编译，垫片 (tests/avx2_emu/immintrin.h) 用 fmaf/lane 展开
// 逐位等价地仿真所需的内联函数。边界 zero_outside 的 bug 就是这么找到的：
//   clang++ -std=c++17 -O2 -D__AVX2__ -I tests/avx2_emu -I src \
//           tests/test_v2_kernels.cpp src/gsv_kernels.cpp -o /tmp/t && /tmp/t
#include "gsv_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

float frand(uint32_t & st) {          // xorshift32 → [-0.5, 0.5)
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    return (float) (st & 0xffffff) / (float) 0x1000000 - 0.5f;
}

inline float leaky(float v) { return v < 0 ? 0.1f * v : v; }

// 参考实现：一个 stage 的全部 ResBlock1（3 块 × n_pairs 对，逐 block 求和）
void reference(const gsv_rb_stage & st, const float * in, float * out) {
    const int C = st.C, T = st.T;
    std::vector<float> x((size_t) T * C), y((size_t) T * C);
    std::fill(out, out + (size_t) T * C, 0.0f);
    auto tap = [&](const std::vector<float> & b, int oc, int row) {
        return (row < 0 || row >= T) ? 0.0f : b[(size_t) row + (size_t) T * oc];
    };
    for (int blk = 0; blk < st.n_blocks; blk++) {
        std::copy(in, in + (size_t) T * C, x.begin());
        for (int p = 0; p < st.n_pairs; p++) {
            const gsv_rb_pair & pr = st.pairs[(size_t) blk * st.n_pairs + p];
            // conv1: y = b1 + leaky(Σ w1 · leaky(x))
            for (int o = 0; o < C; o++)
                for (int t = 0; t < T; t++) {
                    float acc = pr.b1[o];
                    for (int ic = 0; ic < C; ic++)
                        for (int m = 0; m < pr.K; m++)
                            acc += leaky(tap(x, ic, t + m * pr.dil - pr.pad1)) *
                                   pr.w1[((size_t) m * C + ic) * C + o];
                    y[(size_t) t + (size_t) T * o] = leaky(acc);
                }
            // conv2: x = b2 + Σ w2 · y + x
            for (int o = 0; o < C; o++)
                for (int t = 0; t < T; t++) {
                    float acc = pr.b2[o];
                    for (int ic = 0; ic < C; ic++)
                        for (int m = 0; m < pr.K; m++)
                            acc += tap(y, ic, t + m - pr.pad2) *
                                   pr.w2[((size_t) m * C + ic) * C + o];
                    x[(size_t) t + (size_t) T * o] = acc + x[(size_t) t + (size_t) T * o];
                }
        }
        for (size_t i = 0; i < (size_t) T * C; i++) out[i] += x[i];
    }
}

int fails = 0;
int g_force_threads = 0;   // >0：只跑这一个线程数（共享机器上避免 N 线程负载）

void check_case(int C, int T, int n_blocks, int n_pairs, const int * Ks, const int * dils,
                uint32_t seed, float tol, bool with_ref) {
    gsv_rb_stage st;
    st.T = T; st.C = C; st.n_blocks = n_blocks; st.n_pairs = n_pairs;

    // 权重/偏置
    std::vector<std::vector<float>> w1((size_t) n_blocks * n_pairs), w2((size_t) n_blocks * n_pairs);
    std::vector<std::vector<float>> b1((size_t) n_blocks * n_pairs), b2((size_t) n_blocks * n_pairs);
    std::vector<gsv_rb_pair> prs((size_t) n_blocks * n_pairs);
    uint32_t s = seed;
    for (int i = 0; i < n_blocks * n_pairs; i++) {
        const int K = Ks[i % n_pairs], dil = dils[i % n_pairs];
        const size_t n = (size_t) K * C * C;
        // 按 1/sqrt(K*C) 缩放，否则 3 对 × 3 块的残差链会把值放大到 1e6，
        // 那时 2e-4 的绝对容差就没意义了
        const float sc = 1.0f / std::sqrt((float) K * (float) C);
        w1[(size_t) i].resize(n); w2[(size_t) i].resize(n);
        b1[(size_t) i].resize((size_t) C); b2[(size_t) i].resize((size_t) C);
        for (size_t j = 0; j < n; j++) {
            w1[(size_t) i][j] = sc * frand(s);
            w2[(size_t) i][j] = sc * frand(s);
        }
        for (int j = 0; j < C; j++) {
            b1[(size_t) i][(size_t) j] = 0.1f * frand(s);
            b2[(size_t) i][(size_t) j] = 0.1f * frand(s);
        }
        prs[(size_t) i].w1 = w1[(size_t) i].data();
        prs[(size_t) i].w2 = w2[(size_t) i].data();
        prs[(size_t) i].b1 = b1[(size_t) i].data();
        prs[(size_t) i].b2 = b2[(size_t) i].data();
        prs[(size_t) i].K = K;
        prs[(size_t) i].dil = dil;
        prs[(size_t) i].pad1 = (K / 2) * dil;
        prs[(size_t) i].pad2 = K / 2;
    }
    st.pairs = prs.data();

    std::vector<float> in((size_t) T * C);
    for (size_t i = 0; i < in.size(); i++) in[i] = frand(s);

    std::vector<float> ref((size_t) T * C, 0.0f);
    std::vector<float> got((size_t) T * C, 0.0f);
    gsv_v2_dec_resblocks(st, in.data(), got.data(), 1);
    double maxd = -1.0, ref_absmax = 0.0;
    bool ok_ref = true;
    if (with_ref) {
        reference(st, in.data(), ref.data());
        maxd = 0.0;
        for (size_t i = 0; i < ref.size(); i++) {
            maxd = std::max(maxd, (double) std::fabs(ref[i] - got[i]));
            ref_absmax = std::max(ref_absmax, (double) std::fabs(ref[i]));
        }
        ok_ref = maxd <= tol * std::max(1.0, ref_absmax);
        if (!ok_ref) {
            fails++;
            if (getenv("GSV_TEST_DEBUG")) {
                printf("          debug (first 6):\n");
                for (int i = 0; i < 6 && i < (int) ref.size(); i++)
                    printf("            ref=%+.8f got=%+.8f diff=%+.3e\n", ref[(size_t) i], got[(size_t) i],
                           (double) (ref[(size_t) i] - got[(size_t) i]));
                for (int i = 0; i < n_blocks * n_pairs; i++)
                    printf("            pair%d K=%d dil=%d pad1=%d pad2=%d b1[0]=%+.6f b2[0]=%+.6f w1[0]=%+.6f w2[0]=%+.6f\n",
                           i, prs[(size_t) i].K, prs[(size_t) i].dil, prs[(size_t) i].pad1,
                           prs[(size_t) i].pad2, (double) prs[(size_t) i].b1[0], (double) prs[(size_t) i].b2[0],
                           (double) prs[(size_t) i].w1[0], (double) prs[(size_t) i].w2[0]);
                printf("            in[0..2]=%+.6f %+.6f %+.6f\n",
                       (double) in[0], in.size() > 1 ? (double) in[1] : 0.0, in.size() > 2 ? (double) in[2] : 0.0);
                fflush(stdout);
            }
        }
    }

    // 线程数不变性：逐位比较（GSV_TEST_THREADS=N 时只跑单线程，跳过此步）
    bool ok_thr = true;
    int bad_thr = 0;
    std::vector<int> tlist = { 2, 3, 4, 5, 7 };
    if (g_force_threads > 0) {
        tlist.clear();
        if (g_force_threads != 1) tlist.push_back(g_force_threads);
    }
    for (int nt : tlist) {
        std::vector<float> o2((size_t) T * C, 0.0f);
        gsv_v2_dec_resblocks(st, in.data(), o2.data(), nt);
        if (std::memcmp(got.data(), o2.data(), got.size() * 4) != 0) { ok_thr = false; bad_thr = nt; }
    }
    if (!ok_thr) fails++;

    printf("  C=%-3d T=%-6d blocks=%d pairs=%d K=[%d,%d,%d] dil=[%d,%d,%d]  "
           "ref %s  threads 2..7 bit-identical: %s%s\n",
           C, T, n_blocks, n_pairs, Ks[0], Ks[1], Ks[2], dils[0], dils[1], dils[2],
           with_ref ? (ok_ref ? "ok" : "FAIL") : "(skipped)",
           tlist.empty() ? "(skipped)" : (ok_thr ? "ok" : "FAIL"),
           ok_thr ? "" : (std::string(" (nt=") + std::to_string(bad_thr) + ")").c_str());
    if (with_ref)
        printf("          ref maxdiff=%.3e (tol %.1e, max|ref|=%.3f)\n", maxd, (double) tol,
               ref_absmax);
    fflush(stdout);
}

}  // namespace

int main() {
    if (const char * e = getenv("GSV_TEST_THREADS")) g_force_threads = std::max(1, atoi(e));
    if (g_force_threads) printf("threads forced to %d (invarianance check skipped)\n", g_force_threads);
    const int Ks[3] = { 3, 7, 11 };
    const int dils[3] = { 1, 3, 5 };
    // 大形状（scale 真实的 stage0/tail）只验线程数不变性；参考实现是标量的，
    // 大形状跑一遍要几十秒，不值得
    check_case(256, 330, 3, 3, Ks, dils, 0x1234567u, 2e-4f, false);
    check_case(16, 1318, 3, 3, Ks, dils, 0x89abcdefu, 2e-4f, false);
    // 小形状：对参考实现 + 线程数不变性（含非 16 倍数的尾巴瓦片、T=1 退化）
    check_case(16, 97, 2, 2, Ks, dils, 0xdeadbeefu, 2e-4f, true);
    check_case(32, 33, 1, 3, Ks, dils, 0x5eed1234u, 2e-4f, true);
    check_case(8, 1, 1, 1, Ks, dils, 0x11112222u, 2e-4f, true);
    check_case(16, 271, 3, 3, Ks, dils, 0x2468aceu, 2e-4f, true);
    printf(fails ? "\nFAILED (%d)\n" : "\nOK\n", fails);    return fails ? 1 : 0;
}
