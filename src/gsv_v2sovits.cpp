// GPT-SoVITS v2 / v2Pro / v2ProPlus S2 decoder in ggml (VITS: enc_p → flow → dec).
//
// ============================================================================
// 1) 为什么不是配置开关 (v5 没有这些结构)
// ============================================================================
//   v5:  phones → cond(enc_p + bridge + wns1) → fea[512,frames] → DiT/CFM → mel → vocoder
//   v2:  sem codes → quantizer.vq.decode → ×2 nearest → enc_p → (m_p, logs_p)
//                   z_p = m_p + ε·exp(logs_p)   (ε ~ N(0,1), noise_scale)
//                   z   = flow(z_p, g=ge, reverse=True)      ← 4×ResidualCouplingLayer
//                   wav = dec(z, g=ge)                        ← HiFi-GAN (5 段上采样)
//   v2 没有 DiT、没有 cond 段、没有 RVQ+bridge；dec 直接出波形。码本只有一层 (n_q=1)。
//   flow 的耦合层是 mean_only=True（checkpoint 里没有 log_scale 参数），与 v5 的 cond WN 无关。
//
// 全链的 torch 出处（GPT-SoVITS-CPUFast/GPT_SoVITS/module/models.py）:
//   SynthesizerTrn.prepare_decode_latent / .decode              — 解码主链
//   TextEncoder.forward                                        — enc_p（本文件复用 gsv_encp）
//   ResidualCouplingBlock.forward(reverse=True)                — flow，注意含 Flip 层
//   Generator.forward                                          — dec
//   TextEncoder.extract_latent (= ssl_proj k2s2 + quantizer)   — 参考侧取码（codes_from_ssl）
//
// ============================================================================
// 2) enc_p 为什么复用 gsv_encp
// ============================================================================
//   v2 的 TextEncoder 与 v5 **逐层同构**：hidden 192 / filter 768 / heads 2 / dk 96 /
//   encoder_ssl×3 / encoder_text×6 / encoder2×3 / MRTE(cross-attn 4×128, c_pre/text_pre/
//   c_post) / proj 192→384。GGUF 里的前缀同样是 enc_p.*。所以这里直接复用 gsv_encp
//   （它已经与 torch 对拍通过），只把 v2 特有的 ssl_proj+VQ 与 flow/dec 写在本文件。
//
// ============================================================================
// 3) weight_norm 折叠
// ============================================================================
//   dec/flow 的 conv 是 torch.nn.utils.weight_norm(conv, name="weight")，GGUF 里存的是
//   weight_v / weight_g（导出器没折叠、也没重排）。weight_norm(dim=0) 的语义：
//       W = v * (g / ||v||₂),  ||v||₂ 沿**除 dim0 外的所有轴**
//     Conv1d          v:[OC,IC,K], g:[OC,1,1]  → 每个输出通道归一
//     ConvTranspose1d v:[IC,OC,K], g:[IC,1,1]  → 每个输入通道归一（dim0 = in！）
//   折叠后统一成 ggml 的 [K,IC,OC]（ggml_conv_direct_1d 的核布局），类型 F32
//   （该算子 CPU 实现要求 F32 核，见 patches/0001-ggml-audio-patch-*）。
//
// ============================================================================
// 4) ConvTranspose1d → 子像素卷积（相位分组核）
// ============================================================================
//   torch: y[t·s + r, o] = Σ_{i,k} x[i, c]·W[c, o, k]，其中 n = t·s + r = i·s - pad + k
//   ⇒ k = (t-i)·s + (r + pad)，记 d = t-i、p = r + pad ⇒ k = d·s + p。
//   所以每个相位 r 是 x 的一个**普通相关**（k 轴被 s 抽样），有效 tap 集合是
//       d ∈ [lo_r, hi_r] = [ceil(-p/s), floor((K-1-p)/s)]
//   对 s=10/K=16/pad=3 这种 K > s 的层，d 会跨负值（d=-1 = 用下一帧）。
//
//   **相位分组**：dhi-dlo+1 = M 就是这组真正要算的 tap 数。相邻相位的 [lo,hi] 往往
//   完全相同，于是把区间相同的相邻相位并成一组，每组一次 conv、M 个 tap 全部命中。
//   早期的实现取所有相位的**并集**区间，会白算 (并集 M) / (K/s) 倍：stage0 1.875x、
//   stage1 1.5x、stage2 1.25x、stage3/4 2x（实测 ups 3.98 → 2.69 GMAC）。
//   副作用同样重要：单组 Wt 只有 ~1/5 大（stage0 15.7 MB → 每组 2-3 MB），不再在每次调用
//   里把整块权重从 DRAM 刷一遍（那正是 stage0 只跑到 17% 峰值的原因）。
//
//   ggml_conv_direct_1d_fused: out[j, oc] = Σ_m w[m, ic, oc]·x[j + m - P]，OL = T + 2P - M + 1
//   令 j = t、P = hi_r ⇒ x[t + m - hi_r] = x[t - d] ⇒ d = hi_r - m，OL = T + lo_r + hi_r。
//   OL ≥ T 时取前 T 行即答案；OL = T 时无需裁剪。
//   dlo < 0（相位用到下一帧）时，用 ggml_pad 在输入末尾补 dmax = max(-dlo) 行零，
//   这样不必再为“让 pad 抵消 dlo”而多算零 tap（torch 的 conv_transpose 在末尾同样视为 0）。
//   组内输出通道 = (r - ph0)·OC + o，按相位升序 concat 就得到 [T, OC·s]，
//   再 reshape/permute/cont 成 [T·s, OC]（与 gsv_vocoder_ggml 相同的一次转置）。
//   bias 按相位展开后直接融进 conv 的 epilogue（省掉 5 个 add 节点）。
//
// ============================================================================
// 5) 图切分
// ============================================================================
//   (a) vq   : ssl[T,768] → ssl_proj(k2s2) → [Tc,768] → mul_mat(codebook) → 相似度 [K,Tc]
//              （argmax 在 host 侧做；|x|² 项对 argmax 无影响，省掉）
//   (b) enc_p: gsv_encp（独立对象，同一 GGUF 的 enc_p.* 前缀）
//   (c) flow : zp[T,192] + ge[512] → z（T 变化时重建图并缓存）
//   (d) dec  : z[T,192] + ge[512] → wav[T·640]
//   中间张量布局一律 "t 主序" [T, C]（ggml_conv_direct_1d 的要求），与 gsv_encp 的
//   torch [C,T] 布局在边界处转置（只在小张量上做，成本可忽略）。
//
// ============================================================================
// 6) 性能剖析（GSV_V2_PROFILE=1）
// ============================================================================
//   把 flow/dec 的每个节点单独 compute 一遍（ggml_graph_view 不对外，改用
//   ggml_graph_clear + ggml_graph_add_node 复用一张单节点 cgraph），按实际张量形状算
//   MAC、按 exec 算 GFLOPS，并用 ggml_set_name 在构图时给 conv 节点起名。
//   实测：逐节点时间合计与整图时间差 < 1%（节点唤醒开销 ~20 µs/节点）。
//   GSV_V2_PEAK 可覆盖 roofline 峰值（默认 319 GFLOPS = tests/bwpeak.c 的 4 线程实测）。
//   结论见 docs/ 或任务报告：dec 是 645k MAC/输出采样（20.6 GMAC/s 音频），
//   fp32 在 4 核 Zen2 上的地板就是 0.162 RTF，别再往 0.10 上追；剩下的差距只能靠 int8。
//
// ============================================================================
// 7) dec 的两条执行路径（GSV_V2_DEC=kernels|graph）
// ============================================================================
//   graph   : 整个 dec 交给 ggml 图（ggml_conv_direct_1d_fused），与历史行为一致。
//   kernels : conv_pre/cond/ups_*/conv_post 仍然走 ggml， 而 resblock 的 45 对 conv
//             改由 src/gsv_kernels.cpp 的 AVX2 内核算（fused bias+leaky+res，t 为 lane，
//             输出直接落在 [T, C]）。执行方式是：按 build_dec 记录的节点范围把图
//             分成 6 段（conv_pre+ups0, ups1..ups4, conv_post），每段之间插一次内核调用
//             —— 也就是每个 stage 只有 1 次 fork-join，stage 内 RESBlock 链无屏障。
//             resblock 的 ggml 节点仍然建在图中（保证 buffer 分配不变），只是不执行。
#include "gsv_v2sovits.h"
#include "gsv_kernels.h"

#include "gsv_device.h"
#include "gsv_encp.h"
#include "gsv_sampler.h"   // gsv_rng (z_p 的噪声)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ---------------------------------------------------------------- 小工具
static void v2_to_f32(const void * src, float * dst, int64_t n, ggml_type t) {
    if (t == GGML_TYPE_F32) { std::memcpy(dst, src, (size_t) n * 4); return; }
    if (t == GGML_TYPE_F16) { ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, dst, n); return; }
    if (t == GGML_TYPE_BF16) { ggml_bf16_to_fp32_row((const ggml_bf16_t *) src, dst, n); return; }
    const ggml_type_traits * tr = ggml_get_type_traits(t);
    tr->to_float(src, dst, n);
}

static int64_t v2_floor_div(int64_t a, int64_t b) {   // b > 0
    return a >= 0 ? a / b : -((-a + b - 1) / b);
}
static int64_t v2_ceil_div(int64_t a, int64_t b) {    // b > 0
    return a >= 0 ? (a + b - 1) / b : -((-a) / b);
}

// 具名 conv（名字供 GSV_V2_PROFILE 的逐节点表使用，同时改善 GGML debug print）
static ggml_tensor * v2_conv(ggml_context * c, ggml_tensor * w, ggml_tensor * x,
                             ggml_tensor * bias, ggml_tensor * res,
                             int pad, int dil, float slope, float in_scale, float in_slope,
                             const char * name) {
    if (w->ne[1] != x->ne[1]) {
        fprintf(stderr, "[gsv_v2] conv %s: w ne=[%ld,%ld,%ld] x ne=[%ld,%ld]\n",
                name, (long) w->ne[0], (long) w->ne[1], (long) w->ne[2],
                (long) x->ne[0], (long) x->ne[1]);
        abort();
    }
    ggml_tensor * t = ggml_conv_direct_1d_fused(c, w, x, bias, res, pad, dil, slope, in_scale, in_slope);
    ggml_set_name(t, name);
    return t;
}

// 一个待折叠/拷贝的张量
struct v2_item {
    std::string dst;    // 折叠后的逻辑名（weight 用 .weight，bias 用 .bias）
    int kind = 0;       // 0 = 已是 [K,IC,OC] 的普通核; 1 = weight_norm conv; 2 = weight_norm TransposeConv 的相位组
                        // 3 = 直接拷贝 (bias); 4 = 码本 [K, D]; 5 = 相位展开后的 bias
    std::string w, g;   // 源张量名
    int K = 0, IC = 0, OC = 0, st = 0, M = 0, P = 0;
    int ph0 = 0, ph1 = 0;   // kind 2/5：本 item 负责的相位区间 [ph0, ph1)
    int dlo = 0, dhi = 0;   // kind 2：本组每个相位都命中的 tap 区间（d = 帧偏移）
};

struct v2_graph {
    ggml_context *  ctx = nullptr;
    ggml_cgraph *   g   = nullptr;
    ggml_gallocr_t  ga  = nullptr;
    int T = 0;
    int nframes = 0;        // vq 图: 输出帧数 (与 torch 的 floor((T-K)/s)+1 一致)
    ggml_tensor * in  = nullptr;
    ggml_tensor * ge  = nullptr;
    ggml_tensor * out = nullptr;
    ggml_tensor * aux = nullptr;   // 诊断用（vq: ssl_proj 的输出）
    void release() {
        if (ga) ggml_gallocr_free(ga);
        if (ctx) ggml_free(ctx);
        ga = nullptr; ctx = nullptr; g = nullptr;
        in = ge = out = aux = nullptr; T = 0; nframes = 0;
    }
};

// ---------------------------------------------------------------- 逐节点剖析
// GSV_V2_PROFILE=1 时把 flow/dec 图按节点逐个 compute（ggml_graph_view）并计时。
// 单独 compute 一个节点多花一次线程池唤醒（~20 µs），所以看合计与整图时间的差
// 就能判断开销占比；MAC 数一律从实际张量形状算（不猜）。
struct v2_prof_row {
    std::string   name;
    enum ggml_op  op = GGML_OP_NONE;
    int64_t K = 0, IC = 0, OC = 0, OL = 0;
    double macs = 0, macs_raw = 0, bytes = 0, ms = 0;
};

static const char * v2_op_str(enum ggml_op op) {
    switch (op) {
        case GGML_OP_CONV_DIRECT_1D: return "conv1d";
        case GGML_OP_MUL_MAT:        return "mul_mat";
        case GGML_OP_ADD:            return "add";
        case GGML_OP_SUB:            return "sub";
        case GGML_OP_MUL:            return "mul";
        case GGML_OP_SCALE:          return "scale";
        case GGML_OP_LEAKY_RELU:     return "leaky";
        case GGML_OP_CONT:           return "cont";
        case GGML_OP_RESHAPE:        return "reshape";
        case GGML_OP_PERMUTE:        return "permute";
        case GGML_OP_VIEW:           return "view";
        case GGML_OP_CONCAT:         return "concat";
        default:                     return ggml_op_name(op);
    }
}

// MAC 与最小搬运字节；macs_raw = 含 16 通道补齐的实际执行量
static void v2_node_cost(ggml_tensor * n, int64_t & K, int64_t & IC, int64_t & OC, int64_t & OL,
                         double & macs, double & macs_raw, double & bytes) {
    K = IC = OC = OL = 0;
    macs = macs_raw = 0.0;
    bytes = (double) ggml_nbytes(n);
    for (int i = 0; i < GGML_MAX_SRC; i++)
        if (n->src[i]) bytes += (double) ggml_nbytes(n->src[i]);

    switch (n->op) {
        case GGML_OP_CONV_DIRECT_1D: {
            const ggml_tensor * w = n->src[0];
            const ggml_tensor * x = n->src[1];
            K  = w->ne[0]; IC = w->ne[1]; OC = w->ne[2];
            OL = x->ne[0];
            const int64_t OCe = (OC + 15) / 16 * 16;
            macs     = (double) OL * OC  * IC * K;
            macs_raw = (double) OL * OCe * IC * K;
            break;
        }
        case GGML_OP_MUL_MAT: {
            const ggml_tensor * a = n->src[0];
            K  = a->ne[0]; IC = a->ne[1];
            OL = n->ne[0]; OC = n->ne[1] * n->ne[2] * n->ne[3];
            macs = macs_raw = (double) ggml_nelements(n) * (double) a->ne[0];
            break;
        }
        default:
            macs = macs_raw = (double) ggml_nelements(n);
            break;
    }
}

static void v2_profile_graph(ggml_backend_t backend, v2_graph & gr, const char * tag) {
    if (!gr.g || ggml_graph_n_nodes(gr.g) <= 0) return;
    const int nn = ggml_graph_n_nodes(gr.g);

    // 单节点图（复用同一个 cgraph 对象，只换 nodes[0]）——不走
    // ggml_build_forward_expand，否则会递归把所有祖先重算一遍。
    ggml_init_params pip = { ggml_tensor_overhead() * 16, nullptr, true };
    ggml_context * pctx = ggml_init(pip);
    ggml_cgraph * one = ggml_new_graph_custom(pctx, 16, false);

    std::vector<v2_prof_row> rows;
    rows.reserve((size_t) nn);
    for (int i = 0; i < nn; i++) {
        ggml_tensor * n = ggml_graph_node(gr.g, i);
        v2_prof_row r;
        r.name = n->name[0] ? n->name : ggml_op_name(n->op);
        r.op = n->op;
        v2_node_cost(n, r.K, r.IC, r.OC, r.OL, r.macs, r.macs_raw, r.bytes);
        ggml_graph_clear(one);
        ggml_graph_add_node(one, n);
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, one);
        r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        rows.push_back(std::move(r));
    }
    ggml_free(pctx);

    double tot_ms = 0, tot_macs = 0, tot_raw = 0, tot_bytes = 0;
    for (const auto & r : rows) { tot_ms += r.ms; tot_macs += r.macs; tot_raw += r.macs_raw; tot_bytes += r.bytes; }
    const char * pk = getenv("GSV_V2_PEAK");
    const double peak = pk ? atof(pk) : 319.0;   // 4 线程 AVX2-FMA roofline (tests/bwpeak.c)

    std::vector<size_t> ord(rows.size());
    for (size_t i = 0; i < ord.size(); i++) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return rows[a].ms > rows[b].ms; });

    printf("\n==== [%s] per-node profile: %zu nodes, sum=%.1f ms, MAC=%.1f M (raw %.1f M, pad %.1f%%), "
           "min-traffic=%.1f MB -> %.1f GFLOPS = %.0f%% of %.0f GFLOPS peak\n",
           tag, rows.size(), tot_ms, tot_macs / 1e6, tot_raw / 1e6,
           100.0 * (tot_raw - tot_macs) / (tot_raw > 0 ? tot_raw : 1),
           tot_bytes / 1048576.0, 2.0 * tot_macs / (tot_ms * 1e6),
           100.0 * (2.0 * tot_macs / (tot_ms * 1e6)) / peak, peak);
    printf("  %-18s %-8s %6s %6s %6s %7s %12s %8s %8s %5s %8s %6s\n",
           "node", "op", "K", "IC", "OC", "OL", "MAC", "ms", "GFLOPS", "%pk", "MB", "GB/s");
    for (size_t oi = 0; oi < ord.size(); oi++) {
        const v2_prof_row & r = rows[ord[oi]];
        if (r.ms < 0.02 && r.macs < 1e5) continue;      // 噪声行不打印
        const double gf = r.ms > 0 ? 2.0 * r.macs / (r.ms * 1e6) : 0.0;
        printf("  %-18s %-8s %6ld %6ld %6ld %7ld %12.2f %8.2f %8.1f %5.1f %8.3f %6.1f\n",
               r.name.c_str(), v2_op_str(r.op), (long) r.K, (long) r.IC, (long) r.OC, (long) r.OL,
               r.macs / 1e6, r.ms, gf, 100.0 * gf / peak, r.bytes / 1048576.0,
               r.ms > 0 ? r.bytes / 1048576.0 / (r.ms / 1e3) : 0.0);
    }
    fflush(stdout);
}

struct gsv_v2sovits::impl {
    ggml_backend_t backend = nullptr;
    gguf_context * gf = nullptr;
    ggml_context * wctx = nullptr;            // gguf 元数据（no_alloc，只用来查形状/偏移）
    FILE * fp = nullptr;
    size_t data_off = 0;

    ggml_context * fctx = nullptr;            // 折叠后的 F32 权重
    ggml_backend_buffer_t fbuf = nullptr;
    std::map<std::string, ggml_tensor *> w;
    std::vector<v2_item> items;

    gsv_encp * encp = nullptr;

    // 码本（host 侧 + 设备侧各一份；host 供 decode_codes，设备供 vq 图）
    std::vector<float> cb;                    // [cb_size * ssl_dim] (k*D + d)
    std::vector<float> cb_sq;                 // |E_k|²

    // ---- 结构参数（由 GGUF KV + 权重形状推出并校验）----
    std::string ver = "v2";
    int sr = 32000;
    int gin = 512;
    int hidden = 192;                         // inter_channels
    int ssl_dim = 768;
    int cb_size = 1024;
    int n_stage = 5;
    int n_res = 3;
    int n_flow = 4;
    int flow_hid = 192;
    int flow_k = 5;
    int n_wntaps = 4;
    int ssl_k = 2, ssl_stride = 2, ssl_pad = 0;
    std::vector<int> ups, ups_k;
    std::vector<int> res_k, res_dil;
    int up_total = 1;

    int n_threads = 0;
    bool verbose = false;
    double ms_encp = 0, ms_flow = 0, ms_dec = 0;

    // 子像素上采样的相位分组（每个 stage 一组，组内按相位升序）
    struct v2_upg { std::string wname, bname; int ph0 = 0, ph1 = 0, dlo = 0, dhi = 0; };
    std::vector<std::vector<v2_upg>> up_groups;

    // ---- GSV_V2_DEC=kernels 手写 ResBlock1 内核（src/gsv_kernels.*）----
    bool use_kernels = false;               // 默认值在 load() 里按 env 决定
    std::vector<float> rb_wpool;            // 重排后的权重池（[K][IC][OC]，oc 最快）
    std::vector<gsv_rb_pair> rb_pairs;      // [n_stage][n_res][3]
    std::vector<gsv_rb_stage> rb_stages;    // [n_stage]
    // kernels 模式执行范围：[0]=conv_pre+cond+ups0, [1..n_stage-1]=ups_i, [n_stage]=conv_post
    struct v2_krange { int s = 0, e = 0; };
    std::vector<v2_krange> kparts;
    std::vector<ggml_tensor *> kup;         // 每个 stage 的 ups 输出（内核的 in）
    std::vector<ggml_tensor *> ksum;        // 每个 stage 的 resblock 输出（内核的 out）
    ggml_context * sctx = nullptr;          // 复用一张子 cgraph 按范围执行
    ggml_cgraph * sub = nullptr;
    void run_range(int s, int e);
    void prepare_kernels();

    v2_graph gvq, gflow, gdec;

    ~impl() {
        gvq.release(); gflow.release(); gdec.release();
        if (fp) fclose(fp);
        delete encp;
        if (fbuf) ggml_backend_buffer_free(fbuf);
        if (fctx) ggml_free(fctx);
        if (wctx) ggml_free(wctx);
        if (gf) gguf_free(gf);
        if (backend) ggml_backend_free(backend);
    }

    ggml_tensor * need(const char * n) const {
        auto it = w.find(n);
        if (it == w.end()) { fprintf(stderr, "[gsv_v2] missing tensor %s\n", n); abort(); }
        return it->second;
    }
    bool has(const char * n) const { return ggml_get_tensor(wctx, n) != nullptr; }
    ggml_tensor * src(const char * n) const { return ggml_get_tensor(wctx, n); }

    std::vector<float> read_f32(const char * name);

    // ---- 折叠 ----
    void plan_weights();
    void create_folded();
    void fill_folded();

    // ---- 图 ----
    void build_vq(int T);
    void build_flow(int T);
    void build_dec(int T);

    ggml_tensor * bias_of(const std::string & wname) const {
        const std::string b = wname.substr(0, wname.size() - 7) + ".bias";
        auto it = w.find(b);
        return it == w.end() ? nullptr : it->second;
    }
};

// ------------------------------------------------------------------ 读张量 (host f32)
std::vector<float> gsv_v2sovits::impl::read_f32(const char * name) {
    const int64_t ti = gguf_find_tensor(gf, name);
    if (ti < 0) { fprintf(stderr, "[gsv_v2] no tensor %s\n", name); return {}; }
    ggml_tensor * t = ggml_get_tensor(wctx, name);
    if (!t) return {};
    const int64_t n = ggml_nelements(t);
    const size_t nbytes = ggml_nbytes(t);
    std::vector<char> raw(nbytes);
    if (fseek(fp, (long) (data_off + gguf_get_tensor_offset(gf, ti)), SEEK_SET) != 0 ||
        fread(raw.data(), 1, nbytes, fp) != nbytes) {
        fprintf(stderr, "[gsv_v2] read %s failed\n", name);
        return {};
    }
    std::vector<float> out((size_t) n);
    v2_to_f32(raw.data(), out.data(), n, t->type);
    return out;
}

// ------------------------------------------------------------------ 权重规划
void gsv_v2sovits::impl::plan_weights() {
    char n1[192], n2[192];

    // GGUF 里张量的 ggml ne（gguf.GGUFWriter 按 shape[::-1] 落盘 ⇒ ne = reversed(numpy shape)）:
    //   普通 Conv1d        torch [OC,IC,K] → ne = [K, IC, OC]
    //   Conv1d.weight_v          [OC,IC,K] → ne = [K, IC, OC]
    //   ConvTranspose1d.weight_v [IC,OC,K] → ne = [K, OC, IC]
    //   Linear                   [out,in]  → ne = [in, out]
    //   码本                     [K,D]     → ne = [D, K]
    //   weight_g / bias          [n,1,1]/[n] → 元素数就是 n
    auto add_plain = [&](const std::string & name) {          // ne = [K, IC, OC]
        ggml_tensor * t = src(name.c_str());
        if (!t) return;
        v2_item it; it.dst = name; it.kind = 0; it.w = name;
        it.K = (int) t->ne[0]; it.IC = (int) t->ne[1]; it.OC = (int) t->ne[2];
        items.push_back(it);
    };
    auto add_bias = [&](const std::string & name) {
        ggml_tensor * t = src(name.c_str());
        if (!t) return;
        v2_item it; it.dst = name; it.kind = 3; it.w = name;
        it.K = (int) ggml_nelements(t);
        items.push_back(it);
    };
    // weight_norm conv: 源 v:[OC,IC,K] g:[OC,1,1]
    auto add_wn = [&](const std::string & base) {
        const std::string v = base + ".weight_v", g = base + ".weight_g";
        ggml_tensor * tv = src(v.c_str());
        if (!tv) { fprintf(stderr, "[gsv_v2] missing %s\n", v.c_str()); abort(); }
        v2_item it; it.dst = base + ".weight"; it.kind = 1; it.w = v; it.g = g;
        it.K = (int) tv->ne[0]; it.IC = (int) tv->ne[1]; it.OC = (int) tv->ne[2];
        items.push_back(it);
        add_bias(base + ".bias");
    };
    // weight_norm ConvTranspose1d: 源 v:[IC,OC,K] g:[IC,1,1]
    // → 子像素卷积按「相位分组」折叠：相位 r 的 tap 集合是 d ∈ [lo_r, hi_r]
    //   （k = d*st + r + pad_tr 必须落在 [0,K)），相邻相位往往共享同一个区间；
    //   把区间相同的相邻相位并成一组，每组一次 conv，K=M=dhi-dlo+1 个 tap **全部命中**。
    //   旧的「所有相位取并集区间」白算了 K/st / M 的倍数：stage0 1.875x、stage1 1.5x、
    //   stage2 1.25x、stage3/4 2x。组内输出通道 = (r-ph0)*OC + o，所以按相位顺序
    //   concat 就得到 [T, st*OC]。bias 也用 phase 展开（kind 5）直接融进 conv 的 epilogue。
    auto add_wnt = [&](const std::string & base, int st) {
        const std::string v = base + ".weight_v", g = base + ".weight_g";
        ggml_tensor * tv = src(v.c_str());
        if (!tv) { fprintf(stderr, "[gsv_v2] missing %s\n", v.c_str()); abort(); }
        const int K = (int) tv->ne[0], OC = (int) tv->ne[1], IC = (int) tv->ne[2];
        const int pad_tr = (K - st) / 2;
        std::vector<int> lo((size_t) st), hi((size_t) st);
        for (int r = 0; r < st; r++) {
            const int64_t p = r + pad_tr;
            lo[(size_t) r] = (int) v2_ceil_div(-p, st);
            hi[(size_t) r] = (int) v2_floor_div(K - 1 - p, st);
            if (lo[(size_t) r] > hi[(size_t) r]) {
                fprintf(stderr, "[gsv_v2] %s 相位 %d 无有效 tap\n", base.c_str(), r); abort();
            }
        }
        std::vector<v2_upg> gs;
        int gi = 0;
        for (int r = 0; r < st; ) {
            int r2 = r + 1;
            while (r2 < st && lo[(size_t) r2] == lo[(size_t) r] && hi[(size_t) r2] == hi[(size_t) r]) r2++;
            char nm[224];
            v2_item it;
            snprintf(nm, sizeof(nm), "%s.weight.g%d", base.c_str(), gi);
            it.dst = nm; it.kind = 2; it.w = v; it.g = g; it.st = st;
            it.K = K; it.IC = IC; it.OC = OC;
            it.dlo = lo[(size_t) r]; it.dhi = hi[(size_t) r];
            it.M = it.dhi - it.dlo + 1; it.P = it.dhi;
            it.ph0 = r; it.ph1 = r2;
            items.push_back(it);
            v2_item ib;
            snprintf(nm, sizeof(nm), "%s.bias.g%d", base.c_str(), gi);
            ib.dst = nm; ib.kind = 5; ib.w = base + ".bias";
            ib.OC = OC; ib.ph0 = r; ib.ph1 = r2;
            items.push_back(ib);
            v2_upg u;
            u.wname = it.dst; u.bname = ib.dst;
            u.ph0 = r; u.ph1 = r2; u.dlo = it.dlo; u.dhi = it.dhi;
            gs.push_back(u);
            r = r2; gi++;
        }
        up_groups.push_back(gs);
    };

    // ---- dec (HiFi-GAN) ----
    add_plain("dec.conv_pre.weight");
    add_bias("dec.conv_pre.bias");
    for (int i = 0; i < n_stage; i++) {
        snprintf(n1, sizeof(n1), "dec.ups.%d", i);
        add_wnt(n1, ups[(size_t) i]);
    }
    for (int blk = 0; blk < n_stage * n_res; blk++) {
        for (int c = 1; c <= 2; c++) {
            for (int k = 0; k < 3; k++) {
                snprintf(n1, sizeof(n1), "dec.resblocks.%d.convs%d.%d", blk, c, k);
                add_wn(n1);
            }
        }
    }
    add_plain("dec.cond.weight");          // gin -> 512 的 1x1，把 ge 变成一个常量向量
    add_bias("dec.cond.bias");
    add_plain("dec.conv_post.weight");
    add_bias("dec.conv_post.bias");      // v2 的 conv_post 无 bias（is_bias=False），不存在则跳过

    // ---- flow (ResidualCouplingBlock) ----
    for (int li = 0; li < n_flow; li++) {
        snprintf(n1, sizeof(n1), "flow.flows.%d", li * 2);   // nn.ModuleList 里 Flip 占奇数位
        add_plain(std::string(n1) + ".pre.weight");
        add_bias(std::string(n1) + ".pre.bias");
        add_plain(std::string(n1) + ".post.weight");
        add_bias(std::string(n1) + ".post.bias");
        for (int i = 0; i < n_wntaps; i++) {
            snprintf(n2, sizeof(n2), "%s.enc.in_layers.%d", n1, i);
            add_wn(n2);
            snprintf(n2, sizeof(n2), "%s.enc.res_skip_layers.%d", n1, i);
            add_wn(n2);
        }
        snprintf(n2, sizeof(n2), "%s.enc.cond_layer", n1);
        add_wn(n2);
    }

    // ---- ssl_proj (参考侧 Conv1d(768,768,2,stride=2)，导出器已重排为 [K,IC,OC]) ----
    add_plain("ssl_proj.weight");
    add_bias("ssl_proj.bias");

    // ---- 码本（保持 [K, D] 布局，另有 host 副本）----
    {
        const std::string nm = "quantizer.vq.layers.0._codebook.embed";
        ggml_tensor * t = src(nm.c_str());
        if (!t) { fprintf(stderr, "[gsv_v2] missing %s\n", nm.c_str()); abort(); }
        v2_item it; it.dst = nm; it.kind = 4; it.w = nm;
        it.IC = (int) t->ne[0];                // D = ssl_dim  (torch [K,D] → ne = [D,K])
        it.OC = (int) t->ne[1];                // K = 码本大小
        items.push_back(it);
    }
}

void gsv_v2sovits::impl::create_folded() {
    ggml_init_params ip = { ggml_tensor_overhead() * (items.size() + 64), nullptr, /*no_alloc*/ true };
    fctx = ggml_init(ip);
    if (!fctx) { fprintf(stderr, "[gsv_v2] fold ctx init failed\n"); abort(); }
    for (const v2_item & it : items) {
        ggml_tensor * t = nullptr;
        switch (it.kind) {
            case 0: t = ggml_new_tensor_3d(fctx, GGML_TYPE_F32, it.K, it.IC, it.OC); break;
            case 1: t = ggml_new_tensor_3d(fctx, GGML_TYPE_F32, it.K, it.IC, it.OC); break;
            case 2: t = ggml_new_tensor_3d(fctx, GGML_TYPE_F32, it.M, it.IC,
                                           (int64_t) it.OC * (it.ph1 - it.ph0)); break;
            case 3: t = ggml_new_tensor_1d(fctx, GGML_TYPE_F32, it.K); break;
            case 4: t = ggml_new_tensor_2d(fctx, GGML_TYPE_F32, it.IC, it.OC); break;   // [D,K]
            case 5: t = ggml_new_tensor_1d(fctx, GGML_TYPE_F32, (int64_t) it.OC * (it.ph1 - it.ph0)); break;
            default: abort();
        }
        ggml_set_name(t, it.dst.c_str());
        w[it.dst] = t;
    }
    {   // torch.flip(x,[1]) 用 1x1 置换卷积实现：W[0, ic, oc] = δ(ic+oc, C-1)
        ggml_tensor * t = ggml_new_tensor_3d(fctx, GGML_TYPE_F32, 1, hidden, hidden);
        ggml_set_name(t, "flow.flip.weight");
        w["flow.flip.weight"] = t;
    }
    fbuf = ggml_backend_alloc_ctx_tensors(fctx, backend);
    if (!fbuf) { fprintf(stderr, "[gsv_v2] fold buffer alloc failed\n"); abort(); }
}

void gsv_v2sovits::impl::fill_folded() {
    std::vector<float> v, g;
    {
        ggml_tensor * t = w["flow.flip.weight"];
        std::vector<float> pv((size_t) hidden * hidden, 0.0f);
        for (int ic = 0; ic < hidden; ic++) pv[(size_t) (hidden - 1 - ic) + (size_t) ic * hidden] = 1.0f;
        ggml_backend_tensor_set(t, pv.data(), 0, pv.size() * 4);
    }
    for (const v2_item & it : items) {
        ggml_tensor * dst = w[it.dst];
        if (it.kind == 3) {
            v = read_f32(it.w.c_str());
            if (v.empty()) abort();
            ggml_backend_tensor_set(dst, v.data(), 0, v.size() * 4);
        } else if (it.kind == 0) {
            // 源和目标都是 [K, IC, OC]，numpy 行主 = flat(k + ic*K + oc*K*IC) → 直接拷
            v = read_f32(it.w.c_str());
            if (v.empty()) abort();
            ggml_backend_tensor_set(dst, v.data(), 0, v.size() * 4);
        } else if (it.kind == 4) {
            v = read_f32(it.w.c_str());          // [K, D] (flat k*D + d)
            if (v.empty()) abort();
            ggml_backend_tensor_set(dst, v.data(), 0, v.size() * 4);
            cb = v;                              // host 副本（decode_codes 用）
            const int K = it.OC, D = it.IC;   // flat = d + k*D
            cb_sq.assign((size_t) K, 0.0f);
            for (int k = 0; k < K; k++) {
                double acc = 0.0;
                for (int d = 0; d < D; d++) { const double x = v[(size_t) k * D + d]; acc += x * x; }
                cb_sq[(size_t) k] = (float) acc;
            }
        } else if (it.kind == 1) {
            // v 的 ggml ne 就是 [K, IC, OC] → flat = k + ic*K + oc*K*IC（与 torch [OC,IC,K]
            // 行主完全相同）; g 的 ne 是 [1,1,OC]
            v = read_f32(it.w.c_str());
            g = read_f32(it.g.c_str());
            if (v.empty() || g.size() != (size_t) it.OC) {
                fprintf(stderr, "[gsv_v2] fold %s failed\n", it.dst.c_str()); abort();
            }
            std::vector<float> norm((size_t) it.OC, 0.0f);
            for (int o = 0; o < it.OC; o++) {
                double acc = 0.0;
                for (int ic = 0; ic < it.IC; ic++)
                    for (int k = 0; k < it.K; k++) {
                        // flat(k + ic*K + oc*K*IC) == torch [OC,IC,K] 的行主序
                        const double x = v[(size_t) k + (size_t) ic * it.K + (size_t) o * it.K * it.IC];
                        acc += x * x;
                    }
                norm[(size_t) o] = (float) std::sqrt(acc);
            }
            std::vector<float> out((size_t) it.K * it.IC * it.OC, 0.0f);
            for (int o = 0; o < it.OC; o++) {
                const float scale = g[(size_t) o] / (norm[(size_t) o] > 1e-12f ? norm[(size_t) o] : 1e-12f);
                for (int ic = 0; ic < it.IC; ic++)
                    for (int k = 0; k < it.K; k++) {
                        const float x = v[(size_t) k + (size_t) ic * it.K + (size_t) o * it.K * it.IC];
                        // dst [K,IC,OC]: flat k + ic*K + o*K*IC（与源同一顺序）
                        out[(size_t) k + (size_t) ic * it.K + (size_t) o * it.K * it.IC] = x * scale;
                    }
            }
            ggml_backend_tensor_set(dst, out.data(), 0, out.size() * 4);
        } else if (it.kind == 2) {
            // v 的 ggml ne 是 [K, OC, IC] → flat = k + oc*K + ic*OC*K（== torch [IC,OC,K] 行主）;
            // g 的 ne 是 [1,1,IC]（dim0 = in，逐输入通道归一）
            v = read_f32(it.w.c_str());
            g = read_f32(it.g.c_str());
            if (v.empty() || g.size() != (size_t) it.IC) {
                fprintf(stderr, "[gsv_v2] fold %s failed\n", it.dst.c_str()); abort();
            }
            std::vector<float> norm((size_t) it.IC, 0.0f);
            for (int ic = 0; ic < it.IC; ic++) {
                double acc = 0.0;
                for (int o = 0; o < it.OC; o++)
                    for (int k = 0; k < it.K; k++) {
                        // flat(k + oc*K + ic*OC*K) == torch [IC,OC,K] 的行主序
                        const double x = v[(size_t) k + (size_t) o * it.K + (size_t) ic * it.OC * it.K];
                        acc += x * x;
                    }
                norm[(size_t) ic] = (float) std::sqrt(acc);
            }
            const int st = it.st, M = it.M, IC = it.IC, OC = it.OC, K = it.K;
            const int pad_tr = (K - st) / 2, np = it.ph1 - it.ph0;
            std::vector<float> out((size_t) M * IC * OC * np, 0.0f);
            for (int ic = 0; ic < IC; ic++) {
                const float scale = g[(size_t) ic] / (norm[(size_t) ic] > 1e-12f ? norm[(size_t) ic] : 1e-12f);
                for (int r = it.ph0; r < it.ph1; r++) {
                    for (int m = 0; m < M; m++) {
                        const int d = it.P - m;
                        const int k = d * st + r + pad_tr;
                        if (k < 0 || k >= K) continue;
                        for (int o = 0; o < OC; o++) {
                            const float x = v[(size_t) k + (size_t) o * K + (size_t) ic * OC * K];
                            // dst [M, IC, np*OC]: flat m + ic*M + ((r-ph0)*OC + o)*M*IC
                            out[(size_t) m + (size_t) ic * M + ((size_t) (r - it.ph0) * OC + o) * M * IC] = x * scale;
                        }
                    }
                }
            }
            ggml_backend_tensor_set(dst, out.data(), 0, out.size() * 4);
        } else if (it.kind == 5) {
            // 相位展开后的 bias：同一 OC 组每个相位一份（torch 的 conv_transpose bias 对每个
            // 输出采样都加一次，所以每个相位相同）
            const int np = it.ph1 - it.ph0, OC = it.OC;
            std::vector<float> b((size_t) np * OC, 0.0f);
            if (src(it.w.c_str())) {
                v = read_f32(it.w.c_str());
                if ((int) v.size() < OC) {
                    fprintf(stderr, "[gsv_v2] bias %s 长度 %zu < %d\n", it.w.c_str(), v.size(), OC);
                    abort();
                }
                for (int r = 0; r < np; r++)
                    for (int o = 0; o < OC; o++) b[(size_t) r * OC + o] = v[(size_t) o];
            }
            ggml_backend_tensor_set(dst, b.data(), 0, b.size() * 4);
        }
    }
}

// ------------------------------------------------------------------ flow 图
// torch.flip(x, [1])：整条通道轴反序。用 1x1 置换卷积实现（out[t,oc] = x[t, C-1-oc]）
static ggml_tensor * v2_flip_channels(ggml_context * c, ggml_tensor * x,
                                      ggml_tensor * perm, int T, int C) {
    (void) T; (void) C;
    return v2_conv(c, perm, x, nullptr, nullptr, 0, 1, 0.0f, 1.0f, 0.0f, "flip");
}

void gsv_v2sovits::impl::build_flow(int T) {
    gflow.release();
    ggml_init_params ip = { ggml_tensor_overhead() * 8192, nullptr, true };
    gflow.ctx = ggml_init(ip);
    ggml_context * c = gflow.ctx;
    gflow.T = T;
    const int C = hidden, HC = hidden / 2;

    ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, C);
    ggml_set_input(x);
    gflow.in = x;
    ggml_tensor * ge = ggml_new_tensor_1d(c, GGML_TYPE_F32, gin);
    ggml_set_input(ge);
    gflow.ge = ge;
    ggml_tensor * ge2 = ggml_reshape_2d(c, ge, gin, 1);

    char wn[192];
    for (int li = n_flow - 1; li >= 0; li--) {
        const std::string base = "flow.flows." + std::to_string(li * 2) + ".enc";
        x = v2_flip_channels(c, x, need("flow.flip.weight"), T, C);   // Flip 层（reverse 序）
        // ---- ResidualCouplingLayer (mean_only) ----
        ggml_tensor * x0 = ggml_cont(c, ggml_view_2d(c, x, T, HC, x->nb[1], 0));
        ggml_tensor * x1 = ggml_cont(c, ggml_view_2d(c, x, T, HC, x->nb[1], (size_t) HC * x->nb[1]));
        snprintf(wn, sizeof(wn), "flow.flows.%d.pre.weight", li * 2);
        char fn[64];
        snprintf(fn, sizeof(fn), "flow%d.pre", li);
        ggml_tensor * h = v2_conv(c, need(wn), x0, bias_of(wn), nullptr,
                                  0, 1, 0.0f, 1.0f, 0.0f, fn);                        // [T,192]
        // ---- WN ----
        ggml_tensor * gc = ggml_mul_mat(c,
                ggml_reshape_2d(c, need((base + ".cond_layer.weight").c_str()), gin, 2 * flow_hid * n_wntaps),
                ge2);                                                              // [1536,1]
        gc = ggml_add(c, gc, ggml_reshape_2d(c, need((base + ".cond_layer.bias").c_str()),
                                             2 * flow_hid * n_wntaps, 1));
        ggml_tensor * acc = nullptr;
        ggml_tensor * hx = h;
        for (int i = 0; i < n_wntaps; i++) {
            const std::string inb = base + ".in_layers." + std::to_string(i);
            const std::string rsb = base + ".res_skip_layers." + std::to_string(i);
            snprintf(fn, sizeof(fn), "flow%d.in%d", li, i);
            ggml_tensor * xin = v2_conv(c, need((inb + ".weight").c_str()), hx,
                                        need((inb + ".bias").c_str()), nullptr,
                                        flow_k / 2, 1, 0.0f, 1.0f, 0.0f, fn);            // [T,384]
            ggml_tensor * gl = ggml_reshape_2d(c, ggml_view_1d(c, gc, 2 * flow_hid,
                                                               (size_t) i * 2 * flow_hid * 4), 1, 2 * flow_hid);
            xin = ggml_add(c, xin, gl);
            ggml_tensor * a = ggml_cont(c, ggml_view_2d(c, xin, T, flow_hid, xin->nb[1], 0));
            ggml_tensor * b = ggml_cont(c, ggml_view_2d(c, xin, T, flow_hid, xin->nb[1], (size_t) flow_hid * xin->nb[1]));
            ggml_tensor * acts = ggml_mul(c, ggml_tanh(c, a), ggml_sigmoid(c, b));
            snprintf(fn, sizeof(fn), "flow%d.rs%d", li, i);
            ggml_tensor * rs = v2_conv(c, need((rsb + ".weight").c_str()), acts,
                                       need((rsb + ".bias").c_str()), nullptr,
                                       0, 1, 0.0f, 1.0f, 0.0f, fn);
            if (i < n_wntaps - 1) {
                ggml_tensor * res = ggml_cont(c, ggml_view_2d(c, rs, T, flow_hid, rs->nb[1], 0));
                ggml_tensor * skp = ggml_cont(c, ggml_view_2d(c, rs, T, flow_hid, rs->nb[1], (size_t) flow_hid * rs->nb[1]));
                hx = ggml_add(c, hx, res);
                acc = acc ? ggml_add(c, acc, skp) : skp;
            } else {
                acc = acc ? ggml_add(c, acc, rs) : rs;
            }
        }
        h = acc;
        snprintf(wn, sizeof(wn), "flow.flows.%d.post.weight", li * 2);
        snprintf(fn, sizeof(fn), "flow%d.post", li);
        ggml_tensor * m = v2_conv(c, need(wn), h, bias_of(wn), nullptr,
                                  0, 1, 0.0f, 1.0f, 0.0f, fn);                        // [T,96]
        x1 = ggml_sub(c, x1, m);                                                   // reverse, logs=0
        x = ggml_concat(c, x0, x1, 1);       // 通道轴拼接 → [T, 2*HC]
    }
    ggml_set_output(x);
    gflow.out = x;
    gflow.g = ggml_new_graph_custom(c, 16384, false);
    ggml_build_forward_expand(gflow.g, x);
    gflow.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(gflow.ga, gflow.g)) {
        fprintf(stderr, "[gsv_v2] flow graph alloc failed (T=%d)\n", T);
        abort();
    }
    if (getenv("GSV_V2_DEBUG")) ggml_graph_print(gflow.g);
}

// ------------------------------------------------------------------ dec 图
void gsv_v2sovits::impl::build_dec(int T) {
    gdec.release();
    ggml_init_params ip = { ggml_tensor_overhead() * 32768, nullptr, true };
    gdec.ctx = ggml_init(ip);
    ggml_context * c = gdec.ctx;
    gdec.T = T;

    ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, hidden);
    ggml_set_input(x);
    gdec.in = x;
    ggml_tensor * ge = ggml_new_tensor_1d(c, GGML_TYPE_F32, gin);
    ggml_set_input(ge);
    gdec.ge = ge;

    // conv_pre (k=7 pad=3) + LeakyReLU(0.1) 融合在 epilogue
    // 参考: conv_pre → +cond → leaky_relu → ups[0]，所以 leaky 不能放在 conv_pre 的 epilogue
    ggml_tensor * h = v2_conv(c, need("dec.conv_pre.weight"), x,
                              bias_of("dec.conv_pre.weight"), nullptr, 3, 1, 0.0f, 1.0f, 0.0f,
                              "conv_pre");
    // + cond(ge)（1x1 conv；与时间无关 → 只算一个 [1,512] 向量广播）
    {
        ggml_tensor * gc = ggml_mul_mat(c, ggml_reshape_2d(c, need("dec.cond.weight"), gin, 512),
                                        ggml_reshape_2d(c, ge, gin, 1));               // [512,1]
        gc = ggml_add(c, gc, ggml_reshape_2d(c, need("dec.cond.bias"), 512, 1));
        h = ggml_add(c, h, ggml_reshape_2d(c, gc, 1, 512));
    }
    const float mean_scale = 1.0f / (float) n_res;
    char wn[192];
    kup.clear(); ksum.clear();
    for (int i = 0; i < n_stage; i++) {
        const int st = ups[(size_t) i];
        // ---- 子像素上采样：ConvTranspose1d(K, stride) == 相位分组 conv + 相位交织 ----
        // 每个相位组的 K=M=dhi-dlo+1 个 tap 全部命中（旧实现取并集区间，白算 1.25~2x）。
        // 组内输出通道 = (r-ph0)*cout + o，按相位顺序 concat → [Tin, st*cout]。
        const std::vector<v2_upg> & gs = up_groups[(size_t) i];
        const int cout = (int) (need(gs[0].wname.c_str())->ne[2] / (gs[0].ph1 - gs[0].ph0));
        int dmax = 0;
        for (const v2_upg & g : gs) dmax = std::max(dmax, -g.dlo);   // dlo<0 时需要往右补零帧
        const float in_scale = (i > 0) ? mean_scale : 1.0f;
        const float in_slope = 0.1f;      // 每级 ups 前都有 F.leaky_relu(x, 0.1)
        const int64_t Tin = h->ne[0];
        ggml_tensor * xin = h;
        if (dmax > 0)
            xin = ggml_pad(c, h, dmax, 0, 0, 0);        // 尾部补 dmax 行零（dim0）
        char nm[64];
        ggml_tensor * ycat = nullptr;
        for (size_t gi = 0; gi < gs.size(); gi++) {
            snprintf(nm, sizeof(nm), "ups.%d.g%zu", i, gi);
            ggml_tensor * y = v2_conv(c, need(gs[gi].wname.c_str()), xin,
                                      need(gs[gi].bname.c_str()), nullptr,
                                      gs[gi].dhi, 1, 0.0f, in_scale, in_slope, nm);
            const int64_t OL = y->ne[0];
            if (OL < Tin) {
                fprintf(stderr, "[gsv_v2] up%d g%zu: OL %ld < T %ld (dmax=%d)\n",
                        i, gi, (long) OL, (long) Tin, dmax);
                abort();
            }
            if (OL != Tin) y = ggml_cont(c, ggml_view_2d(c, y, Tin, y->ne[1], y->nb[1], 0));
            ycat = ycat ? ggml_concat(c, ycat, y, 1) : y;
        }
        ggml_tensor * y3 = ggml_reshape_3d(c, ycat, Tin, cout, st);
        ggml_tensor * yp = ggml_permute(c, y3, 1, 2, 0, 3);
        ggml_tensor * yc = ggml_cont(c, yp);
        ggml_tensor * up = ggml_reshape_2d(c, yc, (int64_t) Tin * st, cout);              // [T·st, cout]
        h = up;
        kup.push_back(up);      // kernels 模式：本 stage 交给 ggml 的节点到此为止
            // ---- 3 × ResBlock1 求和（第 j>0 块的求和融进第 3 对 conv 的 epilogue）----
        ggml_tensor * sum = nullptr;
        for (int j = 0; j < n_res; j++) {
            const int blk = i * n_res + j;
            ggml_tensor * hx = h;
            for (int k = 0; k < 3; k++) {
                const int dil = res_dil[(size_t) k];
                snprintf(wn, sizeof(wn), "dec.resblocks.%d.convs1.%d.weight", blk, k);
                ggml_tensor * w1 = need(wn);
                const int pad1 = ((int) w1->ne[0] / 2) * dil;
                snprintf(nm, sizeof(nm), "rb%d.c1.%d", blk, k);
                ggml_tensor * yy = v2_conv(c, w1, hx, bias_of(wn), nullptr,
                                           pad1, dil, 0.1f, 1.0f, 0.1f, nm);
                snprintf(wn, sizeof(wn), "dec.resblocks.%d.convs2.%d.weight", blk, k);
                ggml_tensor * w2 = need(wn);
                const int pad2 = (int) w2->ne[0] / 2;
                snprintf(nm, sizeof(nm), "rb%d.c2.%d", blk, k);
                // k==2 是块内最后一对 conv：把「前面块的部分和」当 res 交给 epilogue
                // ((bias+acc)+res) 与 res+hx 位级相同（两数相加可交换）
                hx = v2_conv(c, w2, yy, bias_of(wn), hx,
                             pad2, 1, 0.0f, 1.0f, 0.0f, nm);
            }
            sum = sum ? ggml_add(c, sum, hx) : hx;
        }
        h = sum;   // 均值 1/3 与 LeakyReLU 融进下一级 conv 的 in_scale/in_slope
        ksum.push_back(sum);
        }
    // conv_post (k=7 pad=3, 无 bias) + tanh；leaky_slope < 0 = tanh
    ggml_tensor * out = v2_conv(c, need("dec.conv_post.weight"), h,
                                bias_of("dec.conv_post.weight"), nullptr,
                                3, 1, -1.0f, mean_scale, 0.1f, "conv_post");
    ggml_set_output(out);
    gdec.out = out;
    gdec.g = ggml_new_graph_custom(c, 65536, false);
    ggml_build_forward_expand(gdec.g, out);
    // kernels 模式的执行边界（节点下标只有 build 之后才知道）：
    //   part0 = conv_pre+cond+ups0, part_i = ups_i, part_n = conv_post
    {
        const int nn = ggml_graph_n_nodes(gdec.g);
        auto producer = [&](ggml_tensor * t) {
            for (int i = 0; i < nn; i++) if (ggml_graph_node(gdec.g, i) == t) return i;
            return -1;
        };
        kparts.clear();
        int prev = -1;
        for (int i = 0; i < n_stage; i++) {
            const int pu = producer(kup[(size_t) i]);
            const int ps = producer(ksum[(size_t) i]);
            if (pu < 0 || ps < 0 || ps < pu) {
                fprintf(stderr, "[gsv_v2] dec 图边界定位失败 (stage %d: pu=%d ps=%d)\n", i, pu, ps);
                abort();
            }
            kparts.push_back({ prev + 1, pu + 1 });
            prev = ps;
        }
        kparts.push_back({ prev + 1, nn });
    }
    gdec.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(gdec.ga, gdec.g)) {
        fprintf(stderr, "[gsv_v2] dec graph alloc failed (T=%d)\n", T);
        abort();
    }
    if (getenv("GSV_V2_DEBUG")) ggml_graph_print(gdec.g);
}

// ------------------------------------------------------------------ vq 图 (ssl_proj + 码本)
void gsv_v2sovits::impl::build_vq(int T) {
    gvq.release();
    ggml_init_params ip = { ggml_tensor_overhead() * 4096, nullptr, true };
    gvq.ctx = ggml_init(ip);
    ggml_context * c = gvq.ctx;
    gvq.T = T;
    const int D = ssl_dim, Tc = T / ssl_stride;

    ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, D);
    ggml_set_input(x);
    gvq.in = x;

    // ssl_proj: Conv1d(D, D, k=ssl_k, stride=ssl_stride)
    ggml_tensor * w = need("ssl_proj.weight");      // 导出器已重排成 [K, IC, OC]
    if ((int) w->ne[0] != ssl_k || (int) w->ne[1] != D || (int) w->ne[2] != D) {
        fprintf(stderr, "[gsv_v2] ssl_proj ne=[%d,%d,%d] 期望 [%d,%d,%d]\n",
                (int) w->ne[0], (int) w->ne[1], (int) w->ne[2], ssl_k, D, D);
        abort();
    }
    ggml_tensor * im = ggml_im2col_fast_1d(c, w, x, ssl_stride, ssl_pad, 1, GGML_TYPE_F32, 1);
    ggml_tensor * mm = ggml_mul_mat(c,
            ggml_reshape_2d(c, im, im->ne[0], im->ne[1] * im->ne[2]),
            ggml_reshape_2d(c, w, (int64_t) w->ne[0] * w->ne[1], w->ne[2]));
    ggml_tensor * y = ggml_reshape_3d(c, mm, im->ne[1], D, im->ne[2]);      // [Tc, D, 1]
    y = ggml_add(c, y, ggml_reshape_3d(c, need("ssl_proj.bias"), 1, D, 1));
    if ((int) y->ne[0] != Tc) {
        // 奇数帧时 torch 也走 floor((T-K)/s)+1，以图的实际输出为准
        if (verbose)
            printf("[gsv_v2] ssl_proj: T=%d -> %ld 帧 (T/s=%d)\n", T, (long) y->ne[0], Tc);
    }
    gvq.nframes = (int) y->ne[0];
    // [Tc, D] → [D, Tc] 稠密 → 与码本相乘（|x|² 对 argmax 无影响，故只算 ⟨x,E⟩ 与 |E|²）
    ggml_tensor * yt = ggml_cont(c, ggml_permute(c, y, 1, 0, 2, 3));
    ggml_tensor * cbw = need("quantizer.vq.layers.0._codebook.embed");      // [K, D]
    ggml_tensor * dot = ggml_mul_mat(c, cbw, yt);                           // [K, Tc]
    ggml_set_output(dot);
    gvq.aux = y;   // ssl_proj 输出 [Tc, D, 1]（GSV_V2_DEBUG 时落盘对拍）
    gvq.out = dot;
    gvq.g = ggml_new_graph_custom(c, 2048, false);
    ggml_build_forward_expand(gvq.g, dot);
    gvq.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(gvq.ga, gvq.g)) {
        fprintf(stderr, "[gsv_v2] vq graph alloc failed (T=%d)\n", T);
        abort();
    }
}

// ------------------------------------------------------------------ 加载
gsv_v2sovits::gsv_v2sovits() : p(new impl) {}
gsv_v2sovits::~gsv_v2sovits() { delete p; }

gsv_v2sovits * gsv_v2sovits::load(const std::string & gguf_path, const gsv_v2sovits_cfg & cfg) {
    gsv_v2sovits * m = new gsv_v2sovits();
    impl & s = *m->p;
    s.n_threads = cfg.n_threads;
    s.verbose = cfg.verbose;

    int gpu_idx = 0; bool gpu_explicit = false;
    const bool want_gpu = gsv_device_parse(cfg.device, &gpu_idx, &gpu_explicit);
    ggml_backend_dev_t dev = nullptr;
    {
        const int n_dev = (int) ggml_backend_dev_count();
        int gpu_seen = 0;
        for (int i = 0; i < n_dev && !dev; i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            const auto ty = ggml_backend_dev_type(d);
            if (want_gpu) {
                if ((ty == GGML_BACKEND_DEVICE_TYPE_GPU || ty == GGML_BACKEND_DEVICE_TYPE_IGPU) &&
                    gpu_seen++ == gpu_idx) dev = d;
            } else if (ty == GGML_BACKEND_DEVICE_TYPE_CPU) dev = d;
        }
        if (!dev)
            for (int i = 0; i < n_dev && !dev; i++) {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) dev = d;
            }
        if (want_gpu && gpu_explicit && (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU)) {
            fprintf(stderr, "[device] GPU %d not found (run --list-devices)\n", gpu_idx);
            delete m; return nullptr;
        }
    }
    if (!dev) { fprintf(stderr, "[gsv_v2] no usable backend device\n"); delete m; return nullptr; }
    const bool is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    if (cfg.verbose) printf("[gsv_v2] device: %s\n", ggml_backend_dev_name(dev));
    s.backend = ggml_backend_dev_init(dev, nullptr);
    if (!s.backend) { fprintf(stderr, "[gsv_v2] backend init failed\n"); delete m; return nullptr; }
    if (is_cpu && cfg.n_threads > 0) ggml_backend_cpu_set_n_threads(s.backend, cfg.n_threads);

    gguf_init_params gip = { /*no_alloc*/ true, &s.wctx };
    s.gf = gguf_init_from_file(gguf_path.c_str(), gip);
    if (!s.gf) { fprintf(stderr, "[gsv_v2] failed to open %s\n", gguf_path.c_str()); delete m; return nullptr; }
    s.fp = fopen(gguf_path.c_str(), "rb");
    if (!s.fp) { fprintf(stderr, "[gsv_v2] reopen failed\n"); delete m; return nullptr; }
    s.data_off = gguf_get_data_offset(s.gf);

    // ---- 元数据 ----
    auto kv_u32 = [&](const char * k, int def) {
        const int64_t id = gguf_find_key(s.gf, k);
        return id < 0 ? def : (int) gguf_get_val_u32(s.gf, id);
    };
    auto kv_str = [&](const char * k, const char * def) {
        const int64_t id = gguf_find_key(s.gf, k);
        return id < 0 ? std::string(def) : std::string(gguf_get_val_str(s.gf, id));
    };
    s.ver = kv_str("v2sovits.version", "v2");
    s.sr = kv_u32("v2sovits.sample_rate", 32000);
    s.gin = kv_u32("v2sovits.gin_channels", 512);
    s.hidden = kv_u32("v2sovits.hidden", 192);
    s.n_flow = kv_u32("v2sovits.n_layers_flow", 4);

    // 家族表：只有 ConvTranspose1d 的步长不能从权重形状唯一反推（K=16 既可能 s=10 也可能 s=8），
    // 其余全部由权重形状校验。步长表来自 GPT_SoVITS/configs/s2*.json。
    if (s.ver == "v2proplus") {
        s.ups = { 10, 8, 2, 2, 2 };
        s.ups_k = { 20, 16, 8, 2, 2 };
    } else {
        s.ups = { 10, 8, 2, 2, 2 };
        s.ups_k = { 16, 16, 8, 2, 2 };
    }
    s.n_stage = (int) s.ups.size();
    s.res_k = { 3, 7, 11 };
    s.res_dil = { 1, 3, 5 };
    s.up_total = 1;
    for (int u : s.ups) s.up_total *= u;

    if (!s.has("dec.conv_pre.weight") || !s.has("dec.conv_post.weight") || !s.has("flow.flows.0.pre.weight")) {
        fprintf(stderr, "[gsv_v2] %s 不像 v2 家族 SoVITS (缺 dec/flow 张量)\n", gguf_path.c_str());
        delete m; return nullptr;
    }
    {
        int want_in = 512;
        for (int u : s.ups) { (void) u; want_in /= 2; }
        ggml_tensor * cp = s.src("dec.conv_post.weight");
        if ((int) cp->ne[1] != want_in)
            fprintf(stderr, "[gsv_v2] 警告: conv_post 输入 %d != 512/2^%d=%d\n",
                    (int) cp->ne[1], s.n_stage, want_in);
        ggml_tensor * cd = s.src("dec.cond.weight");
        if (cd && (int) cd->ne[1] != s.gin) {
            fprintf(stderr, "[gsv_v2] 警告: dec.cond 输入 %d != KV gin %d（以权重为准）\n",
                    (int) cd->ne[1], s.gin);
            s.gin = (int) cd->ne[1];
        }
    }
    for (int i = 0; i < s.n_stage; i++) {
        char nm[64]; snprintf(nm, sizeof(nm), "dec.ups.%d.weight_v", i);
        ggml_tensor * t = s.src(nm);
        if (!t) { fprintf(stderr, "[gsv_v2] missing %s\n", nm); delete m; return nullptr; }
        const int K = (int) t->ne[0];
        if (K != s.ups_k[(size_t) i]) {
            fprintf(stderr, "[gsv_v2] %s: K=%d 与版本 %s 的表 (%d) 不符 —— 导出时用 --version 指定正确家族\n",
                    nm, K, s.ver.c_str(), s.ups_k[(size_t) i]);
            delete m; return nullptr;
        }
        const int cin = (int) t->ne[2];
        const int want_cin = 512 >> i;
        if (cin != want_cin) {
            fprintf(stderr, "[gsv_v2] %s: 输入通道 %d != %d（upsample_initial_channel 不同的家族）\n",
                    nm, cin, want_cin);
            delete m; return nullptr;
        }
    }
    // v2Pro/v2ProPlus 的 sv_emb 需要外部声纹模型（20480 维）；没有 sv_emb 时按 repo 的
    // version != v2pro 分支走（只用 ref_enc 的 ge）。
    if (s.has("sv_emb.weight"))
        fprintf(stderr, "[gsv_v2] 注意: 该文件含 sv_emb（v2Pro 声纹条件），本实现只用 ref_enc 的 ge\n");

    // 码本大小 / ssl 维度从权重读
    if (ggml_tensor * e = s.src("quantizer.vq.layers.0._codebook.embed")) {
        // torch [K, D] → ggml ne = [D, K]
        s.ssl_dim = (int) e->ne[0];
        s.cb_size = (int) e->ne[1];
    }

    s.plan_weights();
    s.create_folded();
    s.fill_folded();

    // dec 的 ResBlock1 主循环：GSV_V2_DEC=kernels|graph 选路。
    // 默认仍走 ggml graph：kernels 路径还没有干净的墙钟数据（共享机器上测不出来），
    // 不想在没算过的情况下就悄悄换掉默认输出路径。等 lead 的串行化 A/B 下来再改这一行。
    // kernels 模式需要一次性把 resblock 权重重排成 oc 最快布局（~44 MB）
    {
        const char * dm = getenv("GSV_V2_DEC");
        s.use_kernels = (dm && strcmp(dm, "kernels") == 0) && is_cpu;
        if (s.use_kernels) s.prepare_kernels();
        if (cfg.verbose)
            printf("[gsv_v2] dec 执行路径: %s (GSV_V2_DEC=%s)\n",
                   s.use_kernels ? "kernels (手写 AVX2 ResBlock1)" : "ggml graph", dm ? dm : "<unset>");
    }

    // enc_p（同一 GGUF 的 enc_p.* 前缀；v2 与 v5 的 TextEncoder 同构）
    {
        gsv_encp_cfg ec;
        ec.device = cfg.device;
        ec.n_threads = cfg.n_threads;
        ec.verbose = cfg.verbose;
        const std::string f = cfg.encp_file.empty() ? gguf_path : cfg.encp_file;
        s.encp = gsv_encp::load(f, ec);
        if (!s.encp) { fprintf(stderr, "[gsv_v2] enc_p 加载失败 (%s)\n", f.c_str()); delete m; return nullptr; }
    }

    if (cfg.verbose)
        printf("[gsv_v2] loaded %s: version=%s sr=%d gin=%d hidden=%d ups=%d/%d/%d/%d/%d up_total=%d "
               "codebook=%d ssl_dim=%d tensors=%zu\n",
               gguf_path.c_str(), s.ver.c_str(), s.sr, s.gin, s.hidden,
               s.ups[0], s.ups[1], s.ups[2], s.ups[3], s.ups[4], s.up_total,
               s.cb_size, s.ssl_dim, s.w.size());
    return m;
}

// ------------------------------------------------------------------ 查询
const std::string & gsv_v2sovits::version() const { return p->ver; }
int gsv_v2sovits::sample_rate() const { return p->sr; }
int gsv_v2sovits::upsample() const { return p->up_total; }
int gsv_v2sovits::codebook_size() const { return p->cb_size; }
int gsv_v2sovits::hidden() const { return p->hidden; }
int gsv_v2sovits::gin_channels() const { return p->gin; }
int gsv_v2sovits::frame_rate() const { return p->sr / p->up_total; }
void gsv_v2sovits::last_stage_ms(double & encp_ms, double & flow_ms, double & dec_ms) const {
    encp_ms = p->ms_encp; flow_ms = p->ms_flow; dec_ms = p->ms_dec;
}

// ------------------------------------------------------------------ 参考侧取码
bool gsv_v2sovits::codes_from_ssl(const float * ssl, int T, std::vector<int32_t> & codes) {
    impl & s = *p;
    codes.clear();
    if (T < s.ssl_stride) {
        fprintf(stderr, "[gsv_v2] ssl 帧数 %d 太少 (需要 >= %d)\n", T, s.ssl_stride);
        return false;
    }
    if (T != s.gvq.T) s.build_vq(T);
    // ggml [T, D] 的 flat 索引是 t + T*d，与 torch [D, T] 的 d*T+t 完全相同 → 直接搬
    std::vector<float> xs(ssl, ssl + (size_t) T * s.ssl_dim);
    ggml_backend_tensor_set(s.gvq.in, xs.data(), 0, xs.size() * 4);
    if (ggml_backend_graph_compute(s.backend, s.gvq.g) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[gsv_v2] vq graph compute failed\n");
        return false;
    }
    const int Tc = s.gvq.nframes;
    const int K = s.cb_size;
    std::vector<float> dot((size_t) K * Tc);
    ggml_backend_tensor_get(s.gvq.out, dot.data(), 0, dot.size() * 4);
    codes.resize((size_t) Tc);
    for (int t = 0; t < Tc; t++) {
        // torch: dist = -(x² - 2xE + E²), argmax → 2xE - E² 最大（x² 与码选择无关）
        int best = 0; float bestv = -1e30f;
        for (int k = 0; k < K; k++) {
            const float v = 2.0f * dot[(size_t) k + (size_t) t * K] - s.cb_sq[(size_t) k];
            if (v > bestv) { bestv = v; best = k; }
        }
        codes[(size_t) t] = best;
    }
    return true;
}

// ------------------------------------------------------------------ 码本 + ×2
void gsv_v2sovits::decode_codes(const int32_t * codes, int Tc, std::vector<float> & q768) const {
    const impl & s = *p;
    const int D = s.ssl_dim, T = 2 * Tc;
    q768.assign((size_t) D * T, 0.0f);
    for (int t = 0; t < T; t++) {
        const int32_t c = codes[t >> 1];                       // F.interpolate(mode="nearest") ×2
        if (c < 0 || c >= s.cb_size) continue;
        const float * e = s.cb.data() + (size_t) c * D;
        for (int d = 0; d < D; d++) q768[(size_t) d * T + t] = e[d];   // torch [C,T] 行主
    }
}

// ------------------------------------------------------------------ enc_p
bool gsv_v2sovits::encode_p(const int32_t * codes, int Tc, const int32_t * phones, int n_phones,
                            const float * ge, std::vector<float> & m_, std::vector<float> & logs_, int & T,
                            double speed) {
    impl & s = *p;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> q;
    decode_codes(codes, Tc, q);
    const int Tq = 2 * Tc;
    std::vector<float> mo, lo;
    if (!s.encp->encode(q.data(), Tq, phones, n_phones, ge, mo, lo, nullptr, (float) speed)) return false;
    T = (int) (mo.size() / (size_t) s.hidden);
    m_ = std::move(mo);
    logs_ = std::move(lo);
    s.ms_encp = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

// ------------------------------------------------------------------ flow
bool gsv_v2sovits::flow_reverse(const float * zp, int T, const float * ge, std::vector<float> & z) {
    impl & s = *p;
    const auto t0 = std::chrono::steady_clock::now();
    if (T != s.gflow.T) s.build_flow(T);
    // torch [C,T] ↔ ggml [T,C] 在内存里是同一份数据（flat = c*T + t = t + T*c）
    std::vector<float> x(zp, zp + (size_t) T * s.hidden);
    ggml_backend_tensor_set(s.gflow.in, x.data(), 0, x.size() * 4);
    ggml_backend_tensor_set(s.gflow.ge, ge, 0, (size_t) s.gin * 4);
    if (ggml_backend_graph_compute(s.backend, s.gflow.g) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[gsv_v2] flow graph compute failed\n");
        return false;
    }
    std::vector<float> y(x.size());
    ggml_backend_tensor_get(s.gflow.out, y.data(), 0, y.size() * 4);
    z.assign(y.begin(), y.end());       // 同一份内存布局，直接拷
    s.ms_flow = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    // 剖析放在取完输出之后：逐节点重算会撞上 gallocr 的 buffer 复用
    if (getenv("GSV_V2_PROFILE")) v2_profile_graph(s.backend, s.gflow, "flow");
    return true;
}

// ------------------------------------------------------------------ dec
// 把一个节点范围当成一张子图执行（kernels 模式用：跳掉 resblock 节点，
// 其余（conv_pre/cond/ups_*/conv_post）照旧交给 ggml 图）。子 cgraph 复用。
void gsv_v2sovits::impl::run_range(int s, int e) {
    if (e <= s) return;
    if (!sub) {
        // 只装一张 cgraph（不装张量），1024 个节点绰绰有余
        ggml_init_params ip = { (size_t) 1 << 20, nullptr, true };
        sctx = ggml_init(ip);
        sub = ggml_new_graph_custom(sctx, 1024, false);
    }
    ggml_graph_clear(sub);
    for (int i = s; i < e; i++) ggml_graph_add_node(sub, ggml_graph_node(gdec.g, i));
    if (ggml_backend_graph_compute(backend, sub) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[gsv_v2] dec subgraph compute failed [%d,%d)\n", s, e);
        abort();
    }
}

// 把 dec.resblocks.* 的已折叠 F32 权重重排成 [K][IC][OC]（oc 最快），
// 供 gsv_v2_dec_resblocks 直接用。总量 = 5 个 stage 共 ~44 MB。
void gsv_v2sovits::impl::prepare_kernels() {
    const int n_pairs = 3;
    char wn[192];

    // 第一遍：量一下权重池大小（每对的 C 由张量形状决定）
    size_t total = 0;
    for (int i = 0; i < n_stage; i++)
        for (int b = 0; b < n_res; b++)
            for (int p = 0; p < n_pairs; p++) {
                snprintf(wn, sizeof(wn), "dec.resblocks.%d.convs1.%d.weight", i * n_res + b, p);
                ggml_tensor * w1 = need(wn);
                const int K = (int) w1->ne[0], C = (int) w1->ne[1];
                total += 2 * (size_t) K * C * C + 2 * (size_t) C;
            }
    rb_wpool.assign(total, 0.0f);
    rb_pairs.assign((size_t) n_stage * n_res * n_pairs, gsv_rb_pair{});
    rb_stages.assign((size_t) n_stage, gsv_rb_stage{});

    size_t off = 0;
    std::vector<float> tmp;
    for (int i = 0; i < n_stage; i++) {
        for (int b = 0; b < n_res; b++) {
            const int blk = i * n_res + b;
            for (int p = 0; p < n_pairs; p++) {
                gsv_rb_pair & pr = rb_pairs[(size_t) blk * n_pairs + p];
                // 注意：核长跟**块**走（block0=3, block1=7, block2=11），膨胀率跟对走（1,3,5）
                // —— 一律以权重形状为准，不要用 res_k 推
                const int dil = res_dil[(size_t) p];
                snprintf(wn, sizeof(wn), "dec.resblocks.%d.convs1.%d.weight", blk, p);
                ggml_tensor * w1 = need(wn);
                const int K = (int) w1->ne[0], C = (int) w1->ne[1];
                snprintf(wn, sizeof(wn), "dec.resblocks.%d.convs2.%d.weight", blk, p);
                ggml_tensor * w2 = need(wn);
                if ((int) w2->ne[0] != K ||
                    (int) w1->ne[2] != C || (int) w2->ne[2] != C) {
                    fprintf(stderr, "[gsv_v2] resblock %d conv%d 形状异常: K=%d w1=[%d,%d,%d] w2=[%d,%d,%d]\n",
                            blk, p, K, (int) w1->ne[0], (int) w1->ne[1], (int) w1->ne[2],
                            (int) w2->ne[0], (int) w2->ne[1], (int) w2->ne[2]);
                    abort();
                }
                const size_t wn_f = (size_t) K * C * C;
                pr.w1 = rb_wpool.data() + off; tmp.resize(wn_f);
                ggml_backend_tensor_get(w1, tmp.data(), 0, wn_f * 4);
                gsv_v2_repack_w(tmp.data(), rb_wpool.data() + off, K, C, C);
                off += wn_f;

                snprintf(wn, sizeof(wn), "dec.resblocks.%d.convs1.%d.bias", blk, p);
                pr.b1 = rb_wpool.data() + off; tmp.resize((size_t) C);
                ggml_backend_tensor_get(need(wn), tmp.data(), 0, (size_t) C * 4);
                memcpy(rb_wpool.data() + off, tmp.data(), (size_t) C * 4);
                off += (size_t) C;

                pr.w2 = rb_wpool.data() + off; tmp.resize(wn_f);
                ggml_backend_tensor_get(w2, tmp.data(), 0, wn_f * 4);
                gsv_v2_repack_w(tmp.data(), rb_wpool.data() + off, K, C, C);
                off += wn_f;

                snprintf(wn, sizeof(wn), "dec.resblocks.%d.convs2.%d.bias", blk, p);
                pr.b2 = rb_wpool.data() + off; tmp.resize((size_t) C);
                ggml_backend_tensor_get(need(wn), tmp.data(), 0, (size_t) C * 4);
                memcpy(rb_wpool.data() + off, tmp.data(), (size_t) C * 4);
                off += (size_t) C;

                pr.K = K; pr.dil = dil;
                pr.pad1 = (K / 2) * dil;    // conv1: pad = (K/2)*dil（dilated）
                pr.pad2 = K / 2;            // conv2: pad = K/2, dil=1
            }
        }
    }
    if (off != total) { fprintf(stderr, "[gsv_v2] rb 权重池大小不符 %zu/%zu\n", off, total); abort(); }
    if (getenv("GSV_V2_DEBUG"))
        printf("[gsv_v2] resblock 内核权重池 %.1f MB（%zu 对 conv）\n",
               (double) total * 4 / 1048576.0, rb_pairs.size());
}

bool gsv_v2sovits::decode_wav(const float * z, int T, const float * ge, std::vector<float> & wav) {
    impl & s = *p;
    const auto t0 = std::chrono::steady_clock::now();
    if (T != s.gdec.T) s.build_dec(T);
    std::vector<float> x(z, z + (size_t) T * s.hidden);
    ggml_backend_tensor_set(s.gdec.in, x.data(), 0, x.size() * 4);
    ggml_backend_tensor_set(s.gdec.ge, ge, 0, (size_t) s.gin * 4);
    if (s.use_kernels) {
        // kernels 模式：ggml 只负责 conv_pre/cond/ups_*/conv_post，resblock 交给
        // gsv_v2_dec_resblocks（每个 stage 1 次 fork-join，stage 内无屏障）
        if (s.rb_stages.empty() || s.rb_stages[0].T != (int) s.kup[0]->ne[0]) {
            for (int i = 0; i < s.n_stage; i++) {
                gsv_rb_stage & st = s.rb_stages[(size_t) i];
                st.T = (int) s.kup[(size_t) i]->ne[0];
                st.C = (int) s.kup[(size_t) i]->ne[1];
                st.n_blocks = s.n_res;
                st.n_pairs = 3;
                st.pairs = s.rb_pairs.data() + (size_t) i * s.n_res * 3;
                if ((int) s.ksum[(size_t) i]->ne[0] != st.T || (int) s.ksum[(size_t) i]->ne[1] != st.C) {
                    fprintf(stderr, "[gsv_v2] stage %d 张量形状不符\n", i);
                    return false;
                }
            }
        }
        // GSV_V2_AB=1：同一份输入先跑图、再跑内核，各计一次（干扰大的机器上
        // 跨进程比较毫无意义，同窗口背对背才是可用的相对值）
        const bool ab = getenv("GSV_V2_AB") != nullptr;
        const bool prof = getenv("GSV_V2_PROFILE") != nullptr;
        double t_graph = -1.0;
        if (ab) {
            const auto tg = std::chrono::steady_clock::now();
            if (ggml_backend_graph_compute(s.backend, s.gdec.g) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "[gsv_v2] dec graph compute failed (AB)\n");
                return false;
            }
            t_graph = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - tg).count();
        }
        double t_kern = 0.0;
        for (int i = 0; i <= s.n_stage; i++) {
            s.run_range(s.kparts[(size_t) i].s, s.kparts[(size_t) i].e);
            if (i < s.n_stage) {
                const auto tk = std::chrono::steady_clock::now();
                gsv_v2_dec_resblocks(s.rb_stages[(size_t) i],
                                     (const float *) s.kup[(size_t) i]->data,
                                     (float *) s.ksum[(size_t) i]->data,
                                     s.n_threads > 0 ? s.n_threads : 4);
                const double dt = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - tk).count();
                t_kern += dt;
                if (prof) {
                    const gsv_rb_stage & st = s.rb_stages[(size_t) i];
                    double mac = 0.0;
                    for (int b = 0; b < st.n_blocks; b++)
                        for (int q = 0; q < st.n_pairs; q++) {
                            const gsv_rb_pair & pr = st.pairs[(size_t) b * st.n_pairs + q];
                            mac += 2.0 * (double) st.T * st.C * st.C * pr.K;
                        }
                    printf("  [kern] stage%d C=%d T=%d %8.2f ms %6.1f GFLOPS %5.1f%% peak\n",
                           i, st.C, st.T, dt, 2.0 * mac / (dt * 1e6),
                           100.0 * (2.0 * mac / (dt * 1e6)) / 319.0);
                }
            }
        }
        if (ab || prof)
            printf("  [dec] graph %.1f ms | kernels(仅 resblock) %.1f ms | 2阶段共 %d 次 fork-join\n",
                   t_graph, t_kern, s.n_stage);
        fflush(stdout);
    } else if (ggml_backend_graph_compute(s.backend, s.gdec.g) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[gsv_v2] dec graph compute failed\n");
        return false;
    }
    const size_t n = (size_t) ggml_nelements(s.gdec.out);
    wav.resize(n);
    ggml_backend_tensor_get(s.gdec.out, wav.data(), 0, n * 4);
    // 剖析放在取回输出之后：逐节点重算会撞上 gallocr 的 buffer 复用，
    // 会把最终输出覆盖掉（数值本身不受影响，但必须先把 wav 取走）
    if (getenv("GSV_V2_PROFILE")) v2_profile_graph(s.backend, s.gdec, "dec");
    s.ms_dec = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

// ------------------------------------------------------------------ 全链
bool gsv_v2sovits::synthesize(const int32_t * codes, int Tc, const int32_t * phones, int n_phones,
                              const float * ge, std::vector<float> & wav,
                              double speed, uint64_t seed, float noise_scale,
                              const float * zp_override, int T_override) {
    impl & s = *p;
    wav.clear();
    std::vector<float> m, logs, zp;
    int T = 0;
    if (zp_override && T_override > 0) {
        T = T_override;
        zp.assign(zp_override, zp_override + (size_t) T * s.hidden);
        s.ms_encp = 0.0;
    } else {
        if (!encode_p(codes, Tc, phones, n_phones, ge, m, logs, T, speed)) return false;
        // z_p = m + ε·exp(logs)·noise_scale （ε ~ N(0,1)：Box-Muller + gsv_rng，与 gsv_dit 同式）
        gsv_rng rng(seed ? seed : 1);
        zp.assign((size_t) s.hidden * T, 0.0f);
        for (int c = 0; c < s.hidden; c++) {
            for (int t = 0; t < T; t++) {
                float u1 = rng.uniform01();
                const float u2 = rng.uniform01();
                if (u1 < 1e-30f) u1 = 1e-30f;
                const float eps = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.283185307179586f * u2);
                const float lg = logs[(size_t) c * T + t];
                zp[(size_t) c * T + t] = m[(size_t) c * T + t] + eps * std::exp(lg) * noise_scale;
            }
        }
    }
    std::vector<float> z;
    if (!flow_reverse(zp.data(), T, ge, z)) return false;
    if (getenv("GSV_V2_DEBUG")) {
        auto wr = [](const char * n, const std::vector<float> & v) {
            FILE * f = fopen(n, "wb");
            if (f) { fwrite(v.data(), 4, v.size(), f); fclose(f); }
        };
        wr("_mine_v2_zp.f32", zp);
        wr("_mine_v2_z.f32", z);
    }
    return decode_wav(z.data(), T, ge, wav);
}
