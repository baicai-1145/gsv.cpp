#include "gsv_ar.h"
#include "gsv_device.h"
#include "gsv_progress.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#include <chrono>

static bool g_prof = false;
static double g_ms_build = 0, g_ms_copy = 0, g_ms_compute = 0, g_ms_io = 0;
static double g_ms_mask = 0, g_ms_samp = 0, g_ms_emb = 0;   // host 侧分桶
static int    g_steps = 0;
// prefill 单独计时 (TTFT)
static double g_pf_build = 0, g_pf_copy = 0, g_pf_compute = 0, g_pf_io = 0;
static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}
static void prof_reset() {
    g_ms_build = g_ms_copy = g_ms_compute = g_ms_io = 0;
    g_ms_mask = g_ms_samp = g_ms_emb = 0;
    g_steps = 0;
}
static void prof_report(const char * tag) {
    if (!g_prof) return;
    printf("[prof:%s] prefill(build/copy/compute/io) = %.1f/%.1f/%.1f/%.1f ms\n", tag, g_pf_build, g_pf_copy, g_pf_compute, g_pf_io);
    printf("[prof:%s] steps=%d build=%.1fms copy=%.1fms compute=%.1fms other=%.1fms (per-step: %.2f/%.2f/%.2f)\n",
           tag, g_steps, g_ms_build, g_ms_copy, g_ms_compute, g_ms_io,
           g_ms_build / std::max(1, g_steps), g_ms_copy / std::max(1, g_steps), g_ms_compute / std::max(1, g_steps));
    // host 侧分桶 (每步): 建 mask / 采样 / embedding+PE
    printf("[prof:%s] host per-step: mask=%.3fms sampling=%.3fms emb+pe=%.3fms\n",
           tag, g_ms_mask / std::max(1, g_steps), g_ms_samp / std::max(1, g_steps), g_ms_emb / std::max(1, g_steps));
}

static const int Y_PE_MAX = 4096;

struct ar_weights {
    ggml_tensor * qkv_w, * qkv_b, * out_w, * out_b, * n1w, * n1b, * f1w, * f1b, * f2w, * f2b, * n2w, * n2b;
};

// cache 布局: idx = d + l*HD + h*HD*Lmax + b*HD*Lmax*NH
struct ar_cache {
    int Lmax = 0, B = 0, len = 0, hd = 0, nh = 0;
    std::vector<std::vector<float>> k, v;
    void init(int n_layers, int Lmax_, int B_, int hd_, int nh_) {
        Lmax = Lmax_; B = B_; hd = hd_; nh = nh_; len = 0;
        k.assign(n_layers, std::vector<float>((size_t)hd * Lmax * nh * B, 0.0f));
        v.assign(n_layers, std::vector<float>((size_t)hd * Lmax * nh * B, 0.0f));
    }
    size_t idx(int d, int l, int h, int b) const {
        return (size_t)d + (size_t)l * hd + (size_t)h * hd * Lmax + (size_t)b * hd * Lmax * nh;
    }
};

struct gsv_ar::impl {
    // hparams
    int D = 512, NH = 16, HD = 32, NL = 24, VOCAB = 1025, EOS = 1024, PHONES_VOCAB = 732, BERT_DIM = 1024;
    // weights
    std::vector<ar_weights> ws;
    ggml_tensor * predict = nullptr;
    ggml_tensor * text_emb = nullptr;
    ggml_tensor * audio_emb = nullptr;
    ggml_tensor * bert_proj = nullptr;
    ggml_tensor * bert_proj_b = nullptr;
    ggml_tensor * bert_aug = nullptr;   // [BERT_DIM+1, D] = [W | b] 增广 (prefill 图内投影)
    float text_pe_alpha = 1.0f, audio_pe_alpha = 1.0f;
    ggml_context * wctx = nullptr;
    gguf_context * gf = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    // 前端在 host 侧使用的权重副本 (GPU 后端时 tensor->data 不可直读)
    std::vector<float> h_text_emb, h_audio_emb, h_bert_proj, h_bert_proj_b;
    float h_text_alpha = 1.0f, h_audio_alpha = 1.0f;
    // runtime
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t galloc = nullptr;
    std::vector<float> pe_tab;   // [Y_PE_MAX, D] 行主序 (位置在前的正弦表, 无 alpha)
    ar_cache cache;          // host 镜像 (仅用于 Lmax 记录/兼容)
    int prompt_len = 0;
    // 常驻后端的 KV cache (避免每步回传整个 cache)
    ggml_context * cctx = nullptr;
    ggml_backend_buffer_t cbuf = nullptr;
    std::vector<ggml_tensor *> kc_t, vc_t;   // 各 [HD, Lmax, NH, B]
    int Lmax_alloc = 0;
    // KV cache 元素类型: F32 (默认, 与历史版本数值完全一致) 或 F16 (GSV_AR_KV_F16=1)。
    // 解码每步 flash 都要把整个 KV 历史读一遍 —— L≈450 时 44MB/token, 占解码 DRAM
    // 流量的 1/3; F16 把这份流量减半 (llama.cpp 的 KV cache 默认也是 F16)。
    // 默认关: 保持既有 v5/v2 路径逐位不变; 需要在 gsv_ar 之外选开。
    bool kv_f16 = false;

    void alloc_cache(int Lmax, int B) {
        if (cctx && Lmax_alloc >= Lmax && cache.B == B) return;
        // 缓存要重建: 解码图缓存 dgs 直接引用了 kc_t/vc_t, 必须先让它失效,
        // 否则下一个请求若复用旧图 (L <= dgs.Lmax) 就是在写已释放的显存 (use-after-free)。
        dec_free();
        if (cbuf) { ggml_backend_buffer_free(cbuf); cbuf = nullptr; }
        if (cctx) { ggml_free(cctx); cctx = nullptr; }
        ggml_init_params ip = { ggml_tensor_overhead() * (2 * NL + 8), NULL, true };
        cctx = ggml_init(ip);
        kc_t.resize(NL); vc_t.resize(NL);
        for (int li = 0; li < NL; li++) {
            kc_t[li] = ggml_new_tensor_4d(cctx, kv_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32, HD, Lmax, NH, B);
            vc_t[li] = ggml_new_tensor_4d(cctx, kv_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32, HD, Lmax, NH, B);
        }
        cbuf = ggml_backend_alloc_ctx_tensors(cctx, backend);
        if (!cbuf) { fprintf(stderr, "[gsv_ar] cache alloc failed\n"); exit(1); }
        for (int li = 0; li < NL; li++) {
            ggml_backend_tensor_memset(kc_t[li], 0, 0, ggml_nbytes(kc_t[li]));
            ggml_backend_tensor_memset(vc_t[li], 0, 0, ggml_nbytes(vc_t[li]));
        }
        Lmax_alloc = Lmax;
        cache.B = B;
    }

    // 释放 cache（批量后 B 大，及时把显存还给 DiT/vocoder）
    void free_cache() {
        dec_free();
        if (cbuf) { ggml_backend_buffer_free(cbuf); cbuf = nullptr; }
        if (cctx) { ggml_free(cctx); cctx = nullptr; }
        Lmax_alloc = 0;
        cache.B = 0;
        kc_t.clear();
        vc_t.clear();
    }

    // 从常驻 cache 取 [HD, L, NH, B] 视图
    ggml_tensor * k_view(ggml_context * ctx, int li, int L) const {
        ggml_tensor * t = kc_t[li];
        return ggml_view_4d(ctx, t, HD, L, NH, cache.B, t->nb[1], t->nb[2], t->nb[3], 0);
    }
    ggml_tensor * v_view(ggml_context * ctx, int li, int L) const {
        ggml_tensor * t = vc_t[li];
        return ggml_view_4d(ctx, t, HD, L, NH, cache.B, t->nb[1], t->nb[2], t->nb[3], 0);
    }
    ggml_tensor * k_col_view(ggml_context * ctx, int li, int col) const {
        ggml_tensor * t = kc_t[li];
        return ggml_view_4d(ctx, t, HD, 1, NH, cache.B, t->nb[1], t->nb[2], t->nb[3], (int64_t) col * t->nb[1]);
    }
    ggml_tensor * v_col_view(ggml_context * ctx, int li, int col) const {
        ggml_tensor * t = vc_t[li];
        return ggml_view_4d(ctx, t, HD, 1, NH, cache.B, t->nb[1], t->nb[2], t->nb[3], (int64_t) col * t->nb[1]);
    }

    bool fuse_ln = true, fuse_act = true;   // 融合算子开关 (GSV_NO_FUSE / GSV_NO_FUSE_LN / GSV_NO_FUSE_ACT)
    // per-layer 融合 LN 的 [w;b] 打包张量 (与 GGUF 无关, 加载时在专用小 buffer 里生成)
    struct ln_wb { ggml_tensor * wb = nullptr; int n = 0; };
    std::vector<ln_wb> ln_pack;             // [li*2 + 0] = norm1, [li*2 + 1] = norm2
    ggml_context * pctx = nullptr;
    ggml_backend_buffer_t pbuf = nullptr;

    // LN(x + r + bias)*w + b; wb = [w;b] 打包; r/bias 可为 NULL
    ggml_tensor * ln_affine(ggml_context * ctx, ggml_tensor * x, ggml_tensor * r, ggml_tensor * bias,
                            ggml_tensor * wb, int n, float eps) {
        if (fuse_ln) return ggml_layernorm_affine(ctx, x, r, bias, wb, eps);
        if (r)    x = ggml_add(ctx, x, r);
        if (bias) x = ggml_add(ctx, x, bias);
        ggml_tensor * w = ggml_view_1d(ctx, wb, n, 0);
        ggml_tensor * b = ggml_view_1d(ctx, wb, n, (size_t) n * 4);
        return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), w), b);
    }
    // act(a + b): 融合算子不可用时退回 add + 激活
    ggml_tensor * add_act(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, int act) {
        if (fuse_act) return ggml_add_act(ctx, a, b, act);
        ggml_tensor * t = ggml_add(ctx, a, b);
        return act == GGML_ACT_RELU ? ggml_relu(ctx, t) : ggml_gelu_erf(ctx, t);
    }

    void mk_pe() {
        pe_tab.assign((size_t)Y_PE_MAX * D, 0.0f);
        for (int pos = 0; pos < Y_PE_MAX; pos++)
            for (int i = 0; i < D; i += 2) {
                const float div = std::exp(i * -(std::log(10000.0f) / (float)D));
                pe_tab[(size_t)pos * D + i]     = std::sin(pos * div);
                pe_tab[(size_t)pos * D + i + 1] = std::cos(pos * div);
            }
    }

    // 构造 batched 输入: xy_pos [B,S,D] 与 mask [B,S,S] (-inf/0); S = max_len + prompt_len
    // feat [BERT_DIM+1, S, B]: 文本位置 = bert 特征 + 末行 1 (给增广权重 [W|b] 的 bias 用);
    //                         prompt / padding 位置 = 全 0 (不加 bert 也不加 bias)
    void build_inputs(const std::vector<gsv_ar_request> & reqs, std::vector<float> & xy, std::vector<float> & feat,
                      std::vector<float> & mask, int & S, int & max_len) const {
        const int B = (int) reqs.size();
        max_len = 0;
        for (const auto & r : reqs) max_len = std::max(max_len, (int) r.phones.size());
        S = max_len + prompt_len;
        xy.assign((size_t)B * S * D, 0.0f);
        feat.assign((size_t)B * S * (BERT_DIM + 1), 0.0f);
        mask.assign((size_t)B * S * S, 0.0f);
        const float * te  = h_text_emb.data();
        const float * ae  = h_audio_emb.data();
        for (int b = 0; b < B; b++) {
            const gsv_ar_request & r = reqs[b];
            const int T = (int) r.phones.size();
            const int pad = max_len - T;                          // 左 padding
            for (int t = 0; t < T; t++) {
                float * dst = xy.data() + ((size_t)b * S + pad + t) * D;
                const float * tok = te + (size_t) r.phones[t] * D;
                float * fdst = feat.data() + ((size_t)b * S + pad + t) * (BERT_DIM + 1);
                for (int d = 0; d < D; d++) dst[d] = tok[d] + h_text_alpha * pe_tab[(size_t)t * D + d];
                for (int k = 0; k < BERT_DIM; k++) fdst[k] = r.bert[(size_t)k * T + t];
                fdst[BERT_DIM] = 1.0f;                            // 增广位: 触发 bias
            }
            for (int t = 0; t < prompt_len; t++) {
                float * dst = xy.data() + ((size_t)b * S + max_len + t) * D;
                const float * emb = ae + (size_t) r.prompt[t] * D;
                for (int d = 0; d < D; d++)
                    dst[d] = emb[d] + h_audio_alpha * pe_tab[(size_t)t * D + d];
            }
            // mask: 左 padding key 屏蔽; x query 看不到 y key; y query causal
            for (int q = 0; q < S; q++)
                for (int k = 0; k < S; k++) {
                    bool m;
                    if (k < pad) m = true;
                    else if (q < max_len) m = (k >= max_len);
                    else m = (k > q);
                    mask[(size_t)b * S * S + (size_t)q * S + k] = m ? -INFINITY : 0.0f;
                }
        }
    }

    // 首步: 全前向, 输出 logits [B,V] 与各层 K/V cache
    void run_first(const std::vector<float> & xy_host, const std::vector<float> & feat_host,
                   const std::vector<float> & mask_host, int S,
                   std::vector<float> & logits_out) {
        const int B = cache.B;
        const double t_a = now_ms();
        ggml_init_params ip = { ggml_tensor_overhead() * 16384, NULL, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, S, B);
        ggml_tensor * feat = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, BERT_DIM + 1, S, B);
        ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, 1, B);
        ggml_set_input(x); ggml_set_input(feat); ggml_set_input(mask);
        ggml_tensor * mask16 = ggml_cast(ctx, mask, GGML_TYPE_F16);
        // bert 投影在图内 (增广权重 [W|b] × [feat; 1]): host 侧不再做 30M MAC 循环
        ggml_tensor * cur = ggml_add(ctx, x, ggml_mul_mat(ctx, bert_aug, feat));
        std::vector<ggml_tensor *> kout(NL), vout(NL);
        for (int li = 0; li < NL; li++) {
            ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, ws[li].qkv_w, cur), ws[li].qkv_b);
            ggml_tensor * qkv4 = ggml_reshape_4d(ctx, qkv, D, 3, S, B);
            // 视图直送 (省每层 6 次 cont): qkv 行内序是 (hd, nh), S 维步长 = 12D
            // -> (HD,NH,S,B) 视图再 permute(0,2,1,3) 即 flash 布局 (S>1 时真正的转置由
            //    读取端的步长完成, flash/cpy 都按传入步长寻址)
            const size_t nbh = 4 * HD;
            ggml_tensor * qv = ggml_view_4d(ctx, qkv4, HD, NH, S, B, nbh, qkv4->nb[2], qkv4->nb[3], 0);
            ggml_tensor * kv = ggml_view_4d(ctx, qkv4, HD, NH, S, B, nbh, qkv4->nb[2], qkv4->nb[3], qkv4->nb[1]);
            ggml_tensor * vv = ggml_view_4d(ctx, qkv4, HD, NH, S, B, nbh, qkv4->nb[2], qkv4->nb[3], 2 * qkv4->nb[1]);
            ggml_tensor * qh = ggml_permute(ctx, qv, 0, 2, 1, 3);
            ggml_tensor * kh = ggml_permute(ctx, kv, 0, 2, 1, 3);
            ggml_tensor * vh = ggml_permute(ctx, vv, 0, 2, 1, 3);
            kout[li] = kh; vout[li] = vh;
            ggml_tensor * attn_out = ggml_flash_attn_ext(ctx, qh, kh, vh, mask16, 1.0f / std::sqrt((float)HD), 0.0f, 0.0f);
            attn_out = ggml_reshape_4d(ctx, attn_out, D, S, B, 1);
            attn_out = ggml_reshape_3d(ctx, attn_out, D, S, B);
            ggml_tensor * o = ggml_mul_mat(ctx, ws[li].out_w, attn_out);
            ggml_tensor * n1 = ln_affine(ctx, cur, o, ws[li].out_b, ln_pack[2*li+0].wb, ln_pack[2*li+0].n, 1e-5f);
            ggml_tensor * h = add_act(ctx, ggml_mul_mat(ctx, ws[li].f1w, n1), ws[li].f1b, GGML_ACT_RELU);
            h = ggml_mul_mat(ctx, ws[li].f2w, h);
            cur = ln_affine(ctx, n1, h, ws[li].f2b, ln_pack[2*li+1].wb, ln_pack[2*li+1].n, 1e-5f);
        }
        ggml_tensor * logits = ggml_mul_mat(ctx, predict, cur);
        ggml_tensor * last_view = ggml_view_3d(ctx, logits, VOCAB, 1, B, logits->nb[1], logits->nb[2], (int64_t)(S - 1) * logits->nb[1]);
        ggml_tensor * last = ggml_cont(ctx, last_view);   // tensor_get 是裸 memcpy, 需连续
        ggml_set_output(last);
        // K/V 直接写入常驻 cache (无 host 往返)
        std::vector<ggml_tensor *> kcpy(NL), vcpy(NL);
        for (int li = 0; li < NL; li++) {
            kcpy[li] = ggml_cpy(ctx, kout[li], ggml_view_4d(ctx, kc_t[li], HD, S, NH, B, kc_t[li]->nb[1], kc_t[li]->nb[2], kc_t[li]->nb[3], 0));
            vcpy[li] = ggml_cpy(ctx, vout[li], ggml_view_4d(ctx, vc_t[li], HD, S, NH, B, vc_t[li]->nb[1], vc_t[li]->nb[2], vc_t[li]->nb[3], 0));
        }
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16384, false);
        ggml_build_forward_expand(graph, last);
        for (int li = 0; li < NL; li++) { ggml_build_forward_expand(graph, kcpy[li]); ggml_build_forward_expand(graph, vcpy[li]); }
        ggml_gallocr_alloc_graph(galloc, graph);
        const double t_b = now_ms();
        ggml_backend_tensor_set(x, xy_host.data(), 0, xy_host.size() * 4);
        ggml_backend_tensor_set(feat, feat_host.data(), 0, feat_host.size() * 4);
        ggml_backend_tensor_set(mask, mask_host.data(), 0, mask_host.size() * 4);
        const double t_c = now_ms();
        ggml_backend_graph_compute(backend, graph);
        const double t_d = now_ms();
        logits_out.assign((size_t)VOCAB * B, 0.0f);
        ggml_backend_tensor_get(last, logits_out.data(), 0, logits_out.size() * 4);
        const double t_e = now_ms();
        g_pf_build += t_b - t_a; g_pf_copy += t_c - t_b; g_pf_compute += t_d - t_c; g_pf_io += t_e - t_d;
        cache.len = S;
        ggml_free(ctx);
    }

    // 解码步: x [B,1,D]; 每序列 mask = (k < pad_len[b])
    void run_decode(const std::vector<float> & x_host, const std::vector<int> & pad_len, std::vector<float> & logits_out) {
        const double t_a = now_ms();
        const int B = cache.B;
        const int L = cache.len + 1;
        const double t_m0 = now_ms();
        std::vector<ggml_fp16_t> mask_host((size_t)L * B);
        for (int b = 0; b < B; b++)
            for (int k = 0; k < L; k++)
                mask_host[(size_t)k + (size_t)b * L] = ggml_fp32_to_fp16((k < pad_len[b]) ? -INFINITY : 0.0f);
        g_ms_mask += now_ms() - t_m0;

        ggml_init_params ip = { ggml_tensor_overhead() * 2048, NULL, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, B);
        ggml_set_input(x);
        ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, L, 1, 1, B);
        ggml_set_input(mask);
        ggml_tensor * cur = x;
        std::vector<ggml_tensor *> writes;   // KV cpy 副作用根 (必须先于 flash 执行)
        for (int li = 0; li < NL; li++) {
            ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, ws[li].qkv_w, cur), ws[li].qkv_b);
            ggml_tensor * qkv4 = ggml_reshape_4d(ctx, qkv, D, 3, 1, B);
            // S=1: q/k/v 在 qkv 里的内存序已是 (hd, nh)，可直接用 (HD,1,NH,B) 视图送 flash / 送 KV cpy
            // —— 省掉每层 6 次 ggml_cont (3 次切 q/k/v + 3 次 permute)，数值完全相同
            ggml_tensor * qh      = ggml_view_4d(ctx, qkv4, HD, 1, NH, B, 4 * HD, 4 * HD, qkv4->nb[3], 0);
            ggml_tensor * ksrc    = ggml_view_4d(ctx, qkv4, HD, 1, NH, B, 4 * HD, 4 * HD, qkv4->nb[3], qkv4->nb[1]);
            ggml_tensor * vsrc    = ggml_view_4d(ctx, qkv4, HD, 1, NH, B, 4 * HD, 4 * HD, qkv4->nb[3], 2 * qkv4->nb[1]);
            // 新列直接写进常驻 cache 的第 len 列 (cpy 副作用), flash 读 cache 的 [0, len] 视图。
            // 不再 concat 整段 cache —— 那会每层每步复制 L 列 K/V (L≈200 时 2.0 ms/token,
            // 实测占解码 29%); 写->读顺序由根的先后顺序保证: cpy 根先 expand, logits 根后 expand,
            // ggml 的节点序即执行序, 且节点间有 barrier。
            writes.push_back(ggml_cpy(ctx, ksrc, k_col_view(ctx, li, cache.len)));
            writes.push_back(ggml_cpy(ctx, vsrc, v_col_view(ctx, li, cache.len)));
            ggml_tensor * kfull = k_view(ctx, li, cache.len + 1);
            ggml_tensor * vfull = v_view(ctx, li, cache.len + 1);
            ggml_tensor * attn_out = ggml_flash_attn_ext(ctx, qh, kfull, vfull, mask, 1.0f / std::sqrt((float)HD), 0.0f, 0.0f);
            attn_out = ggml_reshape_3d(ctx, attn_out, D, 1, B);
            ggml_tensor * o = ggml_mul_mat(ctx, ws[li].out_w, attn_out);
            ggml_tensor * n1 = ln_affine(ctx, cur, o, ws[li].out_b, ln_pack[2*li+0].wb, ln_pack[2*li+0].n, 1e-5f);
            ggml_tensor * h = add_act(ctx, ggml_mul_mat(ctx, ws[li].f1w, n1), ws[li].f1b, GGML_ACT_RELU);
            h = ggml_mul_mat(ctx, ws[li].f2w, h);
            cur = ln_affine(ctx, n1, h, ws[li].f2b, ln_pack[2*li+1].wb, ln_pack[2*li+1].n, 1e-5f);
        }
        ggml_tensor * logits = ggml_mul_mat(ctx, predict, cur);
        ggml_set_output(logits);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4096, false);
        for (ggml_tensor * w : writes) ggml_build_forward_expand(graph, w);   // 先写 KV (副作用根)
        ggml_build_forward_expand(graph, logits);
        ggml_gallocr_alloc_graph(galloc, graph);
        const double t_b = now_ms();
        ggml_backend_tensor_set(x, x_host.data(), 0, x_host.size() * 4);
        ggml_backend_tensor_set(mask, mask_host.data(), 0, mask_host.size() * 2);
        const double t_c = now_ms();
        ggml_backend_graph_compute(backend, graph);
        const double t_d = now_ms();
        logits_out.assign((size_t)VOCAB * B, 0.0f);
        ggml_backend_tensor_get(logits, logits_out.data(), 0, logits_out.size() * 4);
        const double t_e = now_ms();
        g_ms_build += t_b - t_a; g_ms_copy += t_c - t_b; g_ms_compute += t_d - t_c; g_ms_io += t_e - t_d;
        g_steps++;
        cache.len = L;
        ggml_free(ctx);
    }

    // ---- 解码图缓存 ----
    // 每步重建图是 bs=1 时约 1/4 的开销 (0.78ms/3.3ms)。这里把解码图做成"固定 key 数":
    //   key = concat(cache[0..Lmax), 新列)  -> (HD, Lmax+1, NH, B), 形状恒定
    //   新列用 set_rows 按 idx(=当前长度) 写回 cache (副作用, 与 flash 读的行不重叠)
    //   mask 屏蔽左 padding 与 [L..Lmax) 的无效槽, 新列槽 (Lmax) 放行
    // 于是 (Lmax, B) 不变时图/显存计划可复用, 只在跨桶 (L 超过 Lmax) 时重建。
    bool use_graph_cache = true;
    struct dec_state {
        ggml_context * ctx = nullptr;
        ggml_cgraph  * graph = nullptr;
        ggml_tensor  * x = nullptr, * mask = nullptr, * idx = nullptr, * logits = nullptr;
        int Lmax = 0, B = 0;
    } dgs;

    void dec_free() {
        if (dgs.ctx) { ggml_free(dgs.ctx); dgs.ctx = nullptr; }
        dgs.graph = nullptr; dgs.x = dgs.mask = dgs.idx = dgs.logits = nullptr;
        dgs.Lmax = 0; dgs.B = 0;
    }

    void dec_build(int Lmax) {
        dec_free();
        const int B = cache.B;
        ggml_init_params ip = { ggml_tensor_overhead() * 2048, NULL, true };
        ggml_context * ctx = ggml_init(ip);
        dgs.x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, B);            ggml_set_input(dgs.x);
        dgs.mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, Lmax + 1, 1, 1, B);  ggml_set_input(dgs.mask);
        dgs.idx  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);                  ggml_set_input(dgs.idx);

        ggml_tensor * cur = dgs.x;
        std::vector<ggml_tensor *> writes;
        for (int li = 0; li < NL; li++) {
            ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, ws[li].qkv_w, cur), ws[li].qkv_b);
            ggml_tensor * qkv4 = ggml_reshape_4d(ctx, qkv, D, 3, 1, B);
            ggml_tensor * qh   = ggml_view_4d(ctx, qkv4, HD, 1, NH, B, 4 * HD, 4 * HD, qkv4->nb[3], 0);
            ggml_tensor * ksrc = ggml_view_4d(ctx, qkv4, HD, 1, NH, B, 4 * HD, 4 * HD, qkv4->nb[3], qkv4->nb[1]);
            ggml_tensor * vsrc = ggml_view_4d(ctx, qkv4, HD, 1, NH, B, 4 * HD, 4 * HD, qkv4->nb[3], 2 * qkv4->nb[1]);
            // KV 追加: 按 idx 写入 cache 的第 len 行 (同一行索引对 K/V 与所有 head/batch 生效)
            writes.push_back(ggml_set_rows(ctx, kc_t[li], ksrc, dgs.idx));
            writes.push_back(ggml_set_rows(ctx, vc_t[li], vsrc, dgs.idx));
            ggml_tensor * kfull = ggml_concat(ctx, k_view(ctx, li, Lmax), ksrc, 1);
            ggml_tensor * vfull = ggml_concat(ctx, v_view(ctx, li, Lmax), vsrc, 1);
            ggml_tensor * attn_out = ggml_flash_attn_ext(ctx, qh, kfull, vfull, dgs.mask,
                                                        1.0f / std::sqrt((float)HD), 0.0f, 0.0f);
            attn_out = ggml_reshape_3d(ctx, attn_out, D, 1, B);
            ggml_tensor * o = ggml_mul_mat(ctx, ws[li].out_w, attn_out);
            ggml_tensor * n1 = ln_affine(ctx, cur, o, ws[li].out_b, ln_pack[2*li+0].wb, ln_pack[2*li+0].n, 1e-5f);
            ggml_tensor * h = add_act(ctx, ggml_mul_mat(ctx, ws[li].f1w, n1), ws[li].f1b, GGML_ACT_RELU);
            h = ggml_mul_mat(ctx, ws[li].f2w, h);
            cur = ln_affine(ctx, n1, h, ws[li].f2b, ln_pack[2*li+1].wb, ln_pack[2*li+1].n, 1e-5f);
        }
        dgs.logits = ggml_mul_mat(ctx, predict, cur);
        ggml_set_output(dgs.logits);
        dgs.graph = ggml_new_graph_custom(ctx, 4096, false);
        ggml_build_forward_expand(dgs.graph, dgs.logits);
        for (ggml_tensor * w : writes) ggml_build_forward_expand(dgs.graph, w);   // set_rows 是副作用根
        ggml_gallocr_alloc_graph(galloc, dgs.graph);
        dgs.Lmax = Lmax; dgs.B = B; dgs.ctx = ctx;
    }

    // 桶: L + 25% 余量 (上限 Lmax_alloc)。紧桶让"每步多算 (Lmax-L) 个 key 的
    // attention/concat"几乎为零, 同时把重建摊到 ~L/4 步; 2 的幂桶在 CPU 上会明显变慢
    int bucket_for(int L) const {
        int b = L + std::max(8, L / 4);
        if (b > Lmax_alloc) b = Lmax_alloc;
        return b;
    }

    void run_decode_cached(const std::vector<float> & x_host, const std::vector<int> & pad_len,
                           std::vector<float> & logits_out) {
        const double t_a = now_ms();
        const int B = cache.B;
        const int L = cache.len + 1;
        if (dgs.ctx == nullptr || dgs.B != B || L > dgs.Lmax) {
            int Lmax = bucket_for(L);
            if (Lmax > Lmax_alloc) Lmax = Lmax_alloc;
            if (Lmax < L) { run_decode(x_host, pad_len, logits_out); return; }   // 兜底
            dec_build(Lmax);
        }
        const int Lmax = dgs.Lmax;
        // 有效 key 只有: 历史 [0, cache.len) 与图内新列 (索引 Lmax)。
        // 注意尾部屏蔽必须从 cache.len 开始 (未写入的槽在跨调用/跨桶后是旧数据, 不能放行)。
        const int Lh = cache.len;
        std::vector<ggml_fp16_t> mask_host((size_t)(Lmax + 1) * B);
        for (int b = 0; b < B; b++)
            for (int k = 0; k <= Lmax; k++) {
                const bool bad = (k < pad_len[b]) || (k >= Lh && k < Lmax);
                mask_host[(size_t) k + (size_t) b * (Lmax + 1)] = ggml_fp32_to_fp16(bad ? -INFINITY : 0.0f);
            }
        const double t_b = now_ms();
        const int32_t idx_host = cache.len;   // 新列写入的行号 = 当前长度
        ggml_backend_tensor_set(dgs.x,    x_host.data(), 0, x_host.size() * 4);
        ggml_backend_tensor_set(dgs.mask, mask_host.data(), 0, mask_host.size() * 2);
        ggml_backend_tensor_set(dgs.idx,  &idx_host, 0, 4);
        const double t_c = now_ms();
        ggml_backend_graph_compute(backend, dgs.graph);
        const double t_d = now_ms();
        logits_out.assign((size_t)VOCAB * B, 0.0f);
        ggml_backend_tensor_get(dgs.logits, logits_out.data(), 0, logits_out.size() * 4);
        const double t_e = now_ms();
        g_ms_build += t_b - t_a; g_ms_copy += t_c - t_b; g_ms_compute += t_d - t_c; g_ms_io += t_e - t_d;
        g_steps++;
        cache.len = L;
    }

    void emb_lookup(int tok, float * out) const {
        const float * base = h_audio_emb.data();
        for (int d = 0; d < D; d++) out[d] = base[(size_t)tok * D + d];
    }
};

gsv_ar::gsv_ar() : p(new impl()) {}
gsv_ar::~gsv_ar() { p->dec_free();
    if (p->galloc) ggml_gallocr_free(p->galloc);
    if (p->backend) ggml_backend_free(p->backend);
    if (p->wbuf) ggml_backend_buffer_free(p->wbuf);
    if (p->gf) gguf_free(p->gf);
    delete p;
}

int gsv_ar::hidden_dim() const { return p->D; }
int gsv_ar::vocab_size() const { return p->VOCAB; }
int gsv_ar::eos() const { return p->EOS; }
int gsv_ar::bert_dim() const { return p->BERT_DIM; }
int gsv_ar::prompt_len_expected() const { return p->prompt_len; }

gsv_ar * gsv_ar::load(const std::string & gguf_path, const gsv_ar_cfg & cfg) {
    gsv_ar * m = new gsv_ar();
    impl & s = *m->p;
    // 注意: ggml-vulkan 在**静态初始化**阶段就读取 GGML_VK_DISABLE_COOPMAT2 (早于 main),
    // 因此内部设置无效, 只能提示调用方在启动前设置 (实测: 外部设置 logits Δ 0.0035, 未设置 0.0201)

    // 设备选择 (cfg.device: "" = CPU, "vulkan"/"gpu" = 第一个 GPU 设备)
    ggml_backend_dev_t dev = nullptr;
    {
        int gpu_idx = 0; bool gpu_explicit = false;
        const bool want_gpu = gsv_device_parse(cfg.device, &gpu_idx, &gpu_explicit);
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
        if (!dev) {
            for (int i = 0; i < n_dev && !dev; i++) {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) dev = d;
            }
        }
                if (want_gpu && gpu_explicit && (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU)) {
            fprintf(stderr, "[device] GPU %d not found (run --list-devices)\n", gpu_idx);
            delete m; return nullptr;
        }
if (!dev) { fprintf(stderr, "[gsv_ar] no usable backend device"); delete m; return nullptr; }
        if (cfg.verbose) printf("[gsv_ar] device: %s (%s)", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        // coopmat2 是 VK_NV_cooperative_matrix2（NVIDIA 专有扩展）：非 NVIDIA 设备上
        // ggml 会自动退回 coopmat1（VK_KHR_cooperative_matrix）或纯 shader 路径，不需要干预。
        // 只有 NVIDIA + 未禁用时才提醒（AR 是 F32 matmul，coopmat2 会掉精度）。
        if (want_gpu && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU &&
            getenv("GGML_VK_DISABLE_COOPMAT2") == nullptr) {
            const char * desc = ggml_backend_dev_description(dev);
            const bool nvidia = desc && (strstr(desc, "NVIDIA") || strstr(desc, "GeForce") ||
                                         strstr(desc, "Quadro") || strstr(desc, "Tesla"));
            if (nvidia && cfg.verbose) {
                // Kept for --verbose only: coopmat2 is on by default and the precision impact is
                // accepted (our torch comparisons ran with it enabled too).
                fprintf(stderr, "[gsv_ar] note: Vulkan coopmat2 enabled; AR logits delta 0.0035 -> "
                                "0.0201 (set GGML_VK_DISABLE_COOPMAT2=1 to disable)\n");
            }
        }
    }
    s.backend = ggml_backend_dev_init(dev, nullptr);
    if (!s.backend) { fprintf(stderr, "[gsv_ar] backend init failed"); delete m; return nullptr; }
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && cfg.n_threads > 0)
        ggml_backend_cpu_set_n_threads(s.backend, cfg.n_threads);
    s.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s.backend));

    gguf_init_params gip = { /*no_alloc*/ true, &s.wctx };
    s.gf = gguf_init_from_file(gguf_path.c_str(), gip);
    if (!s.gf) { fprintf(stderr, "[gsv_ar] failed to open %s\n", gguf_path.c_str()); delete m; return nullptr; }
    // 后端 buffer 分配 + 从文件上传 (no_alloc 模式下 tensor 没有数据)
    s.wbuf = ggml_backend_alloc_ctx_tensors(s.wctx, s.backend);
    if (!s.wbuf) { fprintf(stderr, "[gsv_ar] weight buffer alloc failed"); delete m; return nullptr; }
    {
        FILE * fp = fopen(gguf_path.c_str(), "rb");
        if (!fp) { fprintf(stderr, "[gsv_ar] reopen failed"); delete m; return nullptr; }
        const size_t data_off = gguf_get_data_offset(s.gf);
        const int64_t n_tensors = gguf_get_n_tensors(s.gf);
        for (int64_t ti = 0; ti < n_tensors; ti++) {
            const char * tname = gguf_get_tensor_name(s.gf, ti);
            ggml_tensor * tt = ggml_get_tensor(s.wctx, tname);
            if (!tt) continue;
            const size_t nbytes = ggml_nbytes(tt);
            std::vector<char> buf(nbytes);
            if (fseek(fp, (long)(data_off + gguf_get_tensor_offset(s.gf, ti)), SEEK_SET) != 0 ||
                fread(buf.data(), 1, nbytes, fp) != nbytes) {
                fprintf(stderr, "[gsv_ar] read tensor %s failed", tname);
                fclose(fp); delete m; return nullptr;
            }
            ggml_backend_tensor_set(tt, buf.data(), 0, nbytes);
        }
        fclose(fp);
    }
    auto kv_u32 = [&](const char * key, int def) {
        const int64_t id = gguf_find_key(s.gf, key);
        return id < 0 ? def : (int) gguf_get_val_u32(s.gf, id);
    };
    s.D            = kv_u32("ar.hidden_dim", 512);
    s.NH           = kv_u32("ar.head", 16);
    s.NL           = kv_u32("ar.n_layer", 24);
    s.VOCAB        = kv_u32("ar.vocab_size", 1025);
    s.PHONES_VOCAB = kv_u32("ar.phoneme_vocab_size", 732);
    s.EOS          = kv_u32("ar.eos", 1024);
    s.BERT_DIM     = 1024;
    s.HD           = s.D / s.NH;

    auto t = [&](const char * n) {
        ggml_tensor * tt = ggml_get_tensor(s.wctx, n);
        if (!tt) { fprintf(stderr, "[gsv_ar] missing tensor %s\n", n); return (ggml_tensor *) nullptr; }
        return tt;
    };
    s.ws.resize(s.NL);
    char buf[128];
    for (int li = 0; li < s.NL; li++) {
        #define GW_L(f, nm) snprintf(buf, sizeof(buf), "transformer.block%d." nm, li); if (!(s.ws[li].f = t(buf))) { delete m; return nullptr; }
        GW_L(qkv_w, "qkv_w"); GW_L(qkv_b, "qkv_b"); GW_L(out_w, "out_w"); GW_L(out_b, "out_b");
        GW_L(n1w, "norm1_w"); GW_L(n1b, "norm1_b"); GW_L(n2w, "norm2_w"); GW_L(n2b, "norm2_b");
        GW_L(f1w, "ffn1_w"); GW_L(f1b, "ffn1_b"); GW_L(f2w, "ffn2_w"); GW_L(f2b, "ffn2_b");
        #undef GW_L
    }
    // [w;b] 打包 (供 4-src 融合 LN): 专用 ctx + buffer, 只占 ~200KB
    {
        ggml_init_params pp = { ggml_tensor_overhead() * (2 * s.NL + 32), NULL, true };
        s.pctx = ggml_init(pp);
        std::vector<ggml_tensor *> tmp(2 * s.NL, nullptr);
        ggml_tensor * aug = ggml_new_tensor_2d(s.pctx, GGML_TYPE_F32, s.BERT_DIM + 1, s.D);
        for (int li = 0; li < s.NL; li++) {
            tmp[2*li+0] = ggml_new_tensor_1d(s.pctx, GGML_TYPE_F32, 2 * s.D);
            tmp[2*li+1] = ggml_new_tensor_1d(s.pctx, GGML_TYPE_F32, 2 * s.D);
        }
        s.pbuf = ggml_backend_alloc_ctx_tensors(s.pctx, s.backend);
        if (!s.pbuf) { fprintf(stderr, "[gsv_ar] packed wb buffer alloc failed\n"); delete m; return nullptr; }
        s.ln_pack.resize(2 * s.NL);
        for (int li = 0; li < s.NL; li++) {
            ggml_tensor * w1 = s.ws[li].n1w, * b1 = s.ws[li].n1b;
            ggml_tensor * w2 = s.ws[li].n2w, * b2 = s.ws[li].n2b;
            const int n1 = (int) w1->ne[0], n2 = (int) w2->ne[0];
            std::vector<float> buf(std::max(2 * n1, 2 * n2));
            ggml_backend_tensor_get(w1, buf.data(), 0, (size_t) n1 * 4);
            ggml_backend_tensor_get(b1, buf.data() + n1, 0, (size_t) n1 * 4);
            ggml_backend_tensor_set(tmp[2*li+0], buf.data(), 0, (size_t) 2 * n1 * 4);
            ggml_backend_tensor_get(w2, buf.data(), 0, (size_t) n2 * 4);
            ggml_backend_tensor_get(b2, buf.data() + n2, 0, (size_t) n2 * 4);
            ggml_backend_tensor_set(tmp[2*li+1], buf.data(), 0, (size_t) 2 * n2 * 4);
            s.ln_pack[2*li+0] = { tmp[2*li+0], n1 };
            s.ln_pack[2*li+1] = { tmp[2*li+1], n2 };
        }
        s.bert_aug = aug;   // 数据在 host 副本生成后填 (见下)
    }
    if (!(s.predict      = t("ar.predict")))      { delete m; return nullptr; }
    if (!(s.text_emb     = t("ar.text_emb")))     { delete m; return nullptr; }
    if (!(s.audio_emb    = t("ar.audio_emb")))    { delete m; return nullptr; }
    if (!(s.bert_proj    = t("ar.bert_proj")))    { delete m; return nullptr; }
    if (!(s.bert_proj_b  = t("ar.bert_proj_b")))  { delete m; return nullptr; }
    // host 侧依赖的可直读权重必须是 F32 (bert_proj 当前在 host 侧投影; embedding 查表同理)
    if (s.bert_proj->type != GGML_TYPE_F32 || s.text_emb->type != GGML_TYPE_F32 || s.audio_emb->type != GGML_TYPE_F32) {
        fprintf(stderr, "[gsv_ar] text_emb/audio_emb/bert_proj 必须为 F32 (host 侧直读); 当前为 %d/%d/%d\n",
                s.text_emb->type, s.audio_emb->type, s.bert_proj->type);
        delete m; return nullptr;
    }
    // host 侧副本 (CPU/GPU 后端统一)
    auto to_host = [&](ggml_tensor * tt, std::vector<float> & out) {
        out.assign(ggml_nelements(tt), 0.0f);
        ggml_backend_tensor_get(tt, out.data(), 0, out.size() * 4);
    };
    to_host(s.text_emb, s.h_text_emb);
    to_host(s.audio_emb, s.h_audio_emb);
    to_host(s.bert_proj, s.h_bert_proj);
    to_host(s.bert_proj_b, s.h_bert_proj_b);
    // [W | b] 增广 (元素 (k,d): k<BERT_DIM 为 W, k==BERT_DIM 为 bias[d]) —— prefill 图内投影用
    {
        std::vector<float> abuf((size_t) s.D * (s.BERT_DIM + 1));
        for (int d = 0; d < s.D; d++) {
            memcpy(abuf.data() + (size_t) d * (s.BERT_DIM + 1), s.h_bert_proj.data() + (size_t) d * s.BERT_DIM,
                   sizeof(float) * s.BERT_DIM);
            abuf[(size_t) d * (s.BERT_DIM + 1) + s.BERT_DIM] = s.h_bert_proj_b[d];
        }
        ggml_backend_tensor_set(s.bert_aug, abuf.data(), 0, abuf.size() * 4);
    }
    {
        float a1 = 0, a2 = 0;
        ggml_backend_tensor_get(t("ar.text_pe_alpha"), &a1, 0, 4);
        ggml_backend_tensor_get(t("ar.audio_pe_alpha"), &a2, 0, 4);
        s.h_text_alpha = a1; s.h_audio_alpha = a2;
    }

    g_prof = getenv("GSV_AR_PROFILE") != nullptr;
    // 默认 f16: KV buffer 222 MiB -> 111 MiB, 且每步 k/v 写入与 attention 读取流量减半。
    // 实测波形不变 (wav cos 0.999917 / max_abs 1.293e-02, AR 1531 -> 1470 ms)。
    // 需要旧的 f32 行为时设 GSV_AR_KV_F32=1 (或 GSV_AR_KV_F16=0)。
    {
        const char * kvf32 = getenv("GSV_AR_KV_F32");
        const char * kvf16 = getenv("GSV_AR_KV_F16");
        s.kv_f16 = !(kvf32 && *kvf32 && strcmp(kvf32, "0") != 0)
                   && !(kvf16 && *kvf16 && strcmp(kvf16, "0") == 0);
    }
    // 融合算子支持探测: 用代表性形状建一次节点问后端 (CPU 恒支持; Vulkan 需 shader 在册)
    {
        ggml_init_params fp = { ggml_tensor_overhead() * 32, NULL, true };
        ggml_context * fctx = ggml_init(fp);
        ggml_tensor * fx = ggml_new_tensor_2d(fctx, GGML_TYPE_F32, 64, 4);
        ggml_tensor * fw = ggml_new_tensor_1d(fctx, GGML_TYPE_F32, 64);
        ggml_tensor * fwb = ggml_new_tensor_1d(fctx, GGML_TYPE_F32, 128);   // [w;b] 打包
        ggml_tensor * fln = ggml_layernorm_affine(fctx, fx, nullptr, nullptr, fwb, 1e-5f);
        ggml_tensor * fac = ggml_add_act(fctx, fx, fw, GGML_ACT_RELU);
        const bool sup = ggml_backend_supports_op(s.backend, fln) && ggml_backend_supports_op(s.backend, fac);
        ggml_free(fctx);
        s.fuse_ln = s.fuse_act = sup;
    }
    if (getenv("GSV_NO_FUSE"))     s.fuse_ln = s.fuse_act = false;
    // 图缓存需要后端支持 set_rows(F32 + i32); 不支持时退回逐步建图
    {
        ggml_init_params gp = { ggml_tensor_overhead() * 32, NULL, true };
        ggml_context * gctx = ggml_init(gp);
        ggml_tensor * ga = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 64, 16);
        ggml_tensor * gb = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 64, 1);
        ggml_tensor * gi = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
        ggml_tensor * gs = ggml_set_rows(gctx, ga, gb, gi);
        // 只在 GPU 上启用: 固定 key 数会让每步多拷 (Lmax-L) 列的 K/V, GPU 上这部分
        // 几乎免费 (带宽富裕) 而建图占 24%; CPU 是带宽受限, 实测反而慢 10~25%。
        const bool is_gpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU ||
                            ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU;
        s.use_graph_cache = is_gpu && ggml_backend_supports_op(s.backend, gs);
        ggml_free(gctx);
    }
    if (getenv("GSV_AR_NO_GRAPHCACHE")) s.use_graph_cache = false;
    if (getenv("GSV_NO_FUSE_LN"))  s.fuse_ln = false;
    if (getenv("GSV_NO_FUSE_ACT")) s.fuse_act = false;
    s.mk_pe();
    if (cfg.verbose)
        printf("[gsv_ar] loaded %s: D=%d head=%d layers=%d vocab=%d EOS=%d bert=%d alpha=(%.4f, %.4f)\n",
               gguf_path.c_str(), s.D, s.NH, s.NL, s.VOCAB, s.EOS, s.BERT_DIM, s.text_pe_alpha, s.audio_pe_alpha);
    if (cfg.verbose) printf("[gsv_ar] fused ops: ln=%s act=%s | decode graph cache: %s | kv: %s\n",
                            s.fuse_ln ? "on" : "off", s.fuse_act ? "on" : "off",
                            s.use_graph_cache ? "on" : "off", s.kv_f16 ? "f16" : "f32");
    return m;
}

void gsv_ar::first_logits(const std::vector<gsv_ar_request> & reqs, std::vector<float> & logits_out) {
    impl & s = *p;
    if (reqs.empty()) return;
    if (s.prompt_len == 0) s.prompt_len = (int) reqs[0].prompt.size();
    std::vector<float> xy, feat, mask;
    int S = 0, max_len = 0;
    s.build_inputs(reqs, xy, feat, mask, S, max_len);
    s.alloc_cache(S + 8, (int) reqs.size());
    s.cache.len = 0;
    s.run_first(xy, feat, mask, S, logits_out);
}

gsv_ar_result gsv_ar::generate(const std::vector<gsv_ar_request> & reqs,
                               const gsv_sampler_cfg & sampler,
                               uint64_t seed_base,
                               int early_stop_num,
                               int max_steps,
                               const std::vector<int32_t> * oracle_tokens,
                               const std::vector<uint64_t> * seeds) {
    impl & s = *p;
    gsv_ar_result res;
    const int B = (int) reqs.size();
    if (B == 0) return res;
    for (const auto & r : reqs)
        if (r.prompt.size() != reqs[0].prompt.size())
            fprintf(stderr, "[gsv_ar] warning: 批内 prompt 长度不一致\n");
    s.prompt_len = (int) reqs[0].prompt.size();

    std::vector<float> xy, feat, mask, lg;
    int S = 0, max_len = 0;
    s.build_inputs(reqs, xy, feat, mask, S, max_len);
    std::vector<int> pad_len(B, 0);
    for (int b = 0; b < B; b++) pad_len[b] = max_len - (int) reqs[b].phones.size();

    s.alloc_cache(S + max_steps + 8, B);
    s.cache.len = 0;
    s.run_first(xy, feat, mask, S, lg);
    prof_reset();

    std::vector<std::vector<int32_t>> y(B);
    for (int b = 0; b < B; b++) y[b] = reqs[b].prompt;
    std::vector<int32_t> idx_list(B, -1);
    res.tokens.assign(B, {});
    std::vector<gsv_rng> rng;
    rng.reserve(B);
    for (int b = 0; b < B; b++) {
        // 批量里每条序列可以有独立 seed（跨请求聚合时保持各自的确定性）
        const uint64_t sd = (seeds && (size_t) b < seeds->size())
                                ? (*seeds)[b]
                                : seed_base + 0x9E3779B9ull * (uint64_t) b;
        rng.emplace_back(sd);
    }
    std::vector<float> prob_dump;   // [steps][V] (seq0), 由 GSV_AR_PROB_DUMP 触发

    gsvprog::begin(2, max_steps);
    for (int idx = 0; idx < max_steps; idx++) {
        gsvprog::tick();
        for (int b = 0; b < B; b++) {
            if (idx_list[b] >= 0) continue;                  // 已完成
            float row[1025];
            memcpy(row, lg.data() + (size_t) b * s.VOCAB, sizeof(float) * s.VOCAB);
            if (idx < 11) row[s.EOS] = -INFINITY;            // 至少 10 token 不停止 (等价 torch [:, :-1])
            int32_t prev[4096];
            const int np = (int) y[b].size() < 4096 ? (int) y[b].size() : 4096;
            for (int i = 0; i < np; i++) prev[i] = y[b][y[b].size() - np + i];
            float probs[1025];
            const double t_s0 = now_ms();
            gsv_logits_to_probs(sampler, row, s.VOCAB, prev, np, probs);
            if (b == 0 && getenv("GSV_AR_PROB_DUMP")) prob_dump.insert(prob_dump.end(), probs, probs + s.VOCAB);
            int tok = gsv_sample(probs, s.VOCAB, rng[b], nullptr);
            g_ms_samp += now_ms() - t_s0;
            if (oracle_tokens != nullptr && idx < (int) oracle_tokens->size()) tok = (*oracle_tokens)[idx];
            y[b].push_back(tok);
            if (tok == s.EOS) {
                idx_list[b] = idx;
                res.tokens[b].assign(y[b].begin(), y[b].end() - 1);
                res.lens.push_back(idx);
            }
        }
        if (early_stop_num != -1 && ((int) y[0].size() - s.prompt_len) > early_stop_num) {
            for (int b = 0; b < B; b++)
                if (idx_list[b] < 0) {
                    idx_list[b] = idx;
                    res.tokens[b].assign(y[b].begin(), y[b].end() - 1);
                    res.lens.push_back(idx);
                }
        }
        bool any_unfinished = false;
        for (int b = 0; b < B; b++) if (idx_list[b] < 0) any_unfinished = true;
        if (!any_unfinished) break;

        std::vector<float> x_in((size_t) s.D * B, 0.0f);
        const double t_e0 = now_ms();
        for (int b = 0; b < B; b++) {
            float e[512];
            s.emb_lookup(y[b].back(), e);
            const int pos = s.prompt_len + idx;
            for (int d = 0; d < s.D; d++)
                x_in[(size_t) d + (size_t) b * s.D] = e[d] + s.h_audio_alpha * s.pe_tab[(size_t) pos * s.D + d];
        }
        g_ms_emb += now_ms() - t_e0;
        if (s.use_graph_cache) s.run_decode_cached(x_in, pad_len, lg);
        else                   s.run_decode(x_in, pad_len, lg);
    }
    for (int b = 0; b < B; b++)
        if (idx_list[b] < 0) {
            res.tokens[b] = y[b];
            res.lens.push_back(max_steps);
        }
    prof_report("decode");
    const char * pd = getenv("GSV_AR_PROB_DUMP");
    if (pd != nullptr && !prob_dump.empty()) {
        FILE * f = fopen(pd, "wb");
        if (f) { fwrite(prob_dump.data(), 4, prob_dump.size(), f); fclose(f);
                 fprintf(stderr, "[gsv_ar] prob dump: %zu steps -> %s", prob_dump.size() / s.VOCAB, pd); }
    }
    return res;
}

void gsv_ar::release_cache() { if (p) p->free_cache(); }
