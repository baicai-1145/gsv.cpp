// AR (T2S) 解码延迟基准 (task-4).
//
// 两种模式:
//  1) 合成输入 (默认): 固定 phones/bert/prompt + 教师强制 (oracle_tokens), 上下文完全确定,
//     q8/f16/q4k 之间只差权重字节与算子 -> 干净的解码延迟口径。
//  2) 真实输入 (--fixture DIR): 读 probe_v2sovits 的 fixture (phones.i32 / bert.f32 /
//     prompt_phones.i32 / prompt_bert.f32 / prompt_codes.i32), 按 gsv_engine::ar_generate
//     的拼接口径 (prompt_phones ++ phones, bert 沿时间轴拼接) 生成, 正常采样并转储 token
//     (--tok-out), 用于跨量化版本的同种子 sample 级一致性验收。
//
// 用法: bench_ar <ar.gguf> [--threads N] [--steps N] [--phones N] [--prompt N]
//                        [--fixture DIR] [--seed N] [--tok-out FILE] [--json]
//   例: GSV_AR_PROFILE=1 ./bench_ar /tmp/gsv-ar-v2-q8.gguf --threads 4 --steps 179
#include "gsv_ar.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

static bool read_bin(const std::string & p, std::vector<float> & v) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    v.resize((size_t) n / 4);
    const size_t rd = fread(v.data(), 4, v.size(), f);
    fclose(f);
    return rd == v.size();
}
static bool read_bin_i32(const std::string & p, std::vector<int32_t> & v) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    v.resize((size_t) n / 4);
    const size_t rd = fread(v.data(), 4, v.size(), f);
    fclose(f);
    return rd == v.size();
}

// 与 gsv_engine::ar_generate 完全相同的输入拼接
static bool build_from_fixture(const std::string & dir, gsv_ar_request & r) {
    std::vector<int32_t> phones, prompt_phones, prompt_codes;
    std::vector<float>   bert, prompt_bert;
    if (!read_bin_i32(dir + "/phones.i32", phones)) return false;
    if (!read_bin_i32(dir + "/prompt_phones.i32", prompt_phones)) return false;
    if (!read_bin_i32(dir + "/prompt_codes.i32", prompt_codes)) return false;
    read_bin(dir + "/bert.f32", bert);
    read_bin(dir + "/prompt_bert.f32", prompt_bert);
    const int Pp = (int) prompt_phones.size(), Pt = (int) phones.size(), P = Pp + Pt;
    r.phones.reserve((size_t) P);
    r.phones.insert(r.phones.end(), prompt_phones.begin(), prompt_phones.end());
    r.phones.insert(r.phones.end(), phones.begin(), phones.end());
    r.bert.assign((size_t) 1024 * P, 0.0f);
    for (int k = 0; k < 1024; k++) {
        if ((int) prompt_bert.size() == 1024 * Pp)
            memcpy(r.bert.data() + (size_t) k * P, prompt_bert.data() + (size_t) k * Pp, sizeof(float) * Pp);
        if ((int) bert.size() == 1024 * Pt)
            memcpy(r.bert.data() + (size_t) k * P + Pp, bert.data() + (size_t) k * Pt, sizeof(float) * Pt);
    }
    r.prompt = prompt_codes;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <ar.gguf> [--threads N] [--steps N] [--phones N] [--prompt N] [--seqs N]"
                        " [--fixture DIR] [--seed N] [--tok-out FILE] [--seqs N] [--json]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    int threads = 4, steps = 179, n_phones = 40, n_prompt = 32, n_seqs = 1;
    long seed = 12345;
    bool json = false;
    std::string fixture, tok_out;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--phones") && i + 1 < argc) n_phones = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) n_prompt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fixture") && i + 1 < argc) fixture = argv[++i];
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atol(argv[++i]);
        else if (!strcmp(argv[i], "--tok-out") && i + 1 < argc) tok_out = argv[++i];
        else if (!strcmp(argv[i], "--seqs") && i + 1 < argc) n_seqs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json")) json = true;
        else { fprintf(stderr, "bad arg %s\n", argv[i]); return 2; }
    }

    gsv_ar_cfg cfg;
    cfg.n_threads = threads;
    cfg.verbose = !json;
    gsv_ar * m = gsv_ar::load(path, cfg);
    if (!m) { fprintf(stderr, "[bench_ar] load failed: %s\n", path.c_str()); return 1; }

    gsv_ar_request r;
    const bool real = !fixture.empty();
    if (real) {
        if (!build_from_fixture(fixture, r)) { fprintf(stderr, "[bench_ar] fixture read failed: %s\n", fixture.c_str()); return 1; }
    } else {
        r.phones.resize(n_phones);
        for (int t = 0; t < n_phones; t++) r.phones[t] = (t * 7 + 3) % 700;
        r.bert.assign((size_t) 1024 * n_phones, 0.0f);
        for (int t = 0; t < n_phones; t++)
            for (int k = 0; k < 1024; k++)
                r.bert[(size_t) k * n_phones + t] = 0.05f * (float) (((k * 13 + t * 7) % 23) - 11);
        r.prompt.resize(n_prompt);
        for (int t = 0; t < n_prompt; t++) r.prompt[t] = (t * 11 + 5) % 700;
    }

    // 合成模式下用教师强制固定上下文; fixture 模式走真实采样 (与 pipeline 同种子可复现)
    const int eos = m->eos();
    std::vector<int32_t> oracle(steps);
    for (int i = 0; i < steps; i++) {
        const int tok = 40 + (i * 17) % 900;
        oracle[i] = (tok == eos) ? 0 : tok;
    }
    const std::vector<int32_t> * orc = real ? nullptr : &oracle;

    gsv_sampler_cfg sampler;   // 默认: top_k=15 / top_p=1 / temp=1 / rep=1.35
    // --seqs N: N 条同形请求一起解码, 用来量批量 AR 的扩展性 (每 token 权重只读一次, 摊给 N 条)
    std::vector<gsv_ar_request> reqs(n_seqs > 0 ? (size_t) n_seqs : 1, r);

    // 预热 (建图/cache 分配/页错误). --warm 0 可关掉, 用于看冷进程首个请求的开销。
    m->generate(reqs, sampler, (uint64_t) seed, -1, 8, orc, nullptr);
    m->release_cache();

    const double t0 = now_ms();
    gsv_ar_result res = m->generate(reqs, sampler, (uint64_t) seed, real ? steps : -1, steps, orc, nullptr);
    const double t1 = now_ms();
    const int prefill_tokens = (int) r.prompt.size();
    const int got = res.tokens.empty() ? 0 : (int) res.tokens[0].size() - prefill_tokens;
    const double total = t1 - t0;
    const double per_tok = total / (double) (got > 0 ? got : steps);
    const int    B      = (int) reqs.size();
    const double agg_per_tok = per_tok / (double) B;   // 摊到每条序列的有效 ms/token
    // KV 是 B 倍分配, 峰值 RSS 决定批量上限
    long rss_kb = 0;
    if (FILE * f = fopen("/proc/self/status", "r")) {
        char line[256];
        while (fgets(line, sizeof(line), f))
            if (!strncmp(line, "VmHWM:", 6)) { rss_kb = atol(line + 6); break; }
        fclose(f);
    }

    if (real && !res.tokens.empty()) {
        const std::vector<int32_t> & all = res.tokens[0];
        const size_t skip = std::min(r.prompt.size(), all.size());
        printf("[bench_ar] fixture: phones=%zu prompt=%zu -> sem=%zu\n",
               r.phones.size(), r.prompt.size(), all.size() - skip);
        if (!tok_out.empty()) {
            FILE * f = fopen(tok_out.c_str(), "wb");
            if (f) { fwrite(all.data() + skip, 4, all.size() - skip, f); fclose(f); }
        }
    }

    if (json) {
        printf("{\"gguf\":\"%s\",\"threads\":%d,\"seqs\":%d,\"tokens\":%d,\"total_ms\":%.1f,"
               "\"ms_per_token\":%.3f,\"tok_per_s\":%.1f,\"agg_ms_per_token\":%.3f,"
               "\"agg_tok_per_s\":%.1f,\"rss_mib\":%.1f}\n",
               path.c_str(), threads, B, got, total, per_tok, 1000.0 / per_tok,
               agg_per_tok, 1000.0 / agg_per_tok, rss_kb / 1024.0);
    } else {
        printf("[bench_ar] %s\n", path.c_str());
        printf("[bench_ar] threads=%d seqs=%d decode_tokens=%d total=%.1f ms  ->  %.3f ms/token  "
               "(%.1f tok/s)  |  聚合 %.3f ms/token/seq (%.1f tok/s)  |  RSS 峰值 %.1f MiB\n",
               threads, B, got, total, per_tok, 1000.0 / per_tok,
               agg_per_tok, 1000.0 / agg_per_tok, rss_kb / 1024.0);
    }
    delete m;
    return 0;
}
