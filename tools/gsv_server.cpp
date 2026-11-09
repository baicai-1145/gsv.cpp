// gsv_server — GPT-SoVITS 兼容 HTTP 服务
//
//   GAG(fork) --HTTP--> gsv_server(C++) --HTTP--> text_sidecar(Python, 仅 G2P)
//                            └── gsv_engine (进程内, 可换 voice)
//
// 核心逻辑在 tools/svc_core.h（voice 注册 / LRU / 合成 / 日志），本文件只做路由与生命周期。
//
// 端点:
//   POST /tts, GET /tts        合成（切句 + 拼接）
//   POST /batch                多条 -> job；GET /jobs/{id}[/result/{i}]；DELETE /jobs/{id}
//   POST /reference            上传参考音频（raw body, ?name=）
//   GET  /models               列出 voices / 当前 voice / reloading
//   GET  /set_gpt_weights /set_sovits_weights?weights_path=<voice>
//   GET  /control?command=exit|restart
//   GET  /health /metrics
//
// 用法:
//   gsv_server [--models models] [--device vulkan|vulkan:N|cpu]
//              [--host 127.0.0.1] [--port 9880] [--sidecar 127.0.0.1:9882]
//              [--log-dir logs] [--threads N] [--verbose]

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "httplib.h"
#include "json.hpp"

#include "ggml-backend.h"
#include "gsv_progress.h"

#define GSV_VERSION "0.1.2"

#include "svc_core.h"

using namespace svc;

namespace {

constexpr int kMaxWaiters = 32;          // 超过就 503（背压）
constexpr double kQueueTimeoutSec = 300; // 单请求排队上限

// ---------------- crash dump (M1/M5)
#if defined(_WIN32)
static LONG WINAPI on_crash(EXCEPTION_POINTERS * ep) {
    char path[MAX_PATH];
    SYSTEMTIME st;
    GetLocalTime(&st);
    std::snprintf(path, sizeof(path), "logs\\crash-%04d%02d%02d-%02d%02d%02d-%08lx.dmp",
                  st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond,
                  (unsigned long) ep->ExceptionRecord->ExceptionCode);
    std::fprintf(stderr, "[gsv_server] CRASH code=0x%08lx addr=%p -> %s\n",
                 (unsigned long) ep->ExceptionRecord->ExceptionCode,
                 ep->ExceptionRecord->ExceptionAddress, path);
    CreateDirectoryA("logs", nullptr);
    HANDLE h = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h != INVALID_HANDLE_VALUE) {
        MINIDUMP_EXCEPTION_INFORMATION mi{GetCurrentThreadId(), ep, FALSE};
        MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), h, MiniDumpWithThreadInfo,
                          &mi, nullptr, nullptr);
        CloseHandle(h);
    }
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

// 安全相关（g_token / g_allow / path_allowed / authorized）在 svc_core.h

struct TtsOutcome {
    int code = 200;
    std::string error_code;   // bad_request / busy / timeout / engine / sidecar / internal
    std::string message;
    std::string wav;
    json meta;
};

json jerror(int code, const std::string & ec, const std::string & msg) {
    return json{{"message", msg}, {"code", ec}};
}

// OpenAPI 3 description served at /docs and /openapi.json
std::string openapi_json() {
    json s = json::object();
    s["type"] = "string";

    json tts_req = json::object();
    tts_req["type"] = "object";
    tts_req["required"] = json::array({"text", "ref_audio_path", "prompt_text"});
    json props = json::object();
    auto P = [&](const char * name, const char * type, const char * desc) {
        props[name] = json{{"type", type}, {"description", desc}};
    };
    P("text", "string", "text to synthesize");
    P("text_lang", "string", "zh / en / ja / ko / yue (default zh)");
    P("ref_audio_path", "string", "reference wav (must be under --allow-path)");
    P("prompt_text", "string", "text of the reference audio");
    P("prompt_lang", "string", "language of prompt_text (default zh)");
    P("voice", "string", "registered voice name; empty = current voice");
    P("sample_steps", "integer", "DiT steps (default: per voice)");
    P("cfg_rate", "number", "CFG rate");
    P("seed", "integer", "-1 = random");
    P("top_k", "integer", "AR sampler top-k");
    P("top_p", "number", "AR sampler top-p");
    P("temperature", "number", "AR sampler temperature");
    P("repetition_penalty", "number", "AR repetition penalty");
    P("speed_factor", "number", "duration scale, (0, 4]");
    P("text_split_method", "string", "cut0..cut5 (default cut5)");
    P("fragment_interval", "number", "silence between segments, seconds");
    P("dbcache", "string", "cache-dit: off / auto / f4 / f8 / f16");
    P("dit_uncond_reuse", "integer", "DiT alg: 0 off / 1 reuse neg on odd steps / 2 linear extrapolation");
    P("timeout_ms", "integer", "override engine lock / scheduler / AR wait timeout in ms (<=0 = server default)");
    P("stream", "boolean", "true = chunked audio/wav, PCM is flushed per text segment");
    P("media_type", "string", "only wav is supported");
    tts_req["properties"] = props;
    s["TtsRequest"] = tts_req;

    json paths = json::object();
    paths["/tts"]["post"] = json{{"summary", "synthesize (json body)"},
                                 {"requestBody", json{{"required", true},
                                                      {"content", json{{"application/json",
                                                                        json{{"schema", json{{"$ref", "#/components/schemas/TtsRequest"}}}}}}}}},
                                 {"responses", json{{"200", json{{"description", "audio/wav (chunked when stream=true)"}}},
                                                    {"400", json{{"description", "bad_request"}}},
                                                    {"403", json{{"description", "forbidden (path not allowed)"}}},
                                                    {"503", json{{"description", "busy / sidecar down"}}},
                                                    {"504", json{{"description", "timeout"}}}}}};
    paths["/tts"]["get"] = json{{"summary", "synthesize (query parameters, same fields as the json body)"}};
    paths["/batch"]["post"] = json{{"summary", "asynchronous batch"},
                                   {"description", "body: {texts:[...], ...TtsRequest}. Returns {job_id,total}."}};
    paths["/jobs/{id}"]["get"] = json{{"summary", "job state"}, {"parameters", json::array({json{{"name", "id"}, {"in", "path"}, {"required", true}, {"schema", json{{"type", "string"}}}}})}};
    paths["/jobs/{id}/result/{i}"]["get"] = json{{"summary", "job item wav"}};
    paths["/jobs/{id}"]["delete"] = json{{"summary", "cancel / drop job"}};
    paths["/models"]["get"] = json{{"summary", "list voices and current voice"}};
    paths["/change_refer"]["get"] = json{{"summary", "set the default reference audio"},
                                         {"parameters", json::array({json{{"name", "refer_wav_path"}, {"in", "query"}, {"required", true}, {"schema", json{{"type", "string"}}}},
                                                                     json{{"name", "prompt_text"}, {"in", "query"}, {"required", true}, {"schema", json{{"type", "string"}}}},
                                                                     json{{"name", "prompt_lang"}, {"in", "query"}, {"schema", json{{"type", "string"}}}}})}};
    paths["/change_refer"]["post"] = json{{"summary", "same as GET, json body accepted"}};
    paths["/set_refer_audio"]["get"] = json{{"summary", "alias of /change_refer"}};
    paths["/health"]["get"] = json{{"summary", "readiness"}};
    paths["/metrics"]["get"] = json{{"summary", "counters"}};
    paths["/control"]["get"] = json{{"summary", "command=exit to stop the server"}};
    paths["/asr"]["get"] = json{{"summary", "always 501: ASR is not implemented in this engine"}};

    json doc = json{{"openapi", "3.0.3"},
                    {"info", json{{"title", "gsv.cpp TTS service"}, {"version", GSV_VERSION},
                                  {"description", "api_v2-compatible subset. POST /tts with stream=true returns a chunked WAV whose PCM is flushed per text segment."}}},
                    {"paths", paths},
                    {"components", json{{"schemas", s}}}};
    return doc.dump(2);
}

// JSON 取值：兼容 GET 的字符串形式和 POST 的数字形式
std::string jstr(const json & b, const char * k, const std::string & def) {
    if (!b.contains(k) || b[k].is_null()) return def;
    try {
        if (b[k].is_string()) return b[k].get<std::string>();
        return b[k].dump();
    } catch (...) { return def; }
}

double jnum(const json & b, const char * k, double def) {
    if (!b.contains(k) || b[k].is_null()) return def;
    try {
        if (b[k].is_number()) return b[k].get<double>();
        if (b[k].is_string()) return std::stod(b[k].get<std::string>());
    } catch (...) {}
    return def;
}

bool jbool(const json & b, const char * k, bool def) {
    if (!b.contains(k) || b[k].is_null()) return def;
    try {
        if (b[k].is_boolean()) return b[k].get<bool>();
        if (b[k].is_number()) return b[k].get<double>() != 0;
        if (b[k].is_string()) {
            const std::string s = b[k].get<std::string>();
            return s == "true" || s == "1" || s == "True";
        }
    } catch (...) {}
    return def;
}

void parse_params(const json & b, SynthParams & p) {
    p.text = jstr(b, "text", "");
    p.text_lang = jstr(b, "text_lang", "zh");
    p.ref_path = jstr(b, "ref_audio_path", "");
    p.prompt_text = jstr(b, "prompt_text", "");
    p.prompt_lang = jstr(b, "prompt_lang", "zh");
    p.voice = jstr(b, "voice", "");
    p.steps = (int) jnum(b, "sample_steps", -1);
    p.cfg = (float) jnum(b, "cfg_rate", -1.0);
    // cache-dit：dbcache=off|auto|f4|f8|f16（auto = 步数>=16 时 F8/.12）；也可显式给四个参数
    {
        const std::string db = jstr(b, "dbcache", "auto");
        if (db == "off" || db == "0") p.dbcache_fn = 0;
        else if (db == "on") p.dbcache_fn = 8;
        else if (db.size() > 1 && (db[0] == 'f' || db[0] == 'F'))
            p.dbcache_fn = std::atoi(db.c_str() + 1);
        p.dbcache_fn = (int) jnum(b, "dbcache_fn", p.dbcache_fn);
        p.dbcache_thr = (float) jnum(b, "dbcache_threshold", p.dbcache_thr);
        p.dbcache_warmup = (int) jnum(b, "dbcache_warmup", p.dbcache_warmup);
        p.dbcache_ds = (int) jnum(b, "dbcache_downsample", p.dbcache_ds);
    }
    // DiT 算法级加速 (见 docs/usage.md "DiT 算法开关")
    p.dit_uncond_reuse = (int) jnum(b, "dit_uncond_reuse", p.dit_uncond_reuse);
    // 超时覆盖 (毫秒; <=0 = 服务端默认)
    p.timeout_ms = (int) jnum(b, "timeout_ms", p.timeout_ms);
    p.seed = (int) jnum(b, "seed", -1);
    p.max_steps = (int) jnum(b, "max_steps", 1500);
    p.early_stop = (int) jnum(b, "early_stop", 400);
    p.speed = jnum(b, "speed_factor", 1.0);
    p.skip_steps = (int) jnum(b, "skip_first_step", 0);
    p.split_method = jstr(b, "text_split_method", "cut5");
    p.seg_chars = (int) jnum(b, "seg_chars", 0);   // 0 = 旧行为; ~60 = 吞吐优先(段数最少)
    p.fragment_interval = (float) jnum(b, "fragment_interval", 0.3);
    p.sampler.top_k = (int) jnum(b, "top_k", p.sampler.top_k);
    p.sampler.top_p = (float) jnum(b, "top_p", p.sampler.top_p);
    p.sampler.temperature = (float) jnum(b, "temperature", p.sampler.temperature);
    p.sampler.repetition_penalty = (float) jnum(b, "repetition_penalty", p.sampler.repetition_penalty);
}

// 底层合成：切 voice（如需）-> 快照引擎 -> 排队/计时 -> synth_text
bool perform_synth(State & st, const SidecarCfg & side, const SynthParams & p, std::vector<float> & wav,
                   SynthTiming & tm, std::string & err, std::string & ec) {
    if (const char * dbg = std::getenv("GSV_DEBUG_SYNTH")) {
        std::fprintf(stderr,
                     "[perform_synth] steps=%d cfg=%.3f seed=%d max=%d early=%d speed=%.2f "
                     "voice='%s' ref='%s' prompt=%zu flen=%zu split=%s\n",
                     p.steps, p.cfg, p.seed, p.max_steps, p.early_stop, p.speed, p.voice.c_str(),
                     p.ref_path.c_str(), p.prompt_text.size(), p.text.size(), p.split_method.c_str());
        // cfg_rate defaults to -1 ("auto"): synth_text resolves it per voice version
        // (v5dev 1.30 / v5turbo 0). Say so, otherwise the log reads like CFG is disabled.
        if (p.cfg < 0.0f)
            std::fprintf(stderr, "[perform_synth] cfg_rate not given -> auto (v5dev 1.30 / v5turbo 0)\n");
    }
    if (!p.voice.empty() && p.voice != st.voice) {
        double ms = 0;
        if (!switch_voice(st, p.voice, err, ms)) { ec = "engine"; return false; }
    }
    if (!st.mock_engine && !snapshot(st)) { err = "engine not ready"; ec = "engine"; return false; }

    if (st.waiters.fetch_add(1) >= kMaxWaiters) {
        st.waiters.fetch_sub(1);
        st.n_busy++;
        err = "server busy (too many waiting requests)";
        ec = "busy";
        return false;
    }
    // AR 走跨请求聚合队列；DiT/vocoder 在函数内部各自取短锁，不再全程占 compute_mu
    SynthParams pp = p;
    resolve_auto_params(st, pp);
    // 请求级聚合：提交给调度器，与同时到达的 /tts、/batch 文本一起分组
    if (st.sched) {
        auto fut = st.sched->submit(pp, pp.text);
        const int wait_ms = pp.timeout_ms > 0 ? pp.timeout_ms : st.lock_timeout_ms;
        if (fut.wait_for(std::chrono::milliseconds(wait_ms)) != std::future_status::ready) {
            st.waiters.fetch_sub(1);
            st.n_timeouts++;
            err = "scheduler timeout";
            ec = "timeout";
            return false;
        }
        SynthResult r = fut.get();
        st.waiters.fetch_sub(1);
        if (!r.err.empty()) { err = r.err; ec = "engine"; return false; }
        wav = std::move(r.wav);
        tm = r.tm;
        return true;
    }
    const bool ok = synth_text_q(st, side, pp, wav, tm, err);
    st.waiters.fetch_sub(1);
    if (!ok && ec.empty()) ec = "engine";
    return ok;
}

// 请求参数校验（/tts 与流式共用）：返回非 0 code 时直接作为错误响应。
struct ParamError {
    int code = 0;
    std::string ec, msg;
    bool ok() const { return code == 0; }
};

ParamError validate_tts_params(const SynthParams & p, const std::string & media_type) {
    ParamError e;
    constexpr size_t kMaxText = 4000;      // ~ utf8 bytes; longer requests should be split
    auto fail = [&](int code, const char * ec, const char * msg) {
        e.code = code;
        e.ec = ec;
        e.msg = msg;
        return e;
    };
    if (p.text.empty()) return fail(400, "bad_request", "text is required");
    if (p.text.size() > kMaxText) return fail(400, "bad_request", "text too long (> 4000 bytes); use /batch or split it");
    if (p.ref_path.empty()) return fail(400, "bad_request", "ref_audio_path is required");
    if (p.prompt_text.empty())
        return fail(400, "bad_request",
                    "prompt_text is required (V3/V4/V5 do not support reference-text-free mode)");
    if (p.prompt_text.size() > kMaxText) return fail(400, "bad_request", "prompt_text too long (> 4000 bytes)");
    if (media_type != "wav") return fail(400, "bad_request", "only media_type=wav is supported");
    if (p.speed <= 0.0 || p.speed > 4.0) return fail(400, "bad_request", "speed_factor must be in (0, 4]");
    if (p.steps != -1 && (p.steps < 1 || p.steps > 1000))
        return fail(400, "bad_request", "sample_steps must be 1..1000 (or -1 for the voice default)");
    if (p.cfg >= 0.0 && p.cfg > 20.0) return fail(400, "bad_request", "cfg_rate must be <= 20");
    if (p.fragment_interval < 0.0f || p.fragment_interval > 5.0f)
        return fail(400, "bad_request", "fragment_interval must be in [0, 5]");
    if (!path_allowed(p.ref_path))
        return fail(403, "forbidden",
                    "ref_audio_path is not under an allowed directory (see --allow-path)");
    {
        std::error_code ec;
        if (!fs::exists(gsvpath::from_utf8(p.ref_path), ec) || ec)
            return fail(400, "bad_request", "ref_audio_path does not exist");
    }
    if (const double dur = wav_duration_seconds(p.ref_path); dur > 30.0)
        return fail(400, "bad_request",
                    "reference audio too long (> 30 s); trim it to 3-10 s");
    return e;
}

TtsOutcome run_tts(State & st, const SidecarCfg & side, ReqLog & rlog, const json & body,
                   const std::string & rid) {
    TtsOutcome o;
    SynthParams p;
    parse_params(body, p);
    // 请求没带参考时回退到 /change_refer 设置的默认参考
    {
        std::lock_guard<std::mutex> lk(st.ref_mu);
        if (p.ref_path.empty()) p.ref_path = st.default_ref_path;
        if (p.prompt_text.empty()) p.prompt_text = st.default_prompt_text;
        if (!body.contains("prompt_lang") && !st.default_prompt_lang.empty())
            p.prompt_lang = st.default_prompt_lang;
    }
    const std::string media_type = jstr(body, "media_type", "wav");
    const double t0 = now_s();

    const ParamError pe = validate_tts_params(p, media_type);
    if (!pe.ok()) {
        o.code = pe.code;
        o.error_code = pe.ec;
        o.message = pe.msg;
        return o;
    }

    st.n_tts++;
    std::vector<float> wav;
    SynthTiming tm;
    std::string err, ec;
    bool ok = false;
    try {
        ok = perform_synth(st, side, p, wav, tm, err, ec);
    } catch (const std::bad_alloc &) {
        ec = "busy";
        err = "out of memory while synthesizing";
    } catch (const std::exception & e) {
        ec = "internal";
        err = std::string("exception: ") + e.what();
    }
    const double total_ms = (now_s() - t0) * 1000;

    // 生效的 cfg/steps: cfg_rate 不传时按 voice 版本自动解析 (v5dev 1.30 / 其它 0)。
    // cfg=0 时 CFG 支路根本不跑, 打出来才能一眼看出这次到底有没有 CFG。
    SynthParams eff = p;
    resolve_auto_params(st, eff);

    if (!ok) {
        st.n_errors++;
        o.code = (ec == "busy") ? 503 : (ec == "timeout") ? 504 : (ec == "sidecar") ? 503 : 500;
        o.error_code = ec.empty() ? "internal" : ec;
        o.message = err;
    } else {
        normalize(wav);
        o.wav = wav_bytes(wav);
        st.last_total_ms = total_ms;
        o.meta = json{{"audio_s", (double) wav.size() / kSampleRate},
                      {"segments", tm.segments},
                      {"phones", tm.phones},
                      {"ms", total_ms},
                      {"cfg_rate", eff.cfg},
                      {"sample_steps", eff.steps},
                      {"voice_version", st.version}};
        std::printf("[tts %s] voice=%s(%s) segs=%d phones=%d audio=%.2fs cfg=%.2f steps=%d | "
                    "g2p %.0f prep %.0f bert %.0f ar %.0f cond %.0f dit %.0f voc %.0f synth %.0f | "
                    "cache %d/%d | reuse ref=%d text=%d fea=%d/%d | total %.0f ms\n",
                    rid.c_str(), st.voice.c_str(), st.version.c_str(), tm.segments, tm.phones,
                    (double) wav.size() / kSampleRate, eff.cfg, eff.steps,
                    tm.g2p_ms, tm.prep_ms, tm.bert_ms, tm.ar_ms,
                    tm.cond_ms, tm.dit_ms, tm.voc_ms, tm.synth_ms, tm.cache_hits,
                    tm.cache_hits + tm.cache_misses, tm.ref_hits, tm.text_hits, tm.fea_hits,
                    tm.fea_hits + tm.fea_misses, total_ms);
    }

    rlog.write(json{{"ts", now_s()},
                    {"id", rid},
                    {"kind", "tts"},
                    {"voice", st.voice},
                    {"text_chars", (int) utf8_len(p.text)},
                    {"status", o.code},
                    {"error", o.message},
                    {"total_ms", total_ms},
                    {"g2p_ms", tm.g2p_ms},
                    {"prep_ms", tm.prep_ms},
                    {"bert_ms", tm.bert_ms},
                    {"ar_ms", tm.ar_ms},
                    {"synth_ms", tm.synth_ms},
                    {"segments", tm.segments},
                    {"phones", tm.phones},
                    {"reuse_ref", tm.ref_hits},
                    {"reuse_text", tm.text_hits},
                    {"reuse_fea", tm.fea_hits},
                    {"cfg_rate", eff.cfg},
                    {"sample_steps", eff.steps}});
    return o;
}

}  // namespace

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
#if defined(_WIN32)
    // 控制台默认是 GBK；源码里的中文是 UTF-8，不切码页会显示成"鏈◆鏊"之类的乱码
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    SetUnhandledExceptionFilter(on_crash);
#endif
    std::set_terminate([]() {
        try {
            if (std::current_exception()) std::rethrow_exception(std::current_exception());
        } catch (const std::exception & e) {
            std::fprintf(stderr, "[gsv_server] std::terminate: %s\n", e.what());
        } catch (...) {
            std::fprintf(stderr, "[gsv_server] std::terminate: unknown\n");
        }
        std::abort();
    });

    std::string host = "127.0.0.1";
    int port = 9880;
    SidecarCfg side;
    std::string log_dir = "logs";
    bool verbose = false;
    int parent_pid = 0;   // exit when this process disappears (GUI closed / crashed)
    bool list_devices = false;
    double job_ttl_arg = 3600.0;
    int lock_timeout_ms = 300000;   // engine 锁/调度器/AR 等待的全局默认 (请求可用 timeout_ms 覆盖)
    State st;
    st.base.model_dir = "models";

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--models") st.base.model_dir = next("--models");
        else if (a == "--device") st.base.device = next("--device");
        else if (a == "--threads") st.base.n_threads = std::atoi(next("--threads").c_str());
        else if (a == "--host") host = next("--host");
        else if (a == "--port") port = std::atoi(next("--port").c_str());
        else if (a == "--sidecar") { if (!parse_host_port(next("--sidecar"), side)) {
            std::fprintf(stderr, "bad --sidecar\n"); return 2; } }
        else if (a == "--log-dir") log_dir = next("--log-dir");
        else if (a == "--token") g_token = next("--token");
        else if (a == "--allow-path") g_allow.push_back(next("--allow-path"));
        else if (a == "--job-ttl") job_ttl_arg = std::atof(next("--job-ttl").c_str());
        else if (a == "--lock-timeout-ms") lock_timeout_ms = std::atoi(next("--lock-timeout-ms").c_str());
        else if (a == "--parent-pid") parent_pid = std::atoi(next("--parent-pid").c_str());
        else if (a == "--list-devices") list_devices = true;
        else if (a == "--version") { std::printf("gsv_server %s\n", GSV_VERSION); return 0; }
        else if (a == "--mock-engine") st.mock_engine = true;
        else if (a == "--mock-sidecar") g_mock_sidecar = true;
        else if (a == "--verbose") verbose = true;
        else if (a == "-h" || a == "--help") {
            std::printf("usage: gsv_server [--models DIR] [--device cpu|vulkan|vulkan:N|N]\n"
                        "                  [--host 127.0.0.1] [--port 9880] [--sidecar 127.0.0.1:9882]\n"
                        "                  [--allow-path DIR]...  ('*' = allow every directory)\n"
                        "                  [--log-dir logs] [--threads N] [--verbose] [--list-devices] [--version]\n"
                        "                  [--lock-timeout-ms N]  (engine lock / scheduler wait; per-request\n"
                        "                                            override via the timeout_ms field)\n");
            return 0;
        } else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    st.lock_timeout_ms = lock_timeout_ms > 0 ? lock_timeout_ms : 300000;
    // Watchdog: the GUI passes --parent-pid. If it disappears (normal close, crash or a force
    // kill that bypassed the job object), exit so the GPU/VRAM is released.
    if (parent_pid > 0) {
        std::thread([parent_pid]() {
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
#ifdef _WIN32
                HANDLE h = OpenProcess(SYNCHRONIZE, FALSE, (DWORD) parent_pid);
                if (!h) {
                    std::printf("[gsv_server] parent %d is gone, exiting\n", parent_pid);
                    std::_Exit(0);
                }
                const DWORD w = WaitForSingleObject(h, 0);
                CloseHandle(h);
                if (w == WAIT_OBJECT_0) {
                    std::printf("[gsv_server] parent %d exited, shutting down\n", parent_pid);
                    std::_Exit(0);
                }
#else
                if (kill(parent_pid, 0) != 0) {
                    std::printf("[gsv_server] parent %d is gone, exiting\n", parent_pid);
                    std::_Exit(0);
                }
#endif
            }
        }).detach();
    }
    // CPU thread default: physical cores when SMT is present (hyperthreads add spin contention
    // in ggml's barrier); --threads overrides it.
    if (st.base.n_threads <= 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        int nt = (int) (hw == 0 ? 4 : (hw > 8 ? hw / 2 : hw));
        if (nt < 1) nt = 1;
        st.base.n_threads = nt;
        if (verbose)
            std::printf("[gsv_server] --threads not set: using %d (hardware_concurrency=%u)\n", nt, hw);
    }
    if (list_devices) {
        ggml_backend_load_all();
        const int n = (int) ggml_backend_dev_count();
        std::printf("ggml devices (%d):\n", n);
        int gpu = 0;
        for (int i = 0; i < n; i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            const auto ty = ggml_backend_dev_type(d);
            size_t fr = 0, tot = 0;
            ggml_backend_dev_memory(d, &fr, &tot);
            const char * t = ty == GGML_BACKEND_DEVICE_TYPE_GPU ? "GPU"
                           : ty == GGML_BACKEND_DEVICE_TYPE_IGPU ? "iGPU" : "CPU";
            if (ty == GGML_BACKEND_DEVICE_TYPE_GPU || ty == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                std::printf("  #%d  %-4s  %-10s  %s  (%.1f/%.1f GiB free)\n", gpu, t,
                            ggml_backend_dev_name(d), ggml_backend_dev_description(d),
                            (double) fr / (1024.0 * 1024.0 * 1024.0),
                            (double) tot / (1024.0 * 1024.0 * 1024.0));
                gpu++;
            } else {
                std::printf("  --   %-4s  %-10s  %s\n", t, ggml_backend_dev_name(d),
                            ggml_backend_dev_description(d));
            }
        }
        std::printf("use --device vulkan:N (or --device N) for GPU #N, --device cpu for CPU\n");
        return 0;
    }

    if (port <= 0 || port > 65535) { std::fprintf(stderr, "bad --port\n"); return 2; }
    if (!is_loopback(host) && g_token.empty()) {
        std::fprintf(stderr,
                     "[gsv_server] refusing to bind %s without --token "
                     "(non-loopback access requires auth)\n",
                     host.c_str());
        return 2;
    }
    if (g_allow.empty()) {
        g_allow.push_back(fs::current_path().string());
        g_allow.push_back((fs::temp_directory_path() / "gsv_refs").string());
    }
    st.base.verbose = verbose;

    ReqLog rlog;
    rlog.init(log_dir);

    // --models 常被传成相对路径 (GAG 默认就是 "runtime/models")。统一解析成绝对路径：
    // 否则 voice manifest 里的相对文件名会再被 gsv_engine::path() 补一次 model_dir,
    // 拼成 ".../voices/v5dev/runtime/models/voices/v5dev/../../gsv-ar-q8.gguf" 这种不存在的路径，
    // voice 加载几毫秒就失败 (只有 default 正常 —— 它的文件名是空的)。
    st.base.model_dir = VoiceRegistry::abs_path(st.base.model_dir);
    st.voices.init(st.base.model_dir);
    if (st.mock_engine) {
        std::printf("[gsv_server] mock engine mode (no models/GPU); sidecar mock=%d\n",
                    (int) g_mock_sidecar);
    } else {
        std::string err;
        double ms = 0;
        std::printf("[gsv_server] loading default voice from %s (device=%s)...\n",
                    st.base.model_dir.c_str(), st.base.device.empty() ? "cpu" : st.base.device.c_str());
        if (!switch_voice(st, "default", err, ms)) {
            std::fprintf(stderr, "[gsv_server] %s\n", err.c_str());
            return 1;
        }
        std::printf("[gsv_server] engine ready (%s, %.0f ms)\n", st.voice.c_str(), ms);
    }
    // 跨请求聚合：
    //  - SynthScheduler：请求级（/tts 与 /batch 的每个文本），同参考/参数自动攒组；
    //  - ArBatcher：组内各 segment 的 AR 再攒批。
    ArBatcher ar_batcher;
    ar_batcher.start(&st);
    st.ar_batcher = &ar_batcher;
    SynthScheduler sched;
    sched.start(&st, side);
    st.sched = &sched;

    if (!g_mock_sidecar && !sidecar_alive(side)) {
        std::fprintf(stderr, "[gsv_server] warning: text sidecar not reachable at %s:%d\n"
                             "             start it with: python tools/text_sidecar.py --port %d\n",
                     side.host.c_str(), side.port, side.port);
    } else if (!g_mock_sidecar) {
        std::printf("[gsv_server] text sidecar ok (%s:%d)\n", side.host.c_str(), side.port);
    }

    // ---- job queue ----
    struct JobItem {
        std::string text;
        std::string wav;
        double audio_s = 0;
        double ms = 0;
        std::string error;
    };
    struct Job {
        std::string id;
        SynthParams base;
        std::vector<JobItem> items;
        int state = 0;   // 0 queued 1 running 2 done 3 canceled 4 failed
        std::string fatal;
        bool cancel = false;
        double created = now_s();
        std::mutex m;
    };
    std::deque<std::shared_ptr<Job>> queue;
    std::mutex qm;
    std::condition_variable qcv;
    std::mutex jobs_m;
    std::map<std::string, std::shared_ptr<Job>> jobs;
    std::atomic<bool> stopping{false};
    std::atomic<uint64_t> job_seq{0};

    auto gen_job_id = [&]() {
        char buf[48];
        std::snprintf(buf, sizeof(buf), "job-%lld-%llu", (long long) (now_s() * 1000),
                      (unsigned long long) job_seq++);
        return std::string(buf);
    };

    // ---- job 持久化 / TTL ----
    double job_ttl = job_ttl_arg;
    const fs::path jobs_root = fs::path(log_dir) / "jobs";
    { std::error_code ec; fs::create_directories(jobs_root, ec); }
    auto job_dir = [&](const std::string & id) { return jobs_root / id; };
    auto state_name = [](int s) {
        static const char * n[] = {"queued", "running", "done", "canceled", "failed"};
        return std::string(n[std::max(0, std::min(4, s))]);
    };
    auto persist_job = [&](const std::shared_ptr<Job> & job) {
        std::error_code ec;
        const fs::path d = job_dir(job->id);
        fs::create_directories(d, ec);
        json items = json::array();
        std::lock_guard<std::mutex> jl(job->m);
        for (size_t i = 0; i < job->items.size(); i++) {
            const auto & it = job->items[i];
            const bool has = !it.wav.empty();
            if (has) {
                std::ofstream f(d / (std::to_string(i) + ".wav"), std::ios::binary);
                if (f) f.write(it.wav.data(), (std::streamsize) it.wav.size());
            }
            items.push_back(json{{"text", it.text}, {"audio_s", it.audio_s}, {"ms", it.ms},
                                 {"error", it.error}, {"done", has}});
        }
        json meta{{"job_id", job->id},
                  {"state", state_name(job->state)},
                  {"fatal", job->fatal},
                  {"created", job->created},
                  {"items", items}};
        std::ofstream f(d / "meta.json");
        if (f) f << meta.dump();
    };

    std::thread worker([&]() {
        try {
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lk(qm);
                qcv.wait(lk, [&] { return !queue.empty() || stopping.load(); });
                if (stopping.load()) return;
                job = queue.front();
                queue.pop_front();
            }
            {
                std::lock_guard<std::mutex> jl(job->m);
                job->state = 1;
            }
            // 批量 AR：把每条文本按切句方式拆成 segment，所有 segment 一起做 AR
            //（每 8 条一批），再逐条 synthesize，最后按 item 拼回（cut0/cut5 都适用）
            {
                std::vector<std::string> texts;
                texts.reserve(job->items.size());
                for (const auto & it : job->items) texts.push_back(it.text);
                std::vector<std::vector<float>> wavs;
                std::vector<SynthTiming> tms;
                std::vector<std::string> errs;
                std::string err, ec;
                const double t0 = now_s();
                bool ok = false;
                try {
                    ok = synth_batch_texts_q(st, side, job->base, texts, wavs, tms, errs, err, ec);
                } catch (const std::exception & e) {
                    err = std::string("exception: ") + e.what();
                    ec = "engine";
                }
                {
                    std::lock_guard<std::mutex> jl(job->m);
                    for (size_t i = 0; i < job->items.size(); i++) {
                        if (!ok) { job->items[i].error = err; continue; }
                        if (wavs[i].empty()) {
                            job->items[i].error = errs[i];
                            continue;
                        }
                        normalize(wavs[i]);
                        job->items[i].wav = wav_bytes(wavs[i]);
                        job->items[i].audio_s = (double) wavs[i].size() / kSampleRate;
                        job->items[i].ms = (now_s() - t0) * 1000;
                    }
                    if (job->state == 1) job->state = 2;
                }
                std::printf("[%s] ar-batch %zu items in %.0f ms\n", job->id.c_str(), texts.size(),
                            (now_s() - t0) * 1000);
                persist_job(job);
                continue;
            }
            for (size_t i = 0; i < job->items.size(); i++) {
                {
                    std::lock_guard<std::mutex> jl(job->m);
                    if (job->cancel) { job->state = 3; break; }
                }
                SynthParams p = job->base;
                p.text = job->items[i].text;
                std::vector<float> wav;
                SynthTiming tm;
                std::string err, ec;
                const double t0 = now_s();
                bool ok = false;
                try {
                    ok = perform_synth(st, side, p, wav, tm, err, ec);
                } catch (const std::exception & e) {
                    err = std::string("exception: ") + e.what();
                    ec = "engine";
                }
                std::lock_guard<std::mutex> jl(job->m);
                if (!ok) {
                    job->items[i].error = err;
                    job->items[i].ms = (now_s() - t0) * 1000;
                } else {
                    normalize(wav);
                    job->items[i].wav = wav_bytes(wav);
                    job->items[i].audio_s = (double) wav.size() / kSampleRate;
                    job->items[i].ms = (now_s() - t0) * 1000;
                }
                std::printf("[%s] %zu/%zu %s %.2fs (%.0f ms)%s\n", job->id.c_str(), i + 1,
                            job->items.size(),
                            job->items[i].text.substr(0, 16).c_str(), job->items[i].audio_s,
                            job->items[i].ms, job->items[i].error.empty() ? "" : " ERR");
            }
            {
                std::lock_guard<std::mutex> jl(job->m);
                if (job->state == 1) job->state = 2;
            }
            persist_job(job);
        }
        } catch (const std::exception & e) {
            std::fprintf(stderr, "[gsv_server] worker exception: %s\n", e.what());
        } catch (...) {
            std::fprintf(stderr, "[gsv_server] worker exception: unknown\n");
        }
    });

    // TTL 清理：终态 job 超过 --job-ttl 后从内存和磁盘删除
    std::thread janitor([&]() {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            if (stopping.load()) return;
            std::vector<std::string> dead;
            {
                std::lock_guard<std::mutex> jl(jobs_m);
                for (auto it = jobs.begin(); it != jobs.end();) {
                    bool terminal = false;
                    double created = 0;
                    {
                        std::lock_guard<std::mutex> jj(it->second->m);
                        terminal = it->second->state >= 2;
                        created = it->second->created;
                    }
                    if (terminal && now_s() - created > job_ttl) {
                        dead.push_back(it->first);
                        it = jobs.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
            for (const auto & id : dead) {
                std::error_code ec;
                fs::remove_all(job_dir(id), ec);
            }
        }
    });

    // ---- HTTP ----
    httplib::Server svr;
    svr.set_pre_routing_handler([&](const httplib::Request & req, httplib::Response & res) {
        if (req.path == "/health") return httplib::Server::HandlerResponse::Unhandled;
        if (authorized(req)) return httplib::Server::HandlerResponse::Unhandled;
        res.status = 401;
        res.set_content(jerror(401, "unauthorized", "missing or invalid token").dump(),
                        "application/json");
        return httplib::Server::HandlerResponse::Handled;
    });
    svr.set_payload_max_length(16 * 1024 * 1024);
    svr.set_logger([](const httplib::Request & req, const httplib::Response & res) {
        std::printf("[http] %s %s -> %d\n", req.method.c_str(), req.path.c_str(), res.status);
    });
    svr.set_exception_handler([](const httplib::Request &, httplib::Response & res, std::exception_ptr ep) {
        std::string msg = "internal error";
        try { std::rethrow_exception(ep); }
        catch (const std::exception & e) { msg = e.what(); }
        catch (...) {}
        res.status = 500;
        res.set_content(jerror(500, "internal", msg).dump(), "application/json");
    });

    auto jerr = [](httplib::Response & res, int code, const std::string & ec, const std::string & msg) {
        res.status = code;
        res.set_content(jerror(code, ec, msg).dump(), "application/json");
    };

    // ---- reference management (api_v2 /change_refer, /set_refer_audio) ------
    auto apply_refer = [&](const std::string & ref, const std::string & prompt,
                           const std::string & lang, httplib::Response & res) {
        if (ref.empty()) return jerr(res, 400, "bad_request", "refer_wav_path is required");
        if (prompt.empty()) return jerr(res, 400, "bad_request", "prompt_text is required");
        if (!path_allowed(ref))
            return jerr(res, 403, "forbidden",
                        "refer_wav_path is not under an allowed directory (see --allow-path)");
        if (!fs::exists(gsvpath::from_utf8(ref)))
            return jerr(res, 400, "bad_request", "refer_wav_path does not exist");
        if (const double dur = wav_duration_seconds(ref); dur > 30.0)
            return jerr(res, 400, "bad_request", "refer_wav_path too long (> 30 s); trim it to 3-10 s");
        const std::string pl = lang.empty() ? "zh" : lang;
        {
            std::lock_guard<std::mutex> lk(st.ref_mu);
            st.default_ref_path = ref;
            st.default_prompt_text = prompt;
            st.default_prompt_lang = pl;
        }
        std::printf("[refer] default reference set: %s (prompt_lang=%s)\n", ref.c_str(), pl.c_str());
        res.set_content(json{{"message", "ok"}, {"refer_wav_path", ref},
                             {"prompt_text", prompt}, {"prompt_lang", pl}}.dump(),
                        "application/json");
    };

    auto refer_params = [](const httplib::Request & req, httplib::Response & res, auto && apply) {
        const std::string ref = req.has_param("refer_wav_path") ? req.get_param_value("refer_wav_path") : "";
        const std::string pr = req.has_param("prompt_text") ? req.get_param_value("prompt_text") : "";
        const std::string pl = req.has_param("prompt_lang") ? req.get_param_value("prompt_lang") : "";
        apply(ref, pr, pl, res);
    };

    svr.Get("/change_refer", [&](const httplib::Request & req, httplib::Response & res) {
        refer_params(req, res, apply_refer);
    });
    svr.Post("/change_refer", [&](const httplib::Request & req, httplib::Response & res) {
        json b = json::object();
        if (!req.body.empty()) { try { b = json::parse(req.body); } catch (...) {} }
        auto pick = [&](const char * k) -> std::string {
            if (b.contains(k) && b[k].is_string()) return b[k].get<std::string>();
            if (req.has_param(k)) return req.get_param_value(k);
            return "";
        };
        apply_refer(pick("refer_wav_path"), pick("prompt_text"), pick("prompt_lang"), res);
    });

    // ASR is not part of this engine (no recognizer); documented as 501.
    auto asr_not_supported = [](const httplib::Request &, httplib::Response & res) {
        res.status = 501;
        res.set_content(jerror(501, "not_implemented",
                               "ASR is not supported by gsv.cpp; run an external ASR service")
                            .dump(),
                        "application/json");
    };
    svr.Get("/asr", asr_not_supported);
    svr.Post("/asr", asr_not_supported);
    svr.Get("/tts_ws", [](const httplib::Request &, httplib::Response & res) {
        res.status = 501;
        res.set_content(jerror(501, "not_implemented",
                               "WebSocket streaming is not available in this build; use POST /tts "
                               "with stream=true (HTTP chunked)")
                            .dump(),
                        "application/json");
    });

    svr.Get("/openapi.json", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(openapi_json(), "application/json");
    });
    svr.Get("/docs", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(openapi_json(), "application/json");
    });

    svr.Get("/progress", [](const httplib::Request &, httplib::Response & res) {
        // stage: 0 idle, 1 text frontend, 2 AR, 3 DiT/CFM, 4 vocoder, 5 done
        res.set_content(json{{"stage", gsvprog::stage.load()},
                             {"cur", gsvprog::cur.load()},
                             {"total", gsvprog::total.load()}}.dump(),
                        "application/json");
    });

    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        size_t pending;
        { std::lock_guard<std::mutex> lk(qm); pending = queue.size(); }
        bool ready;
        { std::lock_guard<std::mutex> lk(st.eng_mu); ready = (bool) st.eng || st.mock_engine; }
        res.set_content(json{{"ok", true},
                             {"engine", ready ? "ready" : "loading"},
                             {"voice", st.voice},
                             {"reloading", st.reloading.load()},
                             {"sidecar", g_mock_sidecar || sidecar_alive(side)},
                             {"queue", pending},
                             {"waiters", st.waiters.load()}}.dump(),
                        "application/json");
    });

    svr.Get("/metrics", [&](const httplib::Request &, httplib::Response & res) {
        size_t pending;
        { std::lock_guard<std::mutex> lk(qm); pending = queue.size(); }
        res.set_content(json{{"tts_requests", st.n_tts.load()},
                             {"errors", st.n_errors.load()},
                             {"jobs", st.n_jobs.load()},
                             {"timeouts", st.n_timeouts.load()},
                             {"busy", st.n_busy.load()},
                             {"reloads", st.n_reloads.load()},
                             {"ar_batches", st.ar_batcher ? (uint64_t) st.ar_batcher->n_batches.load() : 0},
                             {"ar_tasks", st.ar_batcher ? (uint64_t) st.ar_batcher->n_tasks.load() : 0},
                             {"sched_batches", st.sched ? (uint64_t) st.sched->n_batches.load() : 0},
                             {"sched_texts", st.sched ? (uint64_t) st.sched->n_texts.load() : 0},
                             {"queue", pending},
                             {"waiters", st.waiters.load()},
                             {"last_total_ms", st.last_total_ms.load()}}.dump(),
                        "application/json");
    });

    svr.Get("/models", [&](const httplib::Request & req, httplib::Response & res) {
        if (req.has_param("rescan")) st.voices.init(st.base.model_dir);
        json arr = json::array();
        for (const auto & v : st.voices.list()) {
            arr.push_back(json{{"name", v.name},
                               {"dir", v.dir},
                               {"ar", v.cfg.ar_file},
                               {"cond", v.cfg.cond_file},
                               {"dit", v.cfg.dit_file}});
        }
        res.set_content(json{{"voices", arr},
                             {"current", st.voice},
                             {"reloading", st.reloading.load()},
                             {"models_dir", st.base.model_dir}}.dump(),
                        "application/json");
    });

    for (const char * p : {"/set_gpt_weights", "/set_sovits_weights"}) {
        svr.Get(p, [&](const httplib::Request & req, httplib::Response & res) {
            const std::string want = req.has_param("weights_path")
                                         ? req.get_param_value("weights_path")
                                         : (req.has_param("voice") ? req.get_param_value("voice") : "");
            if (want.empty()) return jerr(res, 400, "bad_request", "weights_path is required");
            Voice v;
            if (!st.voices.get(want, v)) {
                st.voices.init(st.base.model_dir);   // 热加载新导入的 voice
                if (!st.voices.get(want, v)) {
                    return jerr(res, 400, "bad_request",
                                "unknown voice '" + want +
                                    "'; export your V5 model to GGUF and register it under models/voices/<name>/");
                }
            }
            std::string err;
            double ms = 0;
            if (!switch_voice(st, want, err, ms)) return jerr(res, 500, "engine", err);
            res.set_content(json{{"voice", st.voice}, {"load_ms", ms}}.dump(), "application/json");
        });
    }
    svr.Get("/set_refer_audio", [&](const httplib::Request & req, httplib::Response & res) {
        refer_params(req, res, apply_refer);
    });

    svr.Get("/control", [&](const httplib::Request & req, httplib::Response & res) {
        const std::string cmd = req.has_param("command") ? req.get_param_value("command") : "";
        // api_v2 compatibility: a bare GET /control answers 400 "command is required"; clients
        // (GAG, studio scripts) use that as the liveness probe. /health is the better endpoint.
        if (cmd.empty()) {
            return jerr(res, 400, "bad_request", "command is required (try 'restart' or 'exit')");
        }
        res.set_content(json{{"message", "ok: " + cmd}}.dump(), "application/json");
        if (cmd == "exit") {
            std::printf("[gsv_server] exit requested\n");
            std::thread([&]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                svr.stop();
            }).detach();
        } else if (cmd == "restart") {
            std::printf("[gsv_server] restart requested (reloading voice '%s')\n", st.voice.c_str());
            std::thread([&]() {
                const std::string name = st.voice;
                std::string err;
                double ms = 0;
                if (!switch_voice(st, name, err, ms)) std::fprintf(stderr, "[gsv_server] restart: %s\n", err.c_str());
            }).detach();
        }
    });

    svr.Post("/reference", [&](const httplib::Request & req, httplib::Response & res) {
        if (req.body.empty()) return jerr(res, 400, "bad_request", "empty body (send raw wav bytes)");
        std::string name = req.has_param("name") ? req.get_param_value("name") : "ref.wav";
        for (char & c : name) if (c == '/' || c == '\\' || c == ':') c = '_';
        try {
            fs::path dir = fs::temp_directory_path() / "gsv_refs";
            fs::create_directories(dir);
            // TTL 清理：24 h 之前的上传
            {
                std::error_code ce;
                const auto now = fs::file_time_type::clock::now();
                for (const auto & e : fs::directory_iterator(dir, ce)) {
                    auto t = fs::last_write_time(e.path(), ce);
                    if (!ce && now - t > std::chrono::hours(24)) fs::remove(e.path(), ce);
                }
            }
            fs::path path = dir / (std::to_string((long long) (now_s() * 1000)) + "_" + name);
            std::ofstream f(path, std::ios::binary);
            f.write(req.body.data(), (std::streamsize) req.body.size());
            f.close();
            res.set_content(json{{"path", path.string()}, {"bytes", req.body.size()}}.dump(),
                            "application/json");
        } catch (const std::exception & e) {
            jerr(res, 500, "internal", std::string("save failed: ") + e.what());
        }
    });

    auto handle_tts = [&](const httplib::Request & req, httplib::Response & res, const json & body) {
        std::string rid = req.get_header_value("X-Request-Id");
        if (rid.empty()) rid = rlog.new_id();
        res.set_header("X-Request-Id", rid);
        gsvprog::reset();
        gsvprog::begin(1);                 // text frontend / g2p
        TtsOutcome o = run_tts(st, side, rlog, body, rid);
        gsvprog::done();
        if (o.code != 200) return jerr(res, o.code, o.error_code, o.message);
        res.set_content(o.wav, "audio/wav");
    };

    // ---- streaming: chunked WAV (header first, PCM per segment) --------------
    auto wav_stream_header = []() {
        std::string h;
        const uint32_t sr = (uint32_t) kSampleRate;
        h += "RIFF"; put_u32(h, 0xFFFFFFFFu);
        h += "WAVEfmt "; put_u32(h, 16); put_u16(h, 1); put_u16(h, 1);
        put_u32(h, sr); put_u32(h, sr * 2); put_u16(h, 2); put_u16(h, 16);
        h += "data"; put_u32(h, 0xFFFFFFFFu);
        return h;
    };

    auto handle_tts_stream = [&](const httplib::Request & req, httplib::Response & res, const json & body) {
        std::string rid = req.get_header_value("X-Request-Id");
        if (rid.empty()) rid = rlog.new_id();
        res.set_header("X-Request-Id", rid);
        res.set_header("Cache-Control", "no-store");
        res.set_header("X-Accel-Buffering", "no");

        SynthParams p;
        parse_params(body, p);
        {
            std::lock_guard<std::mutex> lk(st.ref_mu);
            if (p.ref_path.empty()) p.ref_path = st.default_ref_path;
            if (p.prompt_text.empty()) p.prompt_text = st.default_prompt_text;
            if (!body.contains("prompt_lang") && !st.default_prompt_lang.empty())
                p.prompt_lang = st.default_prompt_lang;
        }
        const std::string media_type = jstr(body, "media_type", "wav");
        {
            const ParamError pe = validate_tts_params(p, media_type);
            if (!pe.ok()) return jerr(res, pe.code, pe.ec, pe.msg);
        }
        if (!p.voice.empty() && p.voice != st.voice) {
            std::string err; double ms = 0;
            if (!switch_voice(st, p.voice, err, ms)) return jerr(res, 500, "engine", err);
        }
        if (!st.mock_engine && !snapshot(st)) return jerr(res, 503, "engine", "engine not ready");
        if (st.waiters.load() >= kMaxWaiters) {
            st.n_busy++;
            return jerr(res, 503, "busy", "server busy (too many waiting requests)");
        }
        st.n_tts++;
        const double t0 = now_s();
        auto sent = std::make_shared<std::atomic<bool>>(false);
        res.set_chunked_content_provider("audio/wav",
            [&, p, rid, t0, sent](size_t, httplib::DataSink & sink) -> bool {
                if (sent->exchange(true)) { return false; }
                st.waiters.fetch_add(1);
                struct WaiterGuard {
                    std::atomic<int> * w;
                    ~WaiterGuard() { w->fetch_sub(1); }
                } wg{ &st.waiters };
                SynthTiming tm;
                std::string err;
                bool ok = false;
                try {
                    const std::string hdr = wav_stream_header();
                    if (!sink.write(hdr.data(), hdr.size())) return false;
                    ok = svc::synth_text_stream_q(
                        st, side, p,
                        [&sink](const int16_t * pcm, size_t n) -> bool {
                            return sink.write(reinterpret_cast<const char *>(pcm), n * sizeof(int16_t));
                        },
                        tm, err);
                } catch (const std::bad_alloc &) {
                    err = "out of memory while streaming";
                } catch (const std::exception & e) {
                    err = std::string("exception: ") + e.what();
                }
                const double total_ms = (now_s() - t0) * 1000;
                if (ok) {
                    st.last_total_ms = total_ms;
                    std::printf("[tts-stream %s] voice=%s segs=%d phones=%d | g2p %.0f prep %.0f "
                                "bert %.0f ar %.0f synth %.0f | total %.0f ms\n",
                                rid.c_str(), st.voice.c_str(), tm.segments, tm.phones, tm.g2p_ms,
                                tm.prep_ms, tm.bert_ms, tm.ar_ms, tm.synth_ms, total_ms);
                } else {
                    st.n_errors++;
                    std::printf("[tts-stream %s] error: %s (after %.0f ms)\n", rid.c_str(),
                                err.c_str(), total_ms);
                }
                sink.done();
                return true;
            });
    };

    svr.Post("/tts", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception & e) { return jerr(res, 400, "bad_request", std::string("bad json: ") + e.what()); }
        if (jbool(body, "stream", false)) return handle_tts_stream(req, res, body);
        handle_tts(req, res, body);
    });

    svr.Get("/tts", [&](const httplib::Request & req, httplib::Response & res) {
        static const char * keys[] = {"text", "text_lang", "ref_audio_path", "prompt_text", "prompt_lang",
                                      "voice", "text_split_method", "media_type", "sample_steps", "cfg_rate",
                                      "seed", "top_k", "top_p", "temperature", "repetition_penalty",
                                      "fragment_interval", "max_steps", "early_stop", "seg_chars", "speed_factor", "skip_first_step",
                                      "dbcache", "dbcache_fn", "dbcache_threshold", "dbcache_warmup",
                                      "dbcache_downsample", "stream",
                                      "dit_uncond_reuse",
                                      "timeout_ms"};
        json body = json::object();
        for (const char * k : keys) if (req.has_param(k)) body[k] = req.get_param_value(k);
        if (jbool(body, "stream", false)) return handle_tts_stream(req, res, body);
        handle_tts(req, res, body);
    });

    svr.Post("/batch", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception & e) { return jerr(res, 400, "bad_request", std::string("bad json: ") + e.what()); }
        std::vector<std::string> texts;
        if (body.contains("texts") && body["texts"].is_array()) {
            for (const auto & v : body["texts"]) if (v.is_string()) texts.push_back(v.get<std::string>());
        } else if (body.contains("text") && body["text"].is_string()) {
            texts.push_back(body["text"].get<std::string>());
        }
        if (texts.empty()) return jerr(res, 400, "bad_request", "texts[] is required");

        auto job = std::make_shared<Job>();
        job->id = gen_job_id();
        parse_params(body, job->base);
        if (job->base.ref_path.empty()) return jerr(res, 400, "bad_request", "ref_audio_path is required");
        if (!path_allowed(job->base.ref_path))
            return jerr(res, 403, "forbidden",
                        "ref_audio_path is not under an allowed directory (see --allow-path)");
        if (job->base.prompt_text.empty()) return jerr(res, 400, "bad_request", "prompt_text is required");
        job->items.resize(texts.size());
        for (size_t i = 0; i < texts.size(); i++) job->items[i].text = texts[i];
        {
            std::lock_guard<std::mutex> jl(jobs_m);
            jobs[job->id] = job;
        }
        {
            std::lock_guard<std::mutex> lk(qm);
            queue.push_back(job);
        }
        qcv.notify_one();
        st.n_jobs++;
        std::printf("[batch] %s: %zu items\n", job->id.c_str(), texts.size());
        res.set_content(json{{"job_id", job->id}, {"total", texts.size()}}.dump(), "application/json");
    });

    svr.Get(R"(/jobs/([^/]+))", [&](const httplib::Request & req, httplib::Response & res) {
        const std::string id = req.matches[1];
        std::shared_ptr<Job> job;
        {
            std::lock_guard<std::mutex> jl(jobs_m);
            auto it = jobs.find(id);
            if (it != jobs.end()) job = it->second;
        }
        if (!job) {
            // 进程重启后从磁盘恢复（只读）
            std::ifstream f(job_dir(id) / "meta.json");
            if (!f) return jerr(res, 404, "not_found", "no such job");
            json meta;
            try { f >> meta; }
            catch (...) { return jerr(res, 500, "internal", "bad job meta"); }
            res.set_content(meta.dump(), "application/json");
            return;
        }
        std::lock_guard<std::mutex> jl(job->m);
        json items = json::array();
        int done = 0;
        for (const auto & it : job->items) {
            if (!it.wav.empty()) done++;
            items.push_back(json{{"text", it.text}, {"audio_s", it.audio_s}, {"ms", it.ms},
                                 {"done", !it.wav.empty()}, {"error", it.error}});
        }
        const char * states[] = {"queued", "running", "done", "canceled", "failed"};
        res.set_content(json{{"job_id", job->id},
                             {"state", states[std::max(0, std::min(4, job->state))]},
                             {"done", done},
                             {"total", (int) job->items.size()},
                             {"fatal", job->fatal},
                             {"items", items}}.dump(),
                        "application/json");
    });

    svr.Get(R"(/jobs/([^/]+)/events)", [&](const httplib::Request & req, httplib::Response & res) {
        const std::string id = req.matches[1];
        std::shared_ptr<Job> job;
        {
            std::lock_guard<std::mutex> jl(jobs_m);
            auto it = jobs.find(id);
            if (it != jobs.end()) job = it->second;
        }
        if (!job) return jerr(res, 404, "not_found", "no such job");
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider(
            "text/event-stream",
            [job](size_t, httplib::DataSink & sink) {
                json j;
                {
                    std::lock_guard<std::mutex> jl(job->m);
                    int done = 0;
                    for (const auto & it : job->items) if (!it.wav.empty()) done++;
                    static const char * states[] = {"queued", "running", "done", "canceled", "failed"};
                    j = json{{"job_id", job->id},
                             {"state", states[std::max(0, std::min(4, job->state))]},
                             {"done", done},
                             {"total", (int) job->items.size()}};
                }
                const std::string s = "data: " + j.dump() + "\n\n";
                if (!sink.write(s.data(), s.size())) return false;
                {
                    std::lock_guard<std::mutex> jl(job->m);
                    if (job->state >= 2) return false;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
                return true;
            });
    });

    svr.Get(R"(/jobs/([^/]+)/result/(\d+))", [&](const httplib::Request & req, httplib::Response & res) {
        const std::string id = req.matches[1];
        const int idx = std::atoi(req.matches[2].str().c_str());
        std::shared_ptr<Job> job;
        {
            std::lock_guard<std::mutex> jl(jobs_m);
            auto it = jobs.find(id);
            if (it != jobs.end()) job = it->second;
        }
        if (!job) {
            std::ifstream f(job_dir(id) / (std::to_string(idx) + ".wav"), std::ios::binary);
            if (!f) return jerr(res, 404, "not_found", "no such job");
            std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            res.set_content(bytes, "audio/wav");
            return;
        }
        std::lock_guard<std::mutex> jl(job->m);
        if (idx < 0 || idx >= (int) job->items.size()) return jerr(res, 404, "not_found", "bad index");
        if (job->items[idx].wav.empty()) return jerr(res, 409, "not_ready", "not ready");
        res.set_content(job->items[idx].wav, "audio/wav");
    });

    svr.Delete(R"(/jobs/([^/]+))", [&](const httplib::Request & req, httplib::Response & res) {
        const std::string id = req.matches[1];
        std::shared_ptr<Job> job;
        {
            std::lock_guard<std::mutex> jl(jobs_m);
            auto it = jobs.find(id);
            if (it == jobs.end()) return jerr(res, 404, "not_found", "no such job");
            job = it->second;
        }
        {
            std::lock_guard<std::mutex> jl(job->m);
            job->cancel = true;
        }
        res.set_content(json{{"message", "cancel requested"}}.dump(), "application/json");
    });

    std::printf("[gsv_server] listening on http://%s:%d\n", host.c_str(), port);
    if (!svr.listen(host, port)) {
        std::fprintf(stderr, "[gsv_server] listen failed on %s:%d (port in use?)\n", host.c_str(), port);
        stopping = true;
        qcv.notify_all();
        st.sched = nullptr;
        sched.shutdown();
        st.ar_batcher = nullptr;
        ar_batcher.shutdown();
        worker.join();
        return 1;
    }
    std::printf("[gsv_server] shutting down\n");
    stopping = true;
    qcv.notify_all();
    st.sched = nullptr;
    sched.shutdown();
    st.ar_batcher = nullptr;
    ar_batcher.shutdown();
    worker.join();
    janitor.join();
    return 0;
}
