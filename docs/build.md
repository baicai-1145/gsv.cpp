# Building gsv.cpp from scratch

This guide takes a clean machine to a working `/tts` server: engine, GGUF models, and the Python
text frontend.

Two supported paths:

- **Linux** (x86_64, glibc 2.35+): the reference build, also what CI produces.
- **Windows** (x86_64, Visual Studio 2022): same code, MSVC; a prebuilt one-click package also
  exists on Hugging Face (`windows/GSV.CPP-win64-*.zip`).

CPU-only builds are supported on both (skip the Vulkan parts).

```
text ──► Python sidecar (G2P + BERT ids) ──┐
                                            ├─► gsv_server (C++/ggml) ──► 48 kHz WAV
GGUF models (AR/cond/DiT/refcode/BERT/HuBERT/vocoder) ─┘
```

## 1. Prerequisites

### Linux (Ubuntu 22.04+)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3 python3-venv \
                        libvulkan-dev vulkan-tools glslang-tools mesa-vulkan-drivers
```

- CMake **3.20+**, a C++17 compiler, Python **3.10+**.
- `glslang-tools` provides `glslc`, which ggml-vulkan needs at build time.
- GPU users need a working Vulkan driver (`vulkaninfo --summary`).

### Windows

- **Visual Studio 2022** with the "Desktop development with C++" workload (MSVC + CMake).
- **Vulkan SDK** 1.3+ (from <https://vulkan.lunarg.com/>), which ships `glslc.exe`.
- **Git** and **Python 3.10+**.
- Build from a *x64 Native Tools Command Prompt for VS 2022* (or run `vcvars64.bat` first).

## 2. Get the sources

```bash
git clone https://github.com/KakaruHayate/gsv.cpp
cd gsv.cpp
```

The engine needs a patched **llama.cpp/ggml** (conv1d + audio ops are not upstream). The exact
commit is pinned in `.github/workflows/ci.yml` (`LLAMA_CPP_COMMIT`):

```bash
git clone https://github.com/ggml-org/llama.cpp
git -C llama.cpp checkout f0c41e0168dfd4b5ef72b21d1a311b24cc7a894a
git -C llama.cpp apply ../patches/0001-ggml-audio-patch-port-on-llamacpp.patch
```

> The patch is what makes `ggml_conv_1d`, `ggml_conv_transpose_1d`, `ggml_pad_reflect_1d` and the
> sub-pixel path available. Without it the engine does not compile.

> NOTE: an optional `0002` vectorised-epilogue patch existed and was withdrawn: the version committed here was the first (broken) revision, which dropped `tanhf` from the scalar
fallback and half-wave-rectified `conv_post` (wav cos 0.999917 -> 0.140). The fix was only ever in a worktree, never in the repo. Its measured gain was 1.087x on dec, not worth the
risk without a verified patch in tree.

## 3. Build ggml

### Vulkan (GPU)

```bash
cmake -S llama.cpp -B llama.cpp/build-vk-rel \
  -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build llama.cpp/build-vk-rel --config Release \
  --target ggml ggml-base ggml-cpu ggml-vulkan -j 2
```

Windows: add `-G "Visual Studio 17 2022" -A x64` to the configure step (the libraries end up in
`llama.cpp/build-vk-rel/Release/`).

### CPU only

```bash
cmake -S llama.cpp -B llama.cpp/build-cpu-rel \
  # WARNING: do not replace the flags above with -DGGML_NATIVE=ON on Zen4.
  # On an EPYC 9754 (Bergamo) VM, GGML_NATIVE=ON selects the AVX-512 kernels and they are
  # 8-10x SLOWER than the AVX2 ones for exactly the ops this pipeline runs (measured with
  # tests/bench_ggml_min.c, same binary, same box, only the ggml build differing):
  #     AVX2   build: mul_mat GEMV 15.20 GMAC/s   conv_1d 13.54 GMAC/s
  #     AVX-512 build: mul_mat GEMV  1.73 GMAC/s   conv_1d  1.33 GMAC/s
  # End to end that turned a 1.7 s decode into 43.6 s and RTF 0.23 into 6.41. The documented
  # flags are the fast ones; GGML_NATIVE=ON is a trap on this hardware.
  #
  # Mechanism, narrowed with perf (not guessed): the slow build executes 115.9 G instructions for
  # the same work the fast build does in 10.6 G, at the SAME 3.06 GHz core clock - so it is code
  # generation, not frequency. Both builds are vectorised and both use zmm; the hot loop of
  # ggml_vec_dot_f16 in the -march=native build spills its accumulators and round-trips through
  # memory every iteration (perf annotate: vmovaps -0x670(%rbp),%zmm0 -> kmovw -0x814(%rbp),%k1 ->
  # vcvtph2ps %ymm1,%zmm0{%k1} -> vmovaps %zmm0,(%rax) -> vmovaps (%rax),%zmm0), i.e. one masked
  # 16-element block at a time with the mask reloaded from the stack, 343 instructions and 7 calls
  # in the function. The fast build's version of the same symbol is 68 instructions, 4 vfmadd,
  # 19 zmm, no calls. Ruled out along the way: the AVX-512 kernels themselves (GGML_NATIVE=OFF +
  # GGML_AVX512=ON is fast), -mtune=znver4 (fast), missing F16C (both builds emit vcvtps2ph), and
  # any source-level branch difference (preprocessed output is identical apart from the usual
  # immintrin.h #pragma GCC target boilerplate). It is a gcc 13 register-allocation/unrolling cliff
  # for this function when tuned for znver4.
  -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DGGML_BUILD_TESTS=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build-cpu-rel --config Release --target ggml ggml-base ggml-cpu -j
```

## 4. Build the server

```bash
cmake -S . -B build -DGSV_GGML_LIB_DIR=llama.cpp/build-vk-rel -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target gsv_server -j 4
```

Output:

- Linux: `build/gsv_server`
- Windows: `build/Release/gsv_server.exe`

Tests (optional but recommended): `-DGSV_BUILD_TESTS=ON` is the default and adds
`test_add_act`, `test_conv_direct_act`, `test_svc`; run them with `ctest --test-dir build`.

Windows: copy `ggml*.dll` next to the exe (they are in `llama.cpp/build-vk-rel/Release/`), or add
that directory to `PATH`. The VC++ runtime (`vcruntime140*.dll`, `msvcp140.dll`) must be present —
the prebuilt package ships them.

## 5. Get the models

The engine reads GGUF only. Either

**a) download the prebuilt set** (recommended) from <https://huggingface.co/KakaruHayate/GSV.CPP>:

```
models/
  gsv-ar-q8.gguf              # AR, Q8 (default recommendation)
  gsv-cond-f16.gguf           # v5turbo cond
  gsv-dit-f16all.gguf         # v5turbo DiT
  gsv-refcode.gguf            # v5turbo reference encoder
  gsv-bert-f16.gguf           # shared BERT
  gsv-hubert-f16.gguf         # shared CN-HuBERT
  gsv-vocoder-f32.gguf        # shared HiFi-GAN (sub-pixel, F32)
  resampy_kaiser_best.bin     # resampling filter table
```

v5dev additionally needs `gsv-v5dev-cond-f16.gguf`, `gsv-v5dev-dit-f16all.gguf`,
`gsv-v5dev-refcode.gguf`; describe it in a voice manifest (`models/voices/v5dev/manifest.json`).

**b) convert your own fine-tune** with `tools/export_gguf` — see
[tools/export_gguf/README.md](../tools/export_gguf/README.md). You still need BERT/HuBERT
(prebuilt or `--kind bert/hubert`).

## 6. Text frontend (G2P sidecar)

The frontend is vendored in `frontend/` and needs no torch/transformers.

```bash
python3 -m venv .venv-sidecar
.venv-sidecar/bin/pip install -r frontend/requirements-zh.txt      # Windows: .venv-sidecar\Scripts\pip
.venv-sidecar/bin/pip install -r frontend/requirements-multi.txt   # en/ja/ko/yue
```

Data (not in git):

| what | where to put it | env override |
|---|---|---|
| G2PW model (`g2pW.onnx` + dicts, ~608 MB) | `models/frontend/g2pw/` | `GSV_G2PW_MODEL` |
| BERT tokenizer (`vocab.txt`, `config.json`, `tokenizer_config.json`) | `models/frontend/bert_tokenizer/` | `GSV_BERT_DIR` |
| `nltk_data` (cmudict + `averaged_perceptron_tagger_eng`) | `models/frontend/nltk_data/` | `NLTK_DATA` |
| fast-langdetect model (`lid.176.bin`, ~125 MB) | `models/frontend/fast_langdetect/` | `GSV_FAST_LANGDETECT` |

All four are published on Hugging Face under `frontend/`. With the layout above the sidecar finds
them automatically; the env vars are only needed for custom locations.

Start it:

```bash
.venv-sidecar/bin/python tools/text_sidecar.py --host 127.0.0.1 --port 9882
```

`GET /health` answers `{"ok": true, "ready": true}` once the Chinese frontend is warm (a few
seconds; the other languages warm on first use).

## 7. Run

```bash
# GPU
./build/gsv_server --models models --device vulkan --sidecar 127.0.0.1:9882 --port 9880
# CPU
./build/gsv_server --models models --device cpu --sidecar 127.0.0.1:9882 --port 9880
```

`--allow-path` restricts which directories may be used for reference audio (defaults to the
current directory + the upload temp dir; `--allow-path '*'` disables the restriction).

Check it:

```bash
curl -s http://127.0.0.1:9880/health
curl -s http://127.0.0.1:9880/models
```

Synthesize (see [api.md](api.md) for all parameters):

```bash
curl -X POST http://127.0.0.1:9880/tts -H 'Content-Type: application/json' -d '{
  "text": "你好，欢迎使用语音合成系统。",
  "text_lang": "all_zh",
  "ref_audio_path": "models/ref.wav",
  "prompt_text": "参考音频对应的文本",
  "prompt_lang": "all_zh",
  "sample_steps": 4
}' -o out.wav
```

Reference audio must be 3–10 s (max 30 s) and readable by the process.

## 8. Troubleshooting

| symptom | cause / fix |
|---|---|
| CMake: `llama.cpp not found` | clone it next to the repo, or pass `-DGSV_LLAMA_DIR=/path/to/llama.cpp` |
| ggml build: `glslc not found` | install `glslang-tools` (Linux) / Vulkan SDK (Windows) |
| `[device] no Vulkan device` | driver missing; try `--device cpu`, or `vulkaninfo` to check |
| `text sidecar 不可达` | sidecar not running / wrong port / it crashed (check its stderr) |
| G2PW download error at startup | `GSV_G2PW_MODEL` points to the wrong directory (must contain `g2pW.onnx`) |
| `reference audio too long` | trim the clip to 3–10 s |
| Windows: missing `ggml-*.dll` | copy them next to `gsv_server.exe` (or add to `PATH`) |
| Linux binary: `GLIBC_2.38 not found` | build on the oldest distro you need (CI uses Ubuntu 22.04) |

## Measured: what q8 and VNNI are worth on a Zen4 box (single core, 1 thread)

`tests/bench_ggml_min.c` on the EPYC 9754 VM, four ggml builds of the same source:

| ggml build | mul_mat f32 GEMV | mul_mat **q8_0** GEMV | conv_1d (im2col) |
|---|---|---|---|
| AVX2 (documented flags) | 15.25 GMAC/s | **19.82 GMAC/s** | **13.72 GMAC/s** |
| AVX-512, no VNNI | 15.80 | — | 10.09 |
| AVX-512 + AVX512_VNNI | 18.67 | **20.74 GMAC/s** | 10.08 |
| -DGGML_NATIVE=ON | 1.73 | — | 1.33 |

Reading it:
* **q8_0 over f32 is +30%** (15.25 -> 19.82) and that is where the AR already lives (`gsv-ar-v2-q8.gguf`).
  The AR is weight-stream bound (84.5 MB per token), so this is a bandwidth win, not an arithmetic one.
* **AVX512_VNNI adds only +4.6% on top of AVX2** for q8_0 (19.82 -> 20.74). The int8 headroom that
  VNNI advertises is not available for these shapes, which is consistent with the earlier finding that
  the dec conv is FMA/L1-bound rather than dot-product-bound. It does not justify rewriting the dec
  conv for int8.
* **AVX-512 costs the conv 26%** (13.72 -> 10.08) while the AR gains ~5%. The dec dominates the
  pipeline (5.18 s vs 2.68 s AR), so on this hardware the **AVX2 build is the fastest overall
  configuration** - shipping AVX-512 would be a net loss.
* `GGML_AVX_VNNI=ON` (VEX-encoded AVX-VNNI) **crashes with SIGILL on this CPU**: it advertises
  `avx512_vnni` but not `avx_vnni`. Use `GGML_AVX512_VNNI=ON` with `GGML_AVX_VNNI=OFF`.

## bf16 weights + fp32 accumulation for the dec conv (implemented, measured, not shipped)

Zen4 has `avx512_bf16` (verified by CPUID: `CPUID.7.1.EAX[5]=1`), so `vdpbf16ps` is available:
two bf16 operands, **fp32 accumulator**, 32 MAC per instruction. `avx512_fp16` is *not* exposed on
this VM (`CPUID.7.0.EDX[23]=0`), and neither is `avx_vnni` (VEX form) nor `avx_vnni_int8`.

Accuracy probe first (`GSV_BF16_SIM=1`, rounds both the folded weights and the conv activations to
bf16 while accumulating in fp32 — numerically what `vdpbf16ps` does):

| path | wav cos vs torch | max_abs | rms_diff |
|---|---|---|---|
| f32 | 0.999917 | 1.293e-02 | 1.925e-03 |
| bf16 in / fp32 acc | **0.999763** | 2.783e-02 | 2.409e-03 |

A real kernel was then written (`GSV_BF16_CONV=1`): weights packed as bf16 pairs along the reduction
axis, im2col materialised as bf16 pairs in `[kp2][t]` layout (8 taps per iteration via
`vcvtneps2bf16` + `vpunpck*epi16`), and a 6-tap x 16-oc micro-kernel of `vdpbf16ps`. It reproduces
the simulation's cos to 6 digits (0.999769), so the arithmetic is right. Micro-benchmark of bb16 vs
f32 GEMV on the same box: 31.0 vs 18.9 GMAC/s (1.67x in cache).

Performance result, however, is a wash:

| config (same binary, threads=2) | dec |
|---|---|
| f32 conv | 2568 ms |
| bf16 conv, scalar im2col | 5669 ms |
| bf16 conv, vectorised im2col | 2367 ms |
| bf16 conv, + two oc blocks per broadcast | 2861 ms |

Three things consumed the 2x arithmetic advantage: (a) a materialised im2col buffer, which the tuned
f32 kernel does not need at all; (b) one `vpbroadcastd` per (tap, t) pair, which perf showed at 66% of
kernel samples against 8% for `vdpbf16ps` itself; (c) `OCp` rounding 24 -> 32. Merging two oc blocks
per broadcast made it worse (12 live accumulators cost more than the saved broadcasts).

Method note, learned the hard way twice in this session: the host is a burstable instance whose clock
oscillates 3.07 -> 1.63..2.04 GHz, so two long runs minutes apart are not comparable; and a buffer
that was never written reads back as the shared zero page, which inflated one "skip the im2col"
measurement to a fake 1264 ms. Both f32 large-GEMV bandwidth and that im2col split were artefacts of
untouched pages. Compare interleaved, on one binary, medians over repeats.
