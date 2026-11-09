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

The clock-independent comparison (one binary, one thread, nothing else running, perf stat over the
whole run - the host had dropped to 2.25 GHz by then, so wall clock is worthless):

| config | instructions | cycles | IPC |
|---|---|---|---|
| f32 conv | 129.74 G | 43.88 G | 2.96 |
| bf16, oc-outer loop order | 88.19 G | 44.09 G | 2.00 |
| bf16, t-block-outer | **88.01 G** | **33.86 G** | 2.60 |

**The loop order was the whole story.** With `oc` outer, every oc block re-read the entire im2col
buffer from DRAM. That buffer is `KPP * OLp * 4` bytes, i.e. **K times the activation**, so the largest
dec layer (OL=16480, IC=128, OC=128, K=11) moved 46.4 MB * 8 oc blocks = 371 MB against the f32
kernel's 8.4 MB * 8 = 67 MB - and the f32 kernel needs no im2col at all. Putting the t block outer keeps
one 12 x KPP x 4 = 34 KB activation tile in L1 and streams the 360 KB weight tile from L2 instead:
44.09 -> 33.86 G cycles, **1.30x against f32 over the whole run** (about 1.4x on the dec stage alone).

Two hypotheses were tested and refuted on the way there, which is worth recording because each looked
convincing:

* "the six accumulators give a reuse distance (6 cycles) shorter than the ~4-6 cycle vdpbf16ps latency,
  so the single dp port idles". Doubling to **12 accumulators changed cycles by 0.7%** (44.39 -> 44.09 G)
  and dropped IPC further (2.08 -> 2.00). Not a latency problem.
* "the burstable clock makes the comparison meaningless". True for wall clock, which is why every number
  above is instructions/cycles, measured in interleaved pairs on one binary.

The remaining gap to f32's IPC (2.60 vs 2.96) is the per-tap `vpbroadcastd`: perf puts it at 66% of
kernel samples against 8% for `vdpbf16ps` itself. One broadcast feeds exactly one dpbf16ps today, so the
fix would be to feed two oc blocks per broadcast - tried earlier and it regressed to 2861 ms on wall
clock, but that measurement was taken with the oc-outer loop order, so it should be retried now that the
traffic problem is fixed.

### Result at the deployment configuration (threads=2)

Interleaved pair, one binary, nothing else running, perf stat over the whole run:

| config | dec | whole synth | instructions | cycles | IPC |
|---|---|---|---|---|---|
| f32 conv | 2615.8 ms | 5004.8 ms | 129.18 G | 76.54 G | 1.69 |
| bf16 conv | **1541.5 ms** | **3076.9 ms** | 87.88 G | 58.42 G | 1.50 |

**1.70x on the dec stage, 1.63x on the whole synthesis**, with the wav cos moving only from 0.999917 to
0.999769 - i.e. the same audio, 1.6x faster. Enable with `GSV_BF16_CONV=1`; the f32 path stays the
default until this has been listened to across more voices and both languages.

Historical wall-clock table, kept only to show how misleading it was (same binary, 2 threads, at the
3.07 GHz the box still had at that point, and all of it measured before the loop order was fixed):

Raw wall-clock numbers, same binary, 2 threads, at the 3.07 GHz the box still had at that point:

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

## AR stage: where the time actually goes (long text)

Measured with `bench_ar --seqs N --fixture ...` on the same box, threads=2, and read off the built-in
`GSV_AR_PROFILE` report (decode only - `bench_ar`'s `total` includes the prefill, which is what made the
first batch experiment look useless):

| quantity | value |
|---|---|
| decode, B=1 | **6.71 ms/step** = 21 GMAC/s (compute bound, see the B scan below) |
| prefill | **~3.0 ms per position** for S ~ 250 |
| decode, B=8 | 46.7 ms/step -> 2.1x aggregate, i.e. batching buys almost nothing |
| prefill, B=8 | 7.6x of B=1 - also nearly linear |
| AVX-512 build vs AVX2 | decode **1.9x slower** (12.71 vs 6.71 ms/step) - keep the AVX2 build |
| prefill hotspot (perf) | 67% `tinyBLAS_Q0_AVX<q8_0,q8_0,float>::gemm4xN`, 21% `flash_attn_ext` |

Why batching does not help: **decode is compute bound**, not weight-bandwidth bound. The B scan settles
it - throughput is flat at 20.7 GMAC/s (B=1) and 25.1 GMAC/s (B=8), so every extra sequence costs a full
share of MACs:

| B | per-step | GMAC/s |
|---|---|---|
| 1 | 6.8 ms | 20.7 |
| 4 | 24.3 ms | 23.2 |
| 8 | 45.2 ms | 25.1 |

At B=1 the two hypotheses are numerically indistinguishable (a q8 weight is one byte and one MAC, so
"85 MB / 6.8 ms" and "77 M MAC / 6.8 ms" both give ~12 per millisecond), which is why only the B scan can
separate them. The model is 88,669,856 bytes, i.e. ~77 M params, not the 150 MB assumed earlier.

The decisive check turned out to be SMT scaling, not the B scan alone: one thread gives 6.99 ms/step and
two give 6.71 ms/step, i.e. **1.04x**. Extra SMT threads add issue capacity and latency hiding but no
memory bandwidth, so 1.04x rules out a latency-bound kernel and leaves a single shared execution unit.

Instruction budget for the non-VNNI q8 GEMV: per 32-weight block it needs one 32 B load, vpmaddubsw,
vpmaddwd, vpaddd and the amortised scale (cvt + fmadd), i.e. ~6 instructions per 32 MAC = 4.7 MAC per
instruction. At 2.25 GHz that is 12.7 GMAC/s at IPC 1.2 and 15.9 at IPC 1.5, and the measured 11.3 (B=1)
to 13.6 (B=8) GMAC/s sits at 85-90% of that. So the AR decode is instruction-issue bound and essentially
saturated - there is no 4x sitting in the loop nest.

Four independent observations agree with that model, and they were worth getting separately because each
one on its own is ambiguous:

1. the instruction budget above (85-90% of what the kernel can issue);
2. SMT scaling of 1.04x - extra SMT threads add issue slots and latency hiding but no bandwidth;
3. the B scan is linear, i.e. every extra sequence pays a full set of MACs;
4. `GSV_MM_ROW_OUTER` changes nothing once it is genuinely active. This is the sharpest test because the
   theory *predicts* a null result: reordering the block loops moves weight bytes from DRAM to L1 but
   issues exactly the same vpmaddubsw/vpmaddwd/vpaddd, and we are not byte bound. B=8 44.79 -> 45.28
   ms/step, and the sampled token stream md5 is identical, so the swap is correct but useless.

Caveat for anyone repeating this: the first two attempts at that experiment were invalid. The guard
started as `ne11 >= 2 && ne12 == 1`, which never matches, because the batch dimension of a 2D weight
broadcast lives in ne12 (decode is ne11=1, ne12=B); a patch also got lost in a broken heredoc, and the
token-stream md5 comparison "passed" because both branches were the untouched original. Verify a change
is live (here: one debug line per (ne11, ne12) showing `swap=1` for both prefill and decode) before
trusting any measurement of it.

The only way past it is fewer instructions per MAC, i.e. a wider dot: VNNI (`vpdpbusd`, 64 MAC/instr)
needs the VEX form this EPYC lacks (AVX-512 VNNI exists but that build is 1.9x slower here), and
`vdpbf16ps` (32 MAC/instr) would force 16-bit weights - which is what the vocoder now does.

This also disposes of `GSV_MM_ROW_OUTER`: swapping the block loops in
`ggml_compute_forward_mul_mat_one_chunk` so a 16-row weight block stays resident is bit-identical
(verified by comparing the sampled token stream md5), but it changes nothing measurable - B=1 6.89 ->
6.79 ms, B=4 24.32 -> 24.09, B=8 45.21 -> 45.37 - because the weights were never being re-read.

The real cost for long text is the **per-segment fixed price**. `split_text` (non-streaming) used to cut
every **8 characters**, so a 240-character sentence became ~30 segments, and every segment paid its own
full prefill of the shared prompt (~1.4 s at S=118) plus a fresh vocoder call. `seg_chars` now makes the
segment length configurable; the physical ceiling is `early_stop` (400 sem tokens ~ 4.6 tokens/char
~ 87 characters), and ~60 characters is a safe default for throughput.

The remaining architectural fix (not implemented) is to stop re-prefilling the shared prompt: it is
identical for every segment (`prm->phones`, `prm->bert`, `ref->codes`), so prefill it once with B=1,
copy the KV into every sequence slot (8 seqs x 103 positions x 48 KiB = 39 MB, cheap), and let each
segment prefill only its own target. That turns the per-segment `O(Pp^2)` attention into
`t * (Pp + t)`.

Also tested and rejected on this CPU: `GGML_AVX_VNNI=ON` builds fine (149 `vpdpbusd` in the library) but
dies with SIGILL - this EPYC exposes `avx512_vnni` (EVEX) and not `avx_vnni` (VEX). `GGML_AVX512=ON` is
1.9x slower for the AR decode (12.71 vs 6.71 ms/step), the same verdict as for the vocoder convolutions.
