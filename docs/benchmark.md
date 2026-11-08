# Benchmark

Full pipeline (frontend → AR → cond → DiT → vocoder), same text / reference audio / seed.
Times in milliseconds, median of three warm runs (rows 2–6). Rows 2–6 were re-measured on
2026-11-03 with the current development build (ggml-only F32 HiFi-GAN vocoder with sub-pixel upsampling,
RoPE/adaLN/mish fusion included); rows 1 and 4 are the unchanged torch reference.

| # | configuration | total | G2P/frontend | cond | AR | DiT | vocoder | cache |
|---|---|---|---|---|---|---|---|---|
| 1 | v5turbo torch | **3142** | 162 | 251 | ~2103* | 561 | 65 | — |
| 2 | v5turbo ggml (AR fp16) | **945** | 102 | 93 | 363 | 275 | 68 | — |
| 3 | v5turbo ggml (AR q8) | **1028** | 101 | 95 | 425 | 279 | 67 | — |
| 4 | v5dev torch | **11716** | 163 | 247 | ~2326* | 8924 | 56 | — |
| 5 | v5dev ggml | **5415** | 102 | 126 | 456 | 4617 | 68 | — |
| 6 | v5dev ggml + cache-dit | **4740** | 101 | 120 | 422 | 4005 | 69 | **17/64 = 26.6%** |

The vocoder is the ggml F32 HiFi-GAN with **sub-pixel upsampling**: the five transposed
convolutions are replaced by one 3-tap conv1d on the input grid plus a phase interleave
(`src/gsv_vocoder_ggml.cpp::upsample_subpixel`), and every convolution runs through
`ggml_conv_direct_1d_fused` (no im2col buffer; bias/LeakyReLU, the resblock input LeakyReLU and
the residual add all fold into the same kernel). For the 3.4 s clip in this table the vocoder
takes **~68 ms**, down from ~140 ms with `ggml_conv_transpose_1d`, ~115 ms with sub-pixel + im2col
and ~75 ms with per-conv direct conv (-51% overall). The old ONNX vocoder was ~114 ms. AR varies
by roughly ±20% between runs on this machine (CPU sidecar + GPU sharing), which dominates the
spread of the totals.

\* torch's AR loop is captured as a single CUDA graph and cannot be timed separately; this
column is `total − (frontend + cond + DiT + vocoder)` (AR + scheduling overhead).

## Environment

- CPU: Intel Xeon E5-2675 v3 @ 1.80 GHz (16 threads)
- GPU: NVIDIA GeForce RTX 2070 8 GB, driver supporting Vulkan 1.2
- OS: Windows 10 x64
- ggml: Vulkan backend (`--device vulkan`), weights as described per row
  (AR fp16 = `gsv-ar-f16.gguf`, AR q8 = `gsv-ar-q8.gguf`; cond/DiT F16; refcode F16)
- vocoder: ggml HiFi-GAN, F32 weights (`gsv-vocoder-f32.gguf`), Vulkan backend
- torch: 2.12.1 + CUDA 13.0, fp16, `use_cuda_graph=True`, batch size 1, text split `cut0`
- sample steps: v5turbo 4 / cfg 0.0; v5dev 32 / cfg 1.30
- cache-dit: DBCache `Fn=8`, threshold 0.02, warmup 8, downsample 4

Notes:

- Text: `你好，欢迎使用语音合成系统。`
- Reference: 16 kHz Chinese prompt, ~3.4 s output audio in every row.
- Timings vary by a few percent between runs; the torch numbers include CUDA graph replay
  (two warm-up runs with the same text are done before the measured run).

## DiT fusion (RoPE + adaLN)

`gsv_dit.cpp` builds the DiT step graph with a fused RoPE (`ggml_rope_ext`, head 0 only, with a
`pos` tensor and freq factors that reproduce the checkpoint `inv_freq`) and a fused
`LN(x)·(1+scale)+shift` (`ggml_layernorm_affine` with `wb = [1+scale, shift]`), replacing the
per-block cos/sin multiply chain and the `norm → add → mul → add` sequence.

Same machine, turbo-q8, same text / seed, warm:

| DiT | before | after | delta |
|---|---|---|---|
| v5turbo DiT | 333 ms | 291 ms | −12.6% |
| v5turbo total | 1130 ms | 1033 ms | −8.6% |

Audio correlation against the pre-fusion build is 0.9955; two runs with the same seed are
bit-identical (corr 1.000000), so determinism is preserved. The `mish` epilogue fusion (A4 in the
DiT report) is also in: `ggml_add_act` gained an `act=3` (mish) branch on CPU and in the Vulkan
shader, and `conv_pos` folds bias + mish into one node. Its effect is below the 0.2% noise floor,
so the numbers above do not change. `tests/test_add_act.cpp` checks all four activations against
a host reference on CPU and on every GPU backend; CI runs it under Mesa lavapipe, which exercises
the Vulkan shader branch (max|diff| 2.4e-7).

## DiT algorithm switches (2026-11-06, same machine/text/seed)

Optional per-request switch. Quality is measured on the DiT mel output (`mel_out`) against the
all-off run — vocoded waveform correlation is dominated by phase and is not a usable fidelity
proxy. The DiT column is the stage time from the server log (`dit`).

| setting | mel relL1 | mel corr | DiT time | verdict |
|---|---|---|---|---|
| all off (upstream behaviour) | 0 | 1.0 | — | default |
| `dit_uncond_reuse=1` | 4.2e-3 | 0.99999 | ~-30% | only for voices with cfg > 0 (v5dev) |
| `dit_uncond_reuse=2` | 2.6e-3 | 0.99999 | ~-28% | safer variant of the above |

Notes: the switch does nothing when cfg = 0 (v5turbo / the `default` voice), because there is no
Uncond branch to reuse then. It stacks with cache-dit.

## CPU backend threading

ggml's barrier spins while waiting, so on SMT CPUs the default hardware_concurrency() (logical
threads) oversubscribes and slows requests down. gsv_server now defaults to physical cores when
--threads is not given (hardware_concurrency()/2 when > 8).

Windows, Xeon E5-2675 v3 (16 cores / 32 threads, 1.8 GHz), turbo-q8, 4 steps, 3.04 s audio:

| threads | total | DiT | cond | AR | vocoder |
|---|---|---|---|---|---|
| 32 (old default) | 39.9 s | 28.8 s | 1.80 s | 2.26 s | 3.07 s |
| 16 (new default) | 35.4 s | 29.0 s | 1.20 s | 1.52 s | 1.62 s |
| 8 | 35.9 s | 28.8 s | 1.15 s | 1.54 s | 1.93 s |

DiT is memory-bound in this configuration and does not improve past 8 threads; the win is in the
AR/cond/vocoder stages. --threads N overrides the default.

## What is not parallelised

- **Pipeline overlap** (AR of segment i+1 while DiT/vocoder runs segment i) is not implemented:
  ggml exposes a single backend queue, so both stages would serialize on the device anyway, and
  on CPU they compete for the same cores. Batches are formed *across* requests instead
  (/batch, AR batcher, request scheduler).
- **Larger conv tiles** in conv_direct_1d were not beneficial: the kernel already uses 6-wide
  time blocks x 16-wide output blocks, and DiT (the dominant CPU stage) is memory-bound, so
  thread count beyond 8 did not move it.

## v2 / v2SoVITS pipeline on two cloud boxes (2026-11-08)

Same text, same 8.3 s reference wav, same fixture, gguf q8 AR + v2sovits + distilled HuBERT frontend
(after the frontend/KV changes in `2e5d753`). Decode length 7.44 s of audio.

| box | cores | ISA | threads | enc_p | flow | dec | synth | AR | **RTF synth** | **RTF total** | peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| shanghai4c4g (EPYC 7K62) | 4 physical | AVX2 | 4 | 155 | 45 | 1488 | 1689 | 1500 | **0.233** | **0.435** | 842 MiB |
| seoul2c2g (EPYC 9754) | 1 physical + SMT2 | AVX2 | 1 | 177 | 108 | 5180 | 5466 | 2683 | **0.735** | **1.095** | 862 MiB |
| seoul2c2g | 1 physical + SMT2 | AVX2 | 2 | 196 | 104 | 4521 | 4823 | 2553 | **0.648** | **0.991** | 862 MiB |

Both boxes produce the same waveform: `cos=0.999917`, `max_abs=1.293e-02`, `peak=0.5471`.

Reading the two rows: one Zen4 core beats one Zen2 core by roughly 1.7x (bare ggml conv_1d 13.54 vs
7.87 GMAC/s, mul_mat GEMV 15.20 vs 10.36 GMAC/s), so per-core the Zen4 box is the better buy - but
four cores still win on wall clock, which is what RTF measures. A 4-core Zen4 host would be expected
to land near RTF 0.18 synth.

**Do not build either of these with `-DGGML_NATIVE=ON` on Zen4.** The same seoul box with an
AVX-512 ggml measured RTF 6.41 (dec 43.6 s, AR 60 s) - an 8-10x slowdown from the flag alone. See the
warning in `build.md` and `tests/bench_ggml_min.c`, which reports a box's ggml health in three
seconds.
