# bf16 direct-conv kernel for the dec stage (ggml-cpu patch)

The kernel that makes the vocoder 1.68× faster lives in the vendored `llama.cpp`, which is **not**
version-controlled here. This directory holds the export so it survives a rebuild of that tree.

## Files

| file | what it is |
|---|---|
| `ops_conv_bf16.cpp` | the kernel, new file → `llama.cpp/ggml/src/ggml-cpu/` |
| `ops-cpu.patch` | diff of `ops.cpp`, `ops.h`, `CMakeLists.txt` (call site, struct decl, source list entry) |

## Why a separate translation unit

`vdpbf16ps` needs AVX512BF16. Building all of ggml-cpu with AVX-512 makes the AR decode **2.2× slower**,
and the dec stage is only ~57% of the pipeline — so that trade is a net loss. The kernel gets its ISA
from a `#pragma GCC target("avx512f,avx512vl,avx512bf16,f16c,fma")` region around one function, and
`gsv_conv_bf16_available()` gates it on `__builtin_cpu_supports("avx512bf16")` at runtime. Everything
else stays AVX2.

## How it gets triggered

`ggml_compute_forward_conv_direct_1d` takes the bf16 path when `src[0]->type == GGML_TYPE_BF16`. Nothing
in the model loader produced bf16 weights, so the kernel was unreachable until `GSV_BF16_CONV=1` was
added to `gsv_vocoder_ggml.cpp` (that part lives in the main repo): it converts 3D `.weight` tensors
f32→bf16 **in place** (bf16 is half the bytes, so the f32 buffer already allocated holds it and the read
pointer never overtakes the write pointer), and fixes `nb[0..3]`. Biases and norm weights stay f32.

The graph planner already sized the workspace for the bf16 path
(`ggml/src/ggml-cpu/ggml-cpu.c`, `case GGML_OP_CONV_DIRECT_1D`), so no change was needed there.

## Measured

4.12 s dec fixture, threads=2, one binary. `perf stat` on an idle host (`load 0.01`), 3 runs each.
Cycles are the metric that matters here — this VM's wall clock varies up to 11x between runs of the
same command (shared VM, `load average` peaked at 31), so treat any single wall-clock reading as noise.

| weights | cycles | instructions | dec | wav cos |
|---|---|---|---|---|
| f32 | 18.14 G | 30.80 G | 2588 ms | 0.999917 |
| bf16, first working version | 10.97 G | 14.58 G | 1539 ms | 0.999792 |
| **bf16, current** | **10.10 G** | 14.13 G | **1373 ms** | 0.999791 |

1.88x wall / 1.80x cycles against f32. IPC falls from 1.70 (f32) to 1.33 (bf16): the kernel issues
fewer, heavier instructions (32 MAC each) and becomes more load-bound, which is why the win is not 2x.

### Where the time goes (`GSV_DEC_T=1`)

```
pack = 1.0%   im2col = 15.2%   kernel = 83.8%
fma/layer = 10.21 M   kernel_eff = 100.5%
```

`kernel_eff` is `dpbf16ps count * 2 cycles` over the measured kernel time, so **100.5% means the
vdpbf16ps issue port is saturated**. The micro-kernel is done; three further attempts to speed it up
are recorded below because each one looked plausible and none worked.

The pack is 1.0% — phase 1a repacks the weights on every call, which is not worth caching.

## What was tried on the kernel and rejected

| attempt | result |
|---|---|
| 32 oc per pass (12 broadcasts feed 24 dpbf16ps instead of 12) | **kept** — put 60% of the samples on `vpbroadcastd` down to 18%, 10.97 → 10.9 G cycles |
| software prefetch of the next Xb row, distance 4 | **kept** — 10.9 → 10.47 G cycles; a predicted branch with 4 prefetches per iteration beat the L2 latency it was hiding |
| tap/oc sweep: (taps, oc-blocks) = (8,2) (8,3) (6,4) (16,1) (12,2) | (8,2) won by 1.4%; (6,3) and (8,2) at 6/8 taps cost 16.5 G — register spills |
| unroll the kp2 loop x2 | 1524 ms, slower — the loop control was not the limiter, the samples on `add` were stalls |
| lanes = t instead of oc (contiguous 512-bit activation loads, vector epilogue) | 10.85 G / 1515 ms — 19% fewer *instructions* but IPC fell 1.24 → 1.00, a wash |
| fuse the im2col into the t-tile loop so the tile stays in L1 | 1608 ms, 16% worse — per-tile packing turns the activation reads into strided gathers and the hardware prefetcher stops working; the whole-row pack streams better |

The kernel is memory-shaped, not instruction-shaped: every reordering that reduced instruction count
paid for it with stalls, and the one thing that helped (`vpbroadcastd` amortized over 2x the oc) was
about load-port pressure, not arithmetic.

## What did not work, recorded so it is not retried

* **oc-outer loop order** — every oc block re-reads the whole im2col buffer from DRAM (K times the
  activation, 371 MB for the largest dec layer). The t-block-outer order is what produced the win;
  keeping the activation tile in L1 and streaming weights from L2 was the whole story.
* **AVX-512 for the whole library** — 2.2× slower on the AR decode at every batch size.
