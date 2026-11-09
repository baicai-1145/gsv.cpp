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

4.12 s dec fixture, threads=2, one binary, same build:

| weights | dec | wav cos vs reference |
|---|---|---|
| f32 | 2587.7 ms | 0.999917 |
| **bf16** | **1538.9 ms** (1.68×) | 0.999792 |

`z cos = 1.000000` — the DiT stage is unaffected, only the vocoder's weights are rounded.

Caveat: this host's wall-clock varies up to 11× between runs of the same command (a shared VM; `load
average` peaked at 31), so the timing above is one clean sample, not a robust number. Re-measure with
`perf stat -e instructions` before quoting it.

## What did not work, recorded so it is not retried

* **oc-outer loop order** — every oc block re-reads the whole im2col buffer from DRAM (K times the
  activation, 371 MB for the largest dec layer). The t-block-outer order is what produced the win;
  keeping the activation tile in L1 and streaming weights from L2 was the whole story.
* **AVX-512 for the whole library** — 2.2× slower on the AR decode at every batch size.
