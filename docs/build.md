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
git -C llama.cpp apply ../patches/0002-ggml-conv-direct-1d-vectorised-epilogue.patch   # optional, see below
```

> The patch is what makes `ggml_conv_1d`, `ggml_conv_transpose_1d`, `ggml_pad_reflect_1d` and the
> sub-pixel path available. Without it the engine does not compile.

> `0002` is optional and pure performance: it vectorises the `conv_direct_1d` epilogue (the old one
> did a 6x16 -> 16x6 register transpose through 96 scalar stores) and folds bias+leaky in. It is
> bit-identical to the unpatched kernel at the pipeline level (same WAV md5), so it cannot change
> output; it helps most on the small-channel tail shapes (K=3: 42% -> 75% of the AVX2-FMA peak).
> If you skip it, nothing else breaks.

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
