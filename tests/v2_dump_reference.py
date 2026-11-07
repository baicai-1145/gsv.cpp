#!/usr/bin/env python3
"""GPT-SoVITS v2 SoVITS 参考张量导出（对拍 fixture + AR 基准输入）。

在装有 GPT-SoVITS-CPUFast 及其 venv 的机器上跑:

    cd ~/GPT-SoVITS-CPUFast && .venv/bin/python <repo>/tests/v2_dump_reference.py \
        --repo ~/GPT-SoVITS-CPUFast --out /tmp/v2fix \
        --ref ref.wav --ref-text "..." --text "..."

产出的 fixture 供 tests/probe_v2sovits.cpp 逐段对拍:
    ssl.f32 / codes_ref.i32 / spec704.f32 / ge_ref.f32 / phones.i32
    m_ref.f32 / logs_ref.f32 / quant_ref.f32 / zp_ref.f32 / z_ref.f32 / wav_ref.f32
    prompt_codes.i32 / prompt_phones.i32 / prompt_bert.f32
    bert.f32（目标文本）
全部是 raw little-endian，torch 布局 [C,T]（flat = c*T + t）。

显存/内存: 顺序加载 cnhubert -> BERT -> vits，每一步后释放；CPU 上大约 2GB 峰值。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch


def wf32(path, arr):
    a = np.ascontiguousarray(np.asarray(arr, dtype="<f4"))
    a.tofile(path)
    print(f"  {os.path.basename(path):<18} {list(a.shape)}  {'%.3e' % np.abs(a).max()}")


def wi32(path, arr):
    a = np.ascontiguousarray(np.asarray(arr, dtype="<i4"))
    a.tofile(path)
    print(f"  {os.path.basename(path):<18} {list(a.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.expanduser("~/GPT-SoVITS-CPUFast"))
    ap.add_argument("--out", default="/tmp/v2fix")
    ap.add_argument("--ref", default="ref.wav")
    ap.add_argument("--ref-text", default="")
    ap.add_argument("--text", default="那天我和朋友一起去看了那场期待已久的电影，散场以后我们在路边的小店坐了很久。")
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--noise-scale", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.repo))
    gsv = os.path.join(repo, "GPT_SoVITS")
    sys.path.insert(0, gsv)
    sys.path.insert(0, repo)
    os.makedirs(args.out, exist_ok=True)
    ref = args.ref if os.path.isabs(args.ref) else os.path.join(repo, args.ref)
    print(f"repo={repo}\nref={ref}\nout={args.out}")

    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, (os.cpu_count() or 4)))

    from module.mel_processing import spectrogram_torch

    # ---------------------------------------------------------------- 1) HuBERT
    # 若 ssl.f32 已存在（由 probe_v2sovits --dump-inputs 产出），直接复用 —— 这样两侧看到的
    # 是**同一份** ssl，比较就与重采样实现无关（本 fork 的 av 重采样 != 上游 resampy）。
    from tools.audio_utils import load_audio_tensor, resample_audio_tensor

    ssl_path = os.path.join(args.out, "ssl.f32")
    raw, raw_sr = load_audio_tensor(ref)
    if raw.dim() == 2:
        raw = raw.mean(0, keepdim=True)
    if os.path.exists(ssl_path):
        n = os.path.getsize(ssl_path) // 4
        ssl = torch.from_numpy(np.fromfile(ssl_path, dtype="<f4").copy())
        print(f"[1] 复用已有 ssl.f32: {n} floats")
    else:
        from feature_extractor.cnhubert import CNHubert

        hu = CNHubert(os.path.join(gsv, "pretrained_models/chinese-hubert-base")).eval()
        w16 = resample_audio_tensor(raw, raw_sr, 16000).flatten().numpy()
        # repo (inference_webui.py) 的尾巴用的是 **目标采样率** * 0.3 个零点，再拼到 16 kHz 波形上
        # —— v2 的 sampling_rate=32000，所以是 9600 点 = 16 kHz 下的 0.6 s。gsv_engine 照此实现。
        ph16 = np.concatenate([w16, np.zeros(int(32000 * 0.3), dtype=np.float32)])
        with torch.no_grad():
            out = hu.model(torch.from_numpy(ph16).float().unsqueeze(0))["last_hidden_state"]
        ssl = out[0].transpose(0, 1).contiguous()                                        # [768,T]
        print(f"[1] hubert: {tuple(ssl.shape)}")
        wf32(ssl_path, ssl.numpy())
        del hu
    Tssl = ssl.numel() // 768
    ssl = ssl.view(768, Tssl)

    # ---------------------------------------------------------------- 2) 参考 spec704
    spec_path = os.path.join(args.out, "spec704.f32")
    if os.path.exists(spec_path):
        n = os.path.getsize(spec_path) // 4
        spec704 = torch.from_numpy(np.fromfile(spec_path, dtype="<f4").copy())
        print(f"[2] 复用已有 spec704.f32: {n} floats")
    else:
        w32 = resample_audio_tensor(raw, raw_sr, 32000).flatten()
        maxx = w32.abs().max()
        if maxx > 1:
            w32 = w32 / min(2, maxx)
        spec = spectrogram_torch(w32.unsqueeze(0), 2048, 32000, 640, 2048, center=False)
        spec704 = spec[0, :704].contiguous()
        print(f"[2] ref spec: {tuple(spec.shape)} -> 704 slice {tuple(spec704.shape)}")
        wf32(spec_path, spec704.numpy())
    Ts = spec704.numel() // 704
    spec704 = spec704.view(704, Ts)

    # ---------------------------------------------------------------- 3) 文本前端
    from TTS_infer_pack.TextPreprocessor import TextPreprocessor
    from text.chinese_bert import load_model as load_chinese_bert_model
    from text.chinese_bert import load_tokenizer as load_chinese_bert_tokenizer

    bert_dir = os.path.join(gsv, "pretrained_models/chinese-roberta-wwm-ext-large")
    tok = load_chinese_bert_tokenizer(bert_dir)
    bmodel = load_chinese_bert_model(bert_dir).eval()
    tp = TextPreprocessor(bmodel, tok, torch.device("cpu"))
    # get_phones_and_bert 直接返回 (phones, bert[1024,P], norm_text)
    phones_t, bert_t, norm_t = tp.get_phones_and_bert(args.text, args.lang, "v2")
    print(f"[3] target text: phones={len(phones_t)} bert={tuple(bert_t.shape)}")
    wi32(os.path.join(args.out, "phones.i32"), phones_t)
    wf32(os.path.join(args.out, "bert.f32"), bert_t.numpy())
    ref_text = args.ref_text or args.text
    phones_p, bert_p, norm_p = tp.get_phones_and_bert(ref_text, args.lang, "v2")
    print(f"    prompt text: phones={len(phones_p)} bert={tuple(bert_p.shape)}")
    wi32(os.path.join(args.out, "prompt_phones.i32"), phones_p)
    wf32(os.path.join(args.out, "prompt_bert.f32"), bert_p.numpy())
    del tp, bmodel

    # ---------------------------------------------------------------- 4) v2 SoVITS
    from module.models import SynthesizerTrn

    s2 = os.path.join(gsv, "pretrained_models/gsv-v2final-pretrained/s2G2333k.pth")
    ck = torch.load(s2, map_location="cpu")
    if isinstance(ck, dict) and "weight" in ck:
        sd, hps = ck["weight"], ck["config"]
    else:
        sd, hps = ck, json.load(open(os.path.join(gsv, "configs/s2.json")))
    hps["model"]["semantic_frame_rate"] = "25hz"
    hps["model"].setdefault("version", "v2")
    kw = dict(hps["model"])
    kw.pop("version", None)
    vits = SynthesizerTrn(hps["data"]["filter_length"] // 2 + 1,
                          hps["train"]["segment_size"] // hps["data"]["hop_length"],
                          n_speakers=hps["data"]["n_speakers"], **kw)
    print("  load_state_dict:", vits.load_state_dict(sd, strict=False))
    if hasattr(vits, "enc_q"):
        del vits.enc_q
    if hasattr(vits, "dec"):
        vits.dec.remove_weight_norm()
    vits = vits.eval()
    gin = hps["model"]["gin_channels"]
    print(f"[4] v2 SoVITS: gin={gin} hp={hps['data']['sampling_rate']}Hz "
          f"is_v2pro={getattr(vits, 'is_v2pro', None)}")

    # 4a) 参考侧语义码 (extract_latent = ssl_proj k2s2 + VQ)
    sem = vits.extract_latent(ssl.unsqueeze(0))                   # [1,1,Tc]
    codes = sem[0, 0].numpy().astype(np.int32)
    Tc = codes.shape[0]
    print(f"     sem codes: {Tc} (ssl {ssl.shape[1]} 帧)")
    wi32(os.path.join(args.out, "codes_ref.i32"), codes)
    wi32(os.path.join(args.out, "prompt_codes.i32"), codes)

    # 4b) ge
    spe = spec704.unsqueeze(0)
    mask = torch.ones(1, 1, spe.shape[-1])
    ge = vits.ref_enc(spe * mask, mask)                          # [1,gin,1]
    print(f"     ge: {tuple(ge.shape)}")
    wf32(os.path.join(args.out, "ge_ref.f32"), ge[0, :, 0].numpy())

    # 4c) enc_p —— ge 按 repo 的 [B, gin, 1] 传入（MRTE/flow/dec 内部广播到 T）
    quantized = vits.quantizer.decode(torch.from_numpy(codes).view(1, 1, -1))       # [1,768,Tc]
    quantized = torch.nn.functional.interpolate(quantized, size=2 * Tc, mode="nearest")
    print(f"     quantized: {tuple(quantized.shape)}")
    wf32(os.path.join(args.out, "quant_ref.f32"), quantized[0].numpy())
    phone_t = torch.LongTensor(phones_t).unsqueeze(0)
    ret = vits.enc_p(quantized, torch.LongTensor([2 * Tc]), phone_t,
                     torch.LongTensor([len(phones_t)]), ge)
    x, m_p, logs_p, y_mask = ret[0], ret[1], ret[2], ret[3]
    print(f"     enc_p: m_p {tuple(m_p.shape)} logs_p {tuple(logs_p.shape)}")
    wf32(os.path.join(args.out, "m_ref.f32"), m_p[0].numpy())
    wf32(os.path.join(args.out, "logs_ref.f32"), logs_p[0].numpy())

    # 4d) z_p / flow / dec（固定噪声，cpp 侧读同一份 zp_ref）
    g = torch.Generator().manual_seed(args.seed)
    eps = torch.randn(m_p.shape, generator=g)
    z_p = m_p + eps * torch.exp(logs_p) * args.noise_scale
    wf32(os.path.join(args.out, "zp_ref.f32"), z_p[0].numpy())
    z = vits.flow(z_p, y_mask, g=ge, reverse=True)
    wf32(os.path.join(args.out, "z_ref.f32"), z[0].numpy())
    wav = vits.dec(z, g=ge)
    print(f"     dec: wav {tuple(wav.shape)} (期望 {2 * Tc * 640} 采样)")
    wf32(os.path.join(args.out, "wav_ref.f32"), wav[0, 0].numpy())

    meta = dict(codes=Tc, ssl_frames=int(ssl.shape[1]), T=int(m_p.shape[-1]),
                samples=int(wav.shape[-1]), sample_rate=int(hps["data"]["sampling_rate"]),
                upsample=int(wav.shape[-1]) // int(m_p.shape[-1]), gin=int(gin),
                noise_scale=args.noise_scale, seed=args.seed, text=args.text,
                ref_text=ref_text)
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), ensure_ascii=False, indent=1)
    print("[done]", json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
