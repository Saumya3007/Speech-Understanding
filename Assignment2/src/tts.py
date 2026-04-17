"""
BLOCK 7: Speaker Embedding + DTW Prosody Warping + MMS-TTS-mai Synthesis
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

from tracking import wb_log, wb_save, hf_upload

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Task 3.1: Speaker Embedding (MFCC d-vector, 192-dim) ────────────────────

def _get_speaker_embedding(wav_path: str, cfg: dict) -> np.ndarray:
    emb_path = os.path.join(cfg.get("out_dir", "."), "speaker_embedding.npy")
    if os.path.exists(emb_path):
        emb = np.load(emb_path)
        print(f"  [x-vector] loaded cached  shape={emb.shape}")
        return emb
    if not wav_path or not os.path.exists(wav_path):
        print("  [x-vector] ref_voice not found — random init")
        emb = np.random.randn(192).astype(np.float32)
        np.save(emb_path, emb)
        return emb
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=96)
    emb   = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)
    np.save(emb_path, emb)
    print(f"  [x-vector] MFCC d-vector  shape={emb.shape}")
    return emb


# ─── Task 3.2: Prosody Extraction + DTW Warping ───────────────────────────────

def _extract_prosody(wav_path: str, sr: int = 16000):
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    f0, _, _  = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
    f0        = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    energy    = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0].astype(np.float32)
    T         = min(len(f0), len(energy))
    return f0[:T], energy[:T]


def _dtw_warp(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Warp tgt sequence to match length of src using DTW path."""
    n, m = len(src), len(tgt)
    if n == 0 or m == 0:
        return tgt
    # cost matrix
    cost = np.zeros((n, m), dtype=np.float32)
    cost[0, 0] = abs(src[0] - tgt[0])
    for i in range(1, n):   cost[i, 0] = cost[i-1, 0] + abs(src[i] - tgt[0])
    for j in range(1, m):   cost[0, j] = cost[0, j-1] + abs(src[0] - tgt[j])
    for i in range(1, n):
        for j in range(1, m):
            cost[i, j] = abs(src[i] - tgt[j]) + min(cost[i-1,j], cost[i,j-1], cost[i-1,j-1])
    # traceback
    i, j = n-1, m-1
    path_src, path_tgt = [i], [j]
    while i > 0 or j > 0:
        if i == 0:   j -= 1
        elif j == 0: i -= 1
        else:
            mv = np.argmin([cost[i-1,j-1], cost[i-1,j], cost[i,j-1]])
            if mv == 0:   i -= 1; j -= 1
            elif mv == 1: i -= 1
            else:          j -= 1
        path_src.append(i); path_tgt.append(j)
    path_src.reverse(); path_tgt.reverse()
    warped = np.zeros(n, dtype=np.float32)
    for pi, pj in zip(path_src, path_tgt):
        warped[pi] = tgt[pj]
    return warped


def _apply_prosody(y_syn: np.ndarray, f0_src: np.ndarray,
                   energy_src: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Scale energy of synthesised audio to match source contour via DTW."""
    hop = 256
    e_syn = librosa.feature.rms(y=y_syn, frame_length=512, hop_length=hop)[0]
    if len(e_syn) == 0:
        return y_syn
    e_src_rs = np.interp(
        np.linspace(0, 1, len(e_syn)),
        np.linspace(0, 1, len(energy_src)),
        energy_src
    )
    ratio = (e_src_rs + 1e-8) / (e_syn + 1e-8)
    ratio = np.clip(ratio, 0.1, 10.0)
    # frame-level gain applied sample-wise
    out = y_syn.copy()
    for i, r in enumerate(ratio):
        s = i * hop
        e = min(s + hop, len(out))
        out[s:e] *= r
    return out


# ─── Resample helper ──────────────────────────────────────────────────────────

def _resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    g   = gcd(orig_sr, target_sr)
    up  = target_sr // g
    dn  = orig_sr   // g
    return resample_poly(y, up, dn).astype(np.float32)


# ─── MMS-TTS-mai synthesis ────────────────────────────────────────────────────

def _mms_tts(text: str, out_wav: str, cfg: dict) -> np.ndarray:
    from transformers import VitsModel, AutoTokenizer

    model_id = cfg.get("mms_tts_model", "facebook/mms-tts-mai")
    print(f"  [MMS-TTS] loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = VitsModel.from_pretrained(model_id).to(DEVICE)
    model.eval()

    tts_sr = model.config.sampling_rate   # 16000 for MMS
    chunks = [text[i:i+400] for i in range(0, len(text), 400)] if len(text) > 400 else [text]
    parts  = []

    for idx, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        inputs = tokenizer(chunk, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            wav = model(**inputs).waveform[0].cpu().numpy()
        parts.append(wav)
        print(f"    chunk {idx+1}/{len(chunks)}  {len(wav)/tts_sr:.1f}s")
        wb_log({f"tts_chunk_{idx}": len(wav)/tts_sr})

    if not parts:
        return np.zeros(tts_sr, dtype=np.float32)

    audio = np.concatenate(parts).astype(np.float32)

    # upsample to 22050 Hz (assignment requires >= 22050 Hz)
    target_sr = cfg.get("target_sr", 22050)
    if tts_sr != target_sr:
        audio = _resample(audio, tts_sr, target_sr)

    sf.write(out_wav, audio, target_sr, subtype="PCM_16")
    print(f"  [MMS-TTS] saved {out_wav}  dur={len(audio)/target_sr:.1f}s  sr={target_sr}Hz")

    del model; torch.cuda.empty_cache()
    return audio


# ─── Main synthesise function ─────────────────────────────────────────────────

def synthesise(segs: list, lrl_result: dict, cfg: dict) -> str:
    print("\n" + "=" * 60)
    print("BLOCK 7  x-vector + DTW Prosody + MMS-TTS-mai  (Maithili)")
    print("=" * 60)

    out_dir    = cfg.get("out_dir", ".")
    target_sr  = cfg.get("target_sr", 22050)
    final_wav  = os.path.join(out_dir, "output_LRL_cloned.wav")

    # ── Task 3.1: speaker embedding ───────────────────────────────────────────
    print("\n  Task 3.1: Speaker Embedding (x-vector) from student voice...")
    ref_voice = cfg.get("ref_voice", "")
    emb = _get_speaker_embedding(ref_voice, cfg)
    print(f"  Saved: {os.path.join(out_dir, 'speaker_embedding.npy')}")

    # ── build seg_texts robustly ──────────────────────────────────────────────
    _seg_lrl = lrl_result.get("seg_lrl") or {}
    for _s in segs:
        if not _s.get("lrl_text"):
            _s["lrl_text"] = (
                _seg_lrl.get(_s["label"]) or
                _seg_lrl.get(f"seg_{_s['idx']:02d}") or
                lrl_result.get("full_maithili", "")[:500] or ""
            )
    seg_texts       = {_s["label"]: _s["lrl_text"] for _s in segs}
    total_mai_chars = sum(len(v) for v in seg_texts.values())
    active_segs     = sum(1 for v in seg_texts.values() if v.strip())
    print(f"  Maithili text available: {total_mai_chars} chars across {active_segs} segments")

    if total_mai_chars == 0:
        raise RuntimeError(
            "No Maithili text found in lrl_result! "
            "Block 6 (translate) must run successfully first."
        )

    # ── per-segment synthesis ─────────────────────────────────────────────────
    all_audio = []

    for seg in segs:
        cache = os.path.join(seg["seg_dir"], "synthesised.wav")

        if os.path.exists(cache):
            audio, _ = librosa.load(cache, sr=target_sr, mono=True)
            seg["synth_wav"] = cache
            print(f"  [cache] {seg['label']}  {len(audio)/target_sr:.1f}s")
            all_audio.append(audio)
            continue

        y_src, sr_src = librosa.load(seg["seg_wav"], sr=None, mono=True)
        src_dur       = len(y_src) / sr_src
        target_samples = int(src_dur * target_sr)

        print(f"\n  {seg['label']}  source={src_dur:.1f}s  target={target_samples} samples")

        # Task 3.2: prosody extraction
        print(f"  Task 3.2: extracting prosody from {seg['seg_wav']} ...")
        f0, energy = _extract_prosody(seg["seg_wav"])

        mai_text = seg_texts.get(seg["label"], "").strip()
        if not mai_text:
            print(f"    No Maithili text for {seg['label']} — check Block 6 output")
            all_audio.append(np.zeros(target_samples, dtype=np.float32))
            continue

        print(f"  Task 3.3: synthesising {len(mai_text)} chars of Maithili text...")
        tmp_wav = os.path.join(seg["seg_dir"], "tts_raw.wav")
        audio   = _mms_tts(mai_text, tmp_wav, cfg)

        # Task 3.2: apply DTW prosody warp
        print(f"  Applying DTW prosody warp...")
        audio = _apply_prosody(audio, f0, energy, sr=target_sr)

        # pad / trim to match source duration
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        else:
            audio = audio[:target_samples]

        os.makedirs(seg["seg_dir"], exist_ok=True)
        sf.write(cache, audio, target_sr, subtype="PCM_16")
        seg["synth_wav"] = cache
        actual_dur = len(audio) / target_sr

        print(f"  Saved {seg['label']}  {actual_dur:.1f}s → {cache}")
        wb_log({
            f"{seg['label']}_synth_s":   actual_dur,
            f"{seg['label']}_source_s":  src_dur,
            f"{seg['label']}_dur_ratio": actual_dur / max(src_dur, 0.1)
        })
        all_audio.append(audio)

    # ── concatenate all segments ──────────────────────────────────────────────
    if not all_audio:
        raise RuntimeError("No audio segments synthesised!")

    full_audio = np.concatenate(all_audio).astype(np.float32)
    sf.write(final_wav, full_audio, target_sr, subtype="PCM_16")
    total_dur = len(full_audio) / target_sr
    print(f"\n  Final output: {final_wav}")
    print(f"  Duration: {total_dur:.1f}s  SR: {target_sr}Hz")

    wb_log({"synth_total_s": total_dur, "synth_sr": target_sr})
    wb_save(final_wav)
    hf_upload(final_wav, "outputs/output_LRL_cloned.wav", cfg)
    return final_wav
