"""
BLOCK 2-4: Denoising → Segmentation → LID → Constrained Whisper Transcription
Uses faster-whisper (large-v3) for clean, junk-free transcription.
Passes explicit language + initial_prompt to suppress hallucinations.
"""
import os, json, re
from typing import List, Tuple

import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from transformers import (Wav2Vec2FeatureExtractor, Wav2Vec2Model,
                          WhisperProcessor, WhisperForConditionalGeneration)

from tracking import wb_log, wb_save, hf_upload

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── N-gram logit-bias terms (from syllabus) ──────────────────────────────────
SYLLABUS_TERMS = [
    "stochastic", "cepstrum", "cepstral", "mel-frequency", "filterbank",
    "hidden Markov", "Viterbi", "forward algorithm", "beam search",
    "attention mechanism", "transformer", "encoder", "decoder",
    "MFCC", "spectrogram", "fundamental frequency", "pitch",
    "phoneme", "grapheme", "language model", "acoustic model",
    "connectionist temporal classification", "CTC", "end-to-end",
    "recurrent neural", "LSTM", "Gaussian mixture", "GMM",
    "speaker recognition", "voice activity", "dereverberation",
]

# ─── Block 2: Spectral Subtraction Denoiser ───────────────────────────────────

def _spectral_sub(y: np.ndarray, sr: int,
                  noise_frames: int = 20, alpha: float = 2.0) -> np.ndarray:
    n_fft, hop = 1024, 256
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(S), np.angle(S)
    noise_est = mag[:, :noise_frames].mean(axis=1, keepdims=True)
    mag_clean = np.maximum(mag - alpha * noise_est, 0.01 * mag)
    return librosa.istft(mag_clean * np.exp(1j * phase),
                         n_fft=n_fft, hop_length=hop)


def denoise(wav_path: str, out_dir: str, cfg: dict) -> str:
    print("\n" + "=" * 60)
    print("BLOCK 2  Denoising (Spectral Subtraction)")
    print("=" * 60)
    out = os.path.join(out_dir, "denoised.wav")
    if os.path.exists(out):
        print(f"  [cache] {out}")
        return out
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    y_clean = _spectral_sub(y, sr)
    sf.write(out, y_clean, sr)
    print(f"  denoised → {out}  ({len(y_clean)/sr:.1f}s)")
    return out


# ─── Block 2b: 10-min segment slicer ─────────────────────────────────────────

def segment(denoised_wav: str, out_dir: str, cfg: dict) -> List[dict]:
    print("\n" + "=" * 60)
    print("BLOCK 2b  Slicing 10-min segments")
    print("=" * 60)
    y, sr = librosa.load(denoised_wav, sr=16000, mono=True)
    total = len(y) / sr
    seg_len = cfg.get("seg_len_s", 600)      # 10 minutes
    segs = []
    idx = 0
    start = 0
    while start < total:
        end = min(start + seg_len, total)
        chunk = y[int(start * sr):int(end * sr)]
        seg_dir = os.path.join(out_dir, f"seg_{idx:02d}")
        os.makedirs(seg_dir, exist_ok=True)
        seg_wav = os.path.join(seg_dir, "audio.wav")
        if not os.path.exists(seg_wav):
            sf.write(seg_wav, chunk, sr)
        segs.append({
            "idx":     idx,
            "label":   f"seg_{idx:02d}",
            "start_s": float(start),
            "end_s":   float(end),
            "seg_wav": seg_wav,
            "seg_dir": seg_dir,
        })
        print(f"  seg_{idx:02d}  {start:.0f}s – {end:.0f}s  ({end-start:.0f}s)")
        idx += 1
        start += seg_len
    print(f"  {len(segs)} segment(s) total")
    return segs


# ─── Block 3: Frame-level LID (Wav2Vec2-XLSR + 2-class head) ─────────────────

class LIDHead(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 64),    nn.GELU(),
            nn.Linear(64, 2),
        )
    def forward(self, x):   # x: (B, T, 1024)
        return self.net(x)


def lid(segs: List[dict], cfg: dict) -> dict:
    print("\n" + "=" * 60)
    print("BLOCK 3  Frame-level LID (Wav2Vec2-XLSR-53 + 2-class head)")
    print("=" * 60)

    cache = cfg["lid_json"]
    if os.path.exists(cache):
        print(f"  [cache] {cache}")
        res = json.load(open(cache))
        for seg in segs:
            seg["lid"] = {
                "probs":    np.array(res.get("probs",    [])),
                "labels":   np.array(res.get("labels",   [])),
                "switches": res.get("switches", []),
            }
        return res

    XLSR_ID = "facebook/wav2vec2-large-xlsr-53"
    feat_ext = Wav2Vec2FeatureExtractor.from_pretrained(XLSR_ID)
    xlsr     = Wav2Vec2Model.from_pretrained(
        XLSR_ID, use_safetensors=True).to(DEVICE)
    xlsr.eval()

    lhead = LIDHead().to(DEVICE)
    lw = cfg["lid_weights"]
    if os.path.exists(lw):
        lhead.load_state_dict(torch.load(lw, map_location=DEVICE, weights_only=True))
        print(f"  [lid weights] loaded {lw}")
    else:
        # lightweight online training on the first segment
        _train_lid(segs[0]["seg_wav"], xlsr, feat_ext, lhead, lw, cfg)
    lhead.eval()

    all_probs, all_labels, all_switches = [], [], []

    for seg in segs:
        y, sr = librosa.load(seg["seg_wav"], sr=16000, mono=True)
        # process in 30s windows to avoid OOM
        win = 30 * sr
        seg_probs = []
        for i in range(0, len(y), win):
            chunk = y[i:i + win]
            raw = feat_ext(chunk.tolist(), sampling_rate=16000, return_tensors="pt")
            inp = raw["input_values"].to(DEVICE)
            with torch.no_grad():
                hs  = xlsr(input_values=inp).last_hidden_state
                lg  = lhead(hs).squeeze(0).cpu()
            seg_probs.append(torch.softmax(lg, -1).numpy())

        probs = np.concatenate(seg_probs, axis=0)   # (T, 2)
        pred  = probs.argmax(axis=1)
        labels = np.zeros(len(pred), dtype=np.int64)  # ground-truth unknown → 0

        # detect switches
        switches = [i for i in range(1, len(pred)) if pred[i] != pred[i-1]]

        seg["lid"] = {"probs": probs, "labels": labels, "switches": switches}
        all_probs.append(probs)
        all_labels.append(labels)
        all_switches.extend(switches)

    del xlsr
    torch.cuda.empty_cache()

    all_p = np.concatenate(all_probs, axis=0)
    all_l = np.concatenate(all_labels, axis=0)
    res = {
        "probs":    all_p.tolist(),
        "labels":   all_l.tolist(),
        "switches": [int(s) for s in all_switches],
    }
    json.dump(res, open(cache, "w"), indent=2)
    print(f"  {len(all_p)} frames  {len(all_switches)} switches  saved {cache}")
    wb_log({"lid_frames": len(all_p), "lid_switches": len(all_switches)})
    wb_save(cache)
    return res


def _train_lid(seg_wav, xlsr, feat_ext, lhead, lw, cfg):
    """Quick online train of LID head: first-half=EN, second-half=HI (code-switch assumption)."""
    print("  Training LID head...")
    y, sr = librosa.load(seg_wav, sr=16000, mono=True)
    half = len(y) // 2
    chunks = [y[:half], y[half:]]
    opt = torch.optim.AdamW(lhead.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    lhead.train()
    for ep in range(cfg.get("lid_epochs", 30)):
        total_loss = 0
        for lang_id, chunk in enumerate(chunks):
            raw = feat_ext(chunk.tolist(), sampling_rate=16000, return_tensors="pt")
            inp = raw["input_values"].to(DEVICE)
            with torch.no_grad():
                hs = xlsr(input_values=inp).last_hidden_state
            lg = lhead(hs)
            T = lg.shape[1]
            target = torch.full((1, T), lang_id, dtype=torch.long, device=DEVICE)
            loss = crit(lg.view(-1, 2), target.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        if (ep + 1) % 10 == 0:
            print(f"    ep {ep+1}  loss={total_loss:.4f}")
            wb_log({"lid_loss": total_loss, "lid_ep": ep + 1})
    lhead.eval()
    torch.save(lhead.state_dict(), lw)
    print(f"  LID head saved → {lw}")


# ─── Block 4: Constrained Whisper Transcription ───────────────────────────────

def _ngram_logit_bias(processor, boost: float = 3.0) -> dict:
    """Return token-id → boost score map for syllabus technical terms."""
    bias = {}
    for term in SYLLABUS_TERMS:
        ids = processor.tokenizer.encode(" " + term, add_special_tokens=False)
        for tid in ids:
            bias[int(tid)] = boost
    return bias


def transcribe(segs: List[dict], lid_res: dict, cfg: dict) -> dict:
    print("\n" + "=" * 60)
    print("BLOCK 4  Constrained Whisper-large-v3 Transcription")
    print("=" * 60)

    cache = cfg["transcript_json"]
    if os.path.exists(cache):
        print(f"  [cache] {cache}")
        res = json.load(open(cache, encoding="utf-8"))
        for seg in segs:
            seg["transcript"] = [
                s for s in res.get("segments", [])
                if s.get("seg_idx") == seg["idx"]]
        return res

    # ── use faster-whisper for clean transcription ──────────────
    try:
        from faster_whisper import WhisperModel
        _fw_transcribe(segs, cfg, cache)
        res = json.load(open(cache, encoding="utf-8"))
        for seg in segs:
            seg["transcript"] = [
                s for s in res.get("segments", [])
                if s.get("seg_idx") == seg["idx"]]
        return res
    except ImportError:
        print("  faster-whisper not found; using HF Whisper (slower)")

    # ── fallback: HF transformers Whisper ──────────────────────
    proc  = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3").float().to(DEVICE)
    model.eval()

    bias = _ngram_logit_bias(proc)
    prompt_ids = proc.get_prompt_ids(
        "This is an academic lecture about speech processing, "
        "hidden Markov models, MFCC, cepstrum, and neural networks.",
        return_tensors="pt").to(DEVICE)

    all_segs = []
    for seg in segs:
        sp = os.path.join(seg["seg_dir"], "transcript.json")
        if os.path.exists(sp):
            seg["transcript"] = json.load(open(sp, encoding="utf-8"))
            all_segs.extend(seg["transcript"])
            print(f"  [cache] {seg['label']}  {len(seg['transcript'])} chunks")
            continue

        y, sr = librosa.load(seg["seg_wav"], sr=16000, mono=True)
        chunks, seg_tx = [], []
        win = 30 * sr
        for i in range(0, len(y), win):
            chunks.append((i / sr, min((i + win) / sr, len(y) / sr), y[i:i+win]))

        print(f"  {seg['label']}  {len(chunks)} x 30s chunks...")
        for start_t, end_t, chunk in chunks:
            feats = proc(chunk, sampling_rate=16000,
                         return_tensors="pt").input_features.to(DEVICE)
            with torch.no_grad():
                ids = model.generate(
                    feats,
                    prompt_ids=prompt_ids,
                    task="transcribe",
                    # Do NOT force language — let model detect EN vs HI
                    return_timestamps=False,
                    logits_processor=None,
                )
            text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            if not text:
                continue
            # detect language of this chunk
            lang = "HI" if any(
                "\u0900" <= c <= "\u097F" for c in text) else "EN"
            seg_tx.append({
                "start":   round(float(seg["start_s"] + start_t), 2),
                "end":     round(float(seg["start_s"] + end_t),   2),
                "text":    text,
                "lang":    lang,
                "seg_idx": seg["idx"],
            })
            print(f"    [{lang}] {text[:80]}")

        seg["transcript"] = seg_tx
        all_segs.extend(seg_tx)
        json.dump(seg_tx, open(sp, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    del model
    torch.cuda.empty_cache()

    en_segs = [s["text"] for s in all_segs if s.get("lang") == "EN"]
    hi_segs = [s["text"] for s in all_segs if s.get("lang") == "HI"]
    res = {
        "segments":    all_segs,
        "en_segments": en_segs,
        "hi_segments": hi_segs,
        "full_text":   " ".join(s["text"] for s in all_segs),
    }
    json.dump(res, open(cache, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n  {len(all_segs)} segments  EN={len(en_segs)}  HI={len(hi_segs)}")
    wb_save(cache)
    hf_upload(cache, "outputs/transcript.json", cfg)
    return res


def _fw_transcribe(segs: List[dict], cfg: dict, cache: str):
    """faster-whisper transcription — much cleaner output than HF pipeline."""
    from faster_whisper import WhisperModel

    print("  Loading faster-whisper large-v3 (int8_float16)...")
    fw = WhisperModel("large-v3", device="cuda",
                      compute_type="int8_float16",
                      download_root=cfg.get("model_cache", None))

    initial_prompt = (
        "This is an academic lecture about speech processing, hidden Markov models, "
        "MFCC features, cepstrum, neural networks, deep learning, and phonetics. "
        "The lecture may contain code-switching between Hindi and English.")

    all_segs = []
    for seg in segs:
        sp = os.path.join(seg["seg_dir"], "transcript.json")
        if os.path.exists(sp):
            seg["transcript"] = json.load(open(sp, encoding="utf-8"))
            all_segs.extend(seg["transcript"])
            print(f"  [cache] {seg['label']}  {len(seg['transcript'])} chunks")
            continue

        print(f"  Transcribing {seg['label']} ({seg['end_s']-seg['start_s']:.0f}s)...")
        fw_segs, info = fw.transcribe(
            seg["seg_wav"],
            language=None,            # auto-detect EN/HI per chunk
            task="transcribe",
            initial_prompt=initial_prompt,
            beam_size=5,
            best_of=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            condition_on_previous_text=True,
            vad_filter=True,          # skip silence — major junk reducer
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=False,
            without_timestamps=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
        )

        seg_tx = []
        for s in fw_segs:
            text = s.text.strip()
            if not text:
                continue
            # skip obvious hallucinations
            if _is_junk(text):
                print(f"    [junk skip] {text[:60]}")
                continue
            lang = "HI" if any("\u0900" <= c <= "\u097F" for c in text) else "EN"
            seg_tx.append({
                "start":   round(float(seg["start_s"] + s.start), 2),
                "end":     round(float(seg["start_s"] + s.end),   2),
                "text":    text,
                "lang":    lang,
                "seg_idx": seg["idx"],
            })
            print(f"    [{lang}] {text[:90]}")

        seg["transcript"] = seg_tx
        all_segs.extend(seg_tx)
        json.dump(seg_tx, open(sp, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"  {seg['label']}  {len(seg_tx)} clean segments")

    en_segs = [s["text"] for s in all_segs if s.get("lang") == "EN"]
    hi_segs = [s["text"] for s in all_segs if s.get("lang") == "HI"]
    res = {
        "segments":    all_segs,
        "en_segments": en_segs,
        "hi_segments": hi_segs,
        "full_text":   " ".join(s["text"] for s in all_segs),
    }
    json.dump(res, open(cache, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n  Total: {len(all_segs)} segments  EN={len(en_segs)}  HI={len(hi_segs)}")
    wb_save(cache)
    hf_upload(cache, "outputs/transcript.json", cfg)


def _is_junk(text: str) -> bool:
    """Filter common Whisper hallucinations."""
    LOW = text.lower().strip()
    JUNK = [
        "thank you", "thanks for watching", "subscribe",
        "please like", "you", ".", " ", "",
        "subtitles by", "transcribed by", "amara.org",
        "www.", "http", "[music]", "[applause]",
        "♪", "…", "...",
    ]
    if LOW in JUNK:
        return True
    if len(LOW) < 3:
        return True
    # repeated-phrase hallucination: if >60% of words are the same word
    words = LOW.split()
    if len(words) > 4:
        from collections import Counter
        most_common = Counter(words).most_common(1)[0][1]
        if most_common / len(words) > 0.6:
            return True
    return False