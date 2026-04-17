"""
ingest_denoise.py — Block 1 (audio ingestion) + Block 2 (denoising + slicing)
Task 1.3: Boll (1979) Spectral Subtraction
"""
import math, os, pathlib, shutil, subprocess
from typing import List

import numpy as np
import soundfile as sf

from tracking import wb_log


# ── Block 1 ───────────────────────────────────────────────────────────────
def ingest(cfg: dict) -> str:
    print("\n" + "═"*60)
    print("BLOCK 1  Audio Ingestion")
    print("═"*60)
    raw = cfg["raw_wav"]
    if os.path.exists(raw):
        print(f"  [cache] {raw}  ({sf.info(raw).duration:.1f}s)")
        return raw
    if cfg["local_file"] and os.path.exists(cfg["local_file"]):
        _ffmpeg(cfg["local_file"], raw)
    elif cfg["youtube_url"]:
        _yt_dl(cfg["youtube_url"], cfg["data_dir"], raw)
    else:
        raise ValueError("Provide LOCAL_FILE or YOUTUBE_URL in .env")
    info = sf.info(raw)
    print(f"  ✓ {raw}  {info.duration:.1f}s @ {info.samplerate}Hz")
    wb_log({"raw_dur_s": info.duration})
    return raw


def _ffmpeg(src: str, dst: str):
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", dst],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr}")


def _yt_dl(url: str, data_dir: str, dst: str):
    tmpl = os.path.join(data_dir, "yt_dl.%(ext)s")
    r = subprocess.run([
        "yt-dlp", "--format", "bestaudio", "--output", tmpl,
        "--extract-audio", "--audio-format", "wav", "--audio-quality", "0",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1", url,
    ], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"yt-dlp:\n{r.stderr}")
    cands = list(pathlib.Path(data_dir).glob("yt_dl.*"))
    if not cands:
        raise FileNotFoundError("yt-dlp produced no output")
    tmp = str(cands[0])
    shutil.move(tmp, dst) if tmp.endswith(".wav") else (_ffmpeg(tmp, dst), os.remove(tmp))


# ── Spectral Subtraction (Task 1.3) ─────────────────────────────────────
def spectral_sub(audio: np.ndarray, sr: int,
                 noise_frames: int = 20,
                 alpha: float = 2.0,
                 beta: float  = 0.01) -> np.ndarray:
    """
    Boll (1979): |X_clean|^2 = max(|X|^2 - α|N|^2,  β|N|^2)
    Noise PSD estimated from first `noise_frames` frames (pre-lecture silence).
    """
    n_fft = int(0.025 * sr)
    hop   = int(0.010 * sr)
    win   = np.hanning(n_fft)
    n_fr  = 1 + (len(audio) - n_fft) // hop
    idx   = np.clip(
        np.arange(n_fft)[None, :] + np.arange(n_fr)[:, None] * hop,
        0, len(audio) - 1,
    )
    spec        = np.fft.rfft(audio[idx] * win, n=n_fft)
    mag, ph     = np.abs(spec), np.angle(spec)
    noise       = mag[:noise_frames].mean(axis=0)
    mc          = np.maximum(mag - alpha * noise, beta * noise)
    cf          = np.fft.irfft(mc * np.exp(1j * ph), n=n_fft)
    out         = np.zeros(n_fr * hop + n_fft)
    for i, f in enumerate(cf):
        out[i * hop: i * hop + n_fft] += f * win
    out = out[:len(audio)]
    pk  = np.max(np.abs(out))
    if pk > 0:
        out = out / pk * np.max(np.abs(audio))
    return out.astype(np.float32)


# ── Block 2 ───────────────────────────────────────────────────────────────
def segment(raw_wav: str, cfg: dict) -> List[dict]:
    print("\n" + "═"*60)
    print("BLOCK 2  Spectral Subtraction + Segment Slicing")
    print("═"*60)
    audio, sr = sf.read(raw_wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    total = len(audio) / sr
    n_seg = math.ceil(total / cfg["seg_dur"])
    print(f"  {total:.1f}s total  →  {n_seg} segment(s) of {cfg['seg_dur']}s")

    all_segs = []
    for i in range(n_seg):
        t0 = i * cfg["seg_dur"]
        t1 = min(t0 + cfg["seg_dur"], total)
        sdir = os.path.join(cfg["segs_dir"], f"seg_{i:03d}")
        all_segs.append(dict(
            idx     = i,
            start   = t0,
            end     = t1,
            dur     = t1 - t0,
            seg_dir = sdir,
            seg_wav = os.path.join(sdir, "original_segment.wav"),
            label   = f"seg_{i:03d}",
        ))

    run = [all_segs[cfg["only_seg"]]] if cfg.get("only_seg") is not None else all_segs

    for seg in run:
        pathlib.Path(seg["seg_dir"]).mkdir(parents=True, exist_ok=True)
        if os.path.exists(seg["seg_wav"]):
            print(f"  [cache] {seg['label']}")
            continue
        s0, s1  = int(seg["start"] * sr), int(seg["end"] * sr)
        denoised = spectral_sub(audio[s0:s1], sr)
        sf.write(seg["seg_wav"], denoised, sr)
        print(f"  ✓ {seg['label']}  {seg['dur']:.1f}s")
        wb_log({f"{seg['label']}_dur_s": seg["dur"]})

    return run