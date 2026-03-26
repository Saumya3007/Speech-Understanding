import os
import numpy as np
import torch
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET_SR = 16_000         
SAMPLE_IDX = 0             


def pre_emphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """y[n] = x[n] - coeff * x[n-1]"""
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def frame_signal(signal: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Returns (n_frames, frame_len) via vectorised stride indexing."""
    n_frames = 1 + (len(signal) - frame_len) // hop
    row = np.arange(frame_len)[None, :]
    col = hop * np.arange(n_frames)[:, None]
    return signal[row + col]


def apply_window(frames: np.ndarray, wtype: str = "hamming") -> np.ndarray:
    L      = frames.shape[1]
    win_fn = {"hamming": np.hamming, "hanning": np.hanning}.get(wtype, np.ones)
    return frames * win_fn(L)


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=float) / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=float) / 2595.0) - 1.0)


def mel_filterbank(n_filters: int, n_fft: int, sr: int,
                   fmin: float = 0.0, fmax: float = None) -> np.ndarray:
    """Triangular mel filterbank → (n_filters, n_fft//2+1)."""
    if fmax is None:
        fmax = sr / 2.0
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filters + 2)
    hz_pts  = mel_to_hz(mel_pts)
    bins    = np.clip(np.floor((n_fft + 1) * hz_pts / sr).astype(int), 0, n_fft // 2)
    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        lo, ctr, hi = bins[m - 1], bins[m], bins[m + 1]
        if ctr > lo:
            fb[m - 1, lo:ctr] = (np.arange(lo, ctr) - lo) / (ctr - lo)
        if hi > ctr:
            fb[m - 1, ctr:hi] = (hi - np.arange(ctr, hi)) / (hi - ctr)
    return fb


# ════════════════════════════ main API ═══════════════════════════════════════

def compute_mfcc(signal: np.ndarray, sr: int,
                 n_mfcc: int     = 13,
                 n_filters: int  = 26,
                 n_fft: int      = 512,
                 hop: int        = 160,
                 win: int        = 400,
                 wtype: str      = "hamming",
                 pre_emph: float = 0.97):
   
    sig    = pre_emphasis(signal, pre_emph)
    frames = frame_signal(sig, win, hop)
    frames = apply_window(frames, wtype)

    mag    = np.abs(np.fft.rfft(frames, n=n_fft))
    power  = (1.0 / n_fft) * mag ** 2

    fb     = mel_filterbank(n_filters, n_fft, sr)
    energy = power @ fb.T
    energy = np.where(energy == 0, np.finfo(float).eps, energy)
    log_e  = np.log(energy)                                
    k = np.arange(n_mfcc)[:, None]
    n = np.arange(n_filters)[None, :]
    D = np.cos(np.pi * k * (2 * n + 1) / (2 * n_filters))
    mfcc = log_e @ D.T                                     
    return mfcc, log_e


def compute_cepstrum(signal: np.ndarray, sr: int,
                     n_fft: int = 512,
                     hop: int   = 160,
                     win: int   = 400) -> np.ndarray:
    """Real cepstrum per frame → (T, n_fft)."""
    sig    = pre_emphasis(signal)
    frames = frame_signal(sig, win, hop)
    frames = apply_window(frames, "hamming")
    mag    = np.abs(np.fft.rfft(frames, n=n_fft))
    logmag = np.log(mag + np.finfo(float).eps)
    return np.fft.irfft(logmag, n=n_fft)


def load_audio_sample(data_dir: str = "./data", idx: int = SAMPLE_IDX):
    
    os.makedirs(data_dir, exist_ok=True)
    print(f"  [data] Loading LJSpeech-1.1 ({data_dir}/) …")
    print("         First run takes time — subsequent runs are instant.")

    dataset = torchaudio.datasets.LJSPEECH(root=data_dir, download=True)
    waveform, orig_sr, _transcript, normalized_transcript = dataset[idx]
    
    if orig_sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, orig_sr, TARGET_SR)

    signal = waveform.squeeze().numpy().astype(np.float32)
    print(f"  [data] Clip #{idx} | orig SR={orig_sr} → resampled to {TARGET_SR} Hz")
    print(f"  [data] Duration: {len(signal)/TARGET_SR:.2f}s")
    print(f"  [data] Transcript: {normalized_transcript[:80]}")
    return signal, TARGET_SR, normalized_transcript.strip()


if __name__ == "__main__":
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./data",    exist_ok=True)

    print("=" * 62)
    print("  [mfcc_manual] Step 1 — Manual MFCC, Mel-Spec & Cepstrum")
    print("=" * 62)

    signal, sr, transcript = load_audio_sample()
    duration  = len(signal) / sr
    n_fft     = 512
    n_filters = 26
    hop       = 160
    win       = 400

    mfcc, log_mel = compute_mfcc(signal, sr,
                                  n_mfcc=13, n_filters=n_filters,
                                  n_fft=n_fft, hop=hop, win=win)
    cep = compute_cepstrum(signal, sr, n_fft=n_fft, hop=hop, win=win)
    fb  = mel_filterbank(n_filters, n_fft, sr)
    print(f"  MFCC shape     : {mfcc.shape}")
    print(f"  Log-Mel shape  : {log_mel.shape}")
    print(f"  Cepstrum shape : {cep.shape}")

    t_wave   = np.linspace(0, duration, len(signal))
    n_frames = mfcc.shape[0]
    t_frames = np.linspace(0, duration, n_frames)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t_wave, signal, linewidth=0.35, color="steelblue")
    ax.set(title=f"Waveform  —  LJSpeech clip #{SAMPLE_IDX}",
           xlabel="Time (s)", ylabel="Amplitude")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./outputs/q1_waveform.png", dpi=150)
    plt.close()
    print("  Saved ./outputs/q1_waveform.png")

    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(log_mel.T, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, duration, 0, n_filters])
    ax.set(title="Log Mel-Spectrogram  (26 Mel bands, manual pipeline)",
           xlabel="Time (s)", ylabel="Mel Band Index")
    plt.colorbar(im, ax=ax, label="Log Energy")
    plt.tight_layout()
    plt.savefig("./outputs/q1_mel_spectrogram.png", dpi=150)
    plt.close()
    print("  Saved ./outputs/q1_mel_spectrogram.png")

    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(mfcc.T, aspect="auto", origin="lower", cmap="inferno",
                   extent=[0, duration, 0, mfcc.shape[1]])
    ax.set(title="Manual MFCC  (13 Coefficients)",
           xlabel="Time (s)", ylabel="Coefficient Index")
    plt.colorbar(im, ax=ax, label="Log Energy")
    plt.tight_layout()
    plt.savefig("./outputs/q1_mfcc_heatmap.png", dpi=150)
    plt.close()
    print("  Saved ./outputs/q1_mfcc_heatmap.png")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    axes[0].imshow(log_mel.T, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, duration, 0, n_filters])
    axes[0].set(title="Log Mel-Spectrogram  (before DCT)",
                ylabel="Mel Band Index")
    axes[1].imshow(mfcc.T, aspect="auto", origin="lower", cmap="inferno",
                   extent=[0, duration, 0, mfcc.shape[1]])
    axes[1].set(title="MFCC  (after DCT-II)",
                xlabel="Time (s)", ylabel="Coefficient Index")
    for ax in axes:
        plt.colorbar(ax.images[0], ax=ax, label="Log Energy")
    plt.suptitle("Mel-Spectrogram  →  MFCC Pipeline  (LJSpeech-1.1)", fontsize=13)
    plt.tight_layout()
    plt.savefig("./outputs/q1_melspec_vs_mfcc.png", dpi=150)
    plt.close()
    print("  Saved ./outputs/q1_melspec_vs_mfcc.png")

    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    for row in fb:
        ax.plot(freqs, row, linewidth=0.8)
    ax.set(title=f"26-band Triangular Mel Filterbank  (SR={sr} Hz)",
           xlabel="Frequency (Hz)", ylabel="Amplitude")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./outputs/q1_mel_filterbank.png", dpi=150)
    plt.close()
    print("  Saved ./outputs/q1_mel_filterbank.png")

    q_lo_end = cep.shape[1] // 10
    q_hi_st  = max(1, int(sr / 400))
    q_hi_end = min(cep.shape[1] // 2, int(sr / 60))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].imshow(cep[:, :q_lo_end].T, aspect="auto", origin="lower",
                   cmap="RdBu_r",
                   extent=[0, duration, 0, q_lo_end / sr * 1000])
    axes[0].set(title="Low-Quefrency  (Vocal Tract Envelope)",
                xlabel="Time (s)", ylabel="Quefrency (ms)")
    axes[1].imshow(cep[:, q_hi_st:q_hi_end].T, aspect="auto", origin="lower",
                   cmap="RdBu_r",
                   extent=[0, duration, q_hi_st/sr*1000, q_hi_end/sr*1000])
    axes[1].set(title="High-Quefrency  (Pitch / F0 Region)",
                xlabel="Time (s)", ylabel="Quefrency (ms)")
    plt.suptitle("Real Cepstrum — Low vs High Quefrency  (LJSpeech)", fontsize=13)
    plt.tight_layout()
    plt.savefig("./outputs/q1_cepstrum_heatmap.png", dpi=150)
    plt.close()
    print("  Saved ./outputs/q1_cepstrum_heatmap.png")

    np.save("./outputs/signal.npy",      signal)
    np.save("./outputs/sample_rate.npy", np.array([sr]))
    np.save("./outputs/mfcc.npy",        mfcc)
    np.save("./outputs/log_mel.npy",     log_mel)
    np.save("./outputs/cepstrum.npy",    cep)
    with open("./outputs/transcript.txt", "w") as f:
        f.write(transcript)

    with open("./data/manifest.txt", "w") as f:
        f.write("# Audio Data Manifest — Q1\n")
        f.write(f"dataset       : LJSpeech-1.1\n")
        f.write(f"source_url    : https://data.keithito.com/data/speech/"
                f"LJSpeech-1.1.tar.bz2\n")
        f.write(f"clip_index    : {SAMPLE_IDX}\n")
        f.write(f"orig_sr       : 22050 Hz  (resampled to {sr} Hz)\n")
        f.write(f"duration      : {duration:.3f} s\n")
        f.write(f"total_clips   : 13100\n")
        f.write(f"transcript    : {transcript}\n")
    print("  Saved ./data/manifest.txt")
    print("[mfcc_manual] Done.\n")
