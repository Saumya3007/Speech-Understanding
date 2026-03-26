import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CMAP = {"voiced": "royalblue", "unvoiced": "tomato", "silence": "lightgray"}


def pre_emphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def frame_signal(signal: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    n   = 1 + (len(signal) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return signal[idx]


def short_time_energy(frames: np.ndarray) -> np.ndarray:
    return np.sum(frames ** 2, axis=1) / frames.shape[1]


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    signs = np.sign(frames)
    signs[signs == 0] = 1
    return np.sum(np.abs(np.diff(signs, axis=1)), axis=1) / (2 * frames.shape[1])


def cepstrum_pitch_strength(signal: np.ndarray, sr: int,
                             frame_len: int = 400,
                             hop: int       = 160,
                             n_fft: int     = 512,
                             f0_min: float  = 60.0,
                             f0_max: float  = 400.0):
    sig    = pre_emphasis(signal)
    frames = frame_signal(sig, frame_len, hop)
    frames = frames * np.hamming(frame_len)
    mag    = np.abs(np.fft.rfft(frames, n=n_fft))
    logmag = np.log(mag + np.finfo(float).eps)
    cep    = np.fft.irfft(logmag, n=n_fft)

    q_lo = max(1, int(sr / f0_max))
    q_hi = min(n_fft // 2, int(sr / f0_min))
    return np.max(np.abs(cep[:, q_lo:q_hi + 1]), axis=1), cep



def detect_voiced_unvoiced(signal: np.ndarray, sr: int,
                            frame_len: int   = 400,
                            hop: int         = 160,
                            sil_pct: float   = 5.0,
                            pitch_pct: float = 35.0,
                            zcr_thr: float   = 0.15):
    frames = frame_signal(signal, frame_len, hop)
    ste    = short_time_energy(frames)
    zcr    = zero_crossing_rate(frames)
    pstr, _= cepstrum_pitch_strength(signal, sr, frame_len, hop)

    sil_thr   = np.percentile(ste,  sil_pct)
    pitch_thr = np.percentile(pstr, pitch_pct)

    labels = np.full(len(frames), "unvoiced", dtype=object)
    labels[ste < sil_thr] = "silence"
    voiced = (ste >= sil_thr) & (pstr >= pitch_thr) & (zcr < zcr_thr)
    labels[voiced] = "voiced"

    times = np.array([(i * hop + frame_len / 2) / sr for i in range(len(frames))])
    bounds = np.array([times[i] for i in range(1, len(labels))
                       if labels[i] != labels[i - 1]])
    return labels, times, bounds, {"ste": ste, "zcr": zcr, "pstr": pstr}


def plot_results(signal, sr, labels, times, boundaries, features, outdir):
    duration  = len(signal) / sr
    t_wave    = np.linspace(0, duration, len(signal))
    frame_dur = float(times[1] - times[0]) if len(times) > 1 else 0.01
    patches   = [mpatches.Patch(color=v, label=k.capitalize())
                 for k, v in CMAP.items()]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t_wave, signal, linewidth=0.3, color="black", alpha=0.7)
    for ts, lbl in zip(times, labels):
        ax.axvspan(ts - frame_dur / 2, ts + frame_dur / 2,
                   alpha=0.22, color=CMAP[lbl], linewidth=0)
    for b in boundaries:
        ax.axvline(b, color="gold", linewidth=0.8, linestyle="--", alpha=0.9)
    ax.legend(handles=patches, loc="upper right", fontsize=9)
    ax.set(title="V/UV/Silence Detection  —  LJSpeech  (gold = transitions)",
           xlabel="Time (s)", ylabel="Amplitude", xlim=(0, duration))
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    p = os.path.join(outdir, "q1_voiced_unvoiced.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"  Saved {p}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].fill_between(times, features["ste"],  color="darkorange",   alpha=0.85)
    axes[0].set(title="Short-Time Energy", ylabel="Energy")
    axes[1].fill_between(times, features["zcr"],  color="mediumpurple", alpha=0.85)
    axes[1].axhline(0.15, color="red", linewidth=1.0, linestyle="--", label="ZCR thr=0.15")
    axes[1].legend(fontsize=8)
    axes[1].set(title="Zero-Crossing Rate", ylabel="ZCR")
    axes[2].fill_between(times, features["pstr"], color="forestgreen",  alpha=0.85)
    axes[2].set(title="Cepstrum Pitch Strength  (High-Quefrency Peak)",
                xlabel="Time (s)", ylabel="Magnitude")
    for ax in axes:
        ax.set_xlim(0, duration); ax.grid(True, alpha=0.25)
        for b in boundaries:
            ax.axvline(b, color="gold", linewidth=0.6, linestyle="--", alpha=0.7)
    plt.suptitle("Feature Tracks for V/UV Classification  —  LJSpeech", fontsize=13)
    plt.tight_layout()
    p = os.path.join(outdir, "q1_vu_features.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"  Saved {p}")

    fig, ax = plt.subplots(figsize=(14, 2.5))
    for ts, lbl in zip(times, labels):
        ax.barh(0, frame_dur, left=ts - frame_dur / 2,
                height=0.8, color=CMAP[lbl], edgecolor="none")
    ax.set(title="Label Timeline  (Voiced / Unvoiced / Silence)",
           xlabel="Time (s)", yticks=[], xlim=(0, duration))
    ax.legend(handles=patches, loc="upper right", fontsize=9)
    plt.tight_layout()
    p = os.path.join(outdir, "q1_vu_timeline.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"  Saved {p}")


if __name__ == "__main__":
    os.makedirs("./outputs", exist_ok=True)

    if os.path.exists("./outputs/signal.npy"):
        signal = np.load("./outputs/signal.npy")
        sr     = int(np.load("./outputs/sample_rate.npy")[0])
    else:
        from mfcc_manual import load_audio_sample
        signal, sr, _ = load_audio_sample()
        np.save("./outputs/signal.npy",      signal)
        np.save("./outputs/sample_rate.npy", np.array([sr]))

    print("=" * 62)
    print("  [voiced_unvoiced] Step 3 — V/UV Boundary Detection")
    print("=" * 62)

    labels, times, boundaries, features = detect_voiced_unvoiced(signal, sr)
    cnt = {k: int((labels == k).sum()) for k in ["voiced", "unvoiced", "silence"]}
    print(f"  voiced={cnt['voiced']}  unvoiced={cnt['unvoiced']}  silence={cnt['silence']}")
    print(f"  Boundary transitions: {len(boundaries)}")

    plot_results(signal, sr, labels, times, boundaries, features, "./outputs")

    np.save("./outputs/v_uv_labels.npy",     labels)
    np.save("./outputs/v_uv_times.npy",      times)
    np.save("./outputs/v_uv_boundaries.npy", boundaries)
    print("[voiced_unvoiced] Done.\n")
