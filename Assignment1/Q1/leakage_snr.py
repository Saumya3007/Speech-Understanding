import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

WINDOWS = {
    "Rectangular": np.ones,
    "Hamming":     np.hamming,
    "Hanning":     np.hanning,
}
COLORS = {"Rectangular": "crimson", "Hamming": "steelblue", "Hanning": "forestgreen"}

def main_lobe_width(spectrum: np.ndarray) -> int:
    mag  = np.abs(spectrum)
    peak = int(np.argmax(mag))
    thr  = mag[peak] / np.sqrt(2)
    lo, hi = peak, peak
    while lo > 0 and mag[lo] > thr:        lo -= 1
    while hi < len(mag) - 1 and mag[hi] > thr: hi += 1
    return max(1, hi - lo)


def peak_sidelobe_db(spectrum: np.ndarray) -> float:
    mag  = np.abs(spectrum)
    peak = int(np.argmax(mag))
    half = main_lobe_width(spectrum)
    lo_  = max(0, peak - half)
    hi_  = min(len(mag), peak + half + 1)
    mask = np.ones(len(mag), dtype=bool)
    mask[lo_:hi_] = False
    sl   = np.max(mag[mask]) if mask.any() else 1e-12
    return float(20 * np.log10(sl / (mag[peak] + 1e-12)))


def leakage_ratio(spectrum: np.ndarray) -> float:
    mag  = np.abs(spectrum)
    peak = int(np.argmax(mag))
    half = main_lobe_width(spectrum)
    lo_  = max(0, peak - half)
    hi_  = min(len(mag), peak + half + 1)
    main_e = np.sum(mag[lo_:hi_] ** 2)
    total  = np.sum(mag ** 2) + 1e-12
    return float(1.0 - main_e / total)


def snr_vs_rect_db(win_sp: np.ndarray, rect_sp: np.ndarray) -> float:
    s = np.mean(np.abs(win_sp) ** 2)
    n = np.mean(np.abs(win_sp - rect_sp) ** 2) + 1e-12
    return float(10 * np.log10(s / n))

def analyse_windows(signal: np.ndarray, sr: int, n_fft: int = 1024):
    seg_len = min(n_fft, len(signal))
    seg     = np.zeros(n_fft, dtype=np.float32)
    seg[:seg_len] = signal[:seg_len]

    spectra  = {}
    win_arrs = {}
    for name, fn in WINDOWS.items():
        w              = fn(n_fft)
        spectra[name]  = np.fft.rfft(seg * w, n=n_fft)
        win_arrs[name] = w

    rect_sp = spectra["Rectangular"]
    results = {}
    for name, sp in spectra.items():
        mlw = main_lobe_width(sp)
        results[name] = {
            "Main-lobe Width (bins)":  mlw,
            "Peak Sidelobe (dBc)":     round(peak_sidelobe_db(sp), 2),
            "Leakage Ratio":           round(leakage_ratio(sp), 5),
            "SNR vs Rectangular (dB)": ("—" if name == "Rectangular"
                                        else round(snr_vs_rect_db(sp, rect_sp), 2)),
        }
    return results, spectra, win_arrs


def plot_window_shapes(win_arrs, sr, n_fft, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (name, w) in zip(axes, win_arrs.items()):
        ax.plot(np.arange(n_fft) / sr * 1000, w, color=COLORS[name], linewidth=1.5)
        ax.set_title(f"{name}", fontsize=11)
        ax.set_xlabel("Time (ms)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Amplitude")
    plt.suptitle("Window Functions  (LJSpeech segment)", fontsize=13)
    plt.tight_layout()
    p = os.path.join(outdir, "q1_window_shapes.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"  Saved {p}")


def plot_spectra_separate(spectra, sr, n_fft, outdir):
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for ax, (name, sp) in zip(axes, spectra.items()):
        ax.plot(freqs, 20 * np.log10(np.abs(sp) + 1e-12),
                color=COLORS[name], linewidth=0.9)
        ax.set_title(f"{name}  —  Magnitude Spectrum (dB)", fontsize=11)
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("dB")
        ax.set_xlim(0, sr / 2); ax.grid(True, alpha=0.3)
    plt.suptitle("Spectral Leakage — Three Windows  (LJSpeech)", fontsize=13)
    plt.tight_layout()
    p = os.path.join(outdir, "q1_leakage_spectra.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"  Saved {p}")


def plot_spectra_overlay(spectra, sr, n_fft, outdir):
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, sp in spectra.items():
        ax.plot(freqs, 20 * np.log10(np.abs(sp) + 1e-12),
                color=COLORS[name], linewidth=0.9, label=name)
    ax.set(title="Spectral Leakage Overlay — All Three Windows",
           xlabel="Frequency (Hz)", ylabel="Magnitude (dB)", xlim=(0, sr / 2))
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(outdir, "q1_leakage_overlay.png")
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
    print("  [leakage_snr] Step 2 — Spectral Leakage & SNR")
    print("=" * 62)

    results, spectra, win_arrs = analyse_windows(signal, sr)

    df = pd.DataFrame(results).T
    print("\n" + "─" * 65)
    print(df.to_string())
    print("─" * 65 + "\n")
    df.to_csv("./outputs/q1_window_analysis.csv")
    print("  Saved ./outputs/q1_window_analysis.csv")

    plot_window_shapes(win_arrs, sr, 1024, "./outputs")
    plot_spectra_separate(spectra, sr, 1024, "./outputs")
    plot_spectra_overlay(spectra, sr, 1024, "./outputs")
    print("[leakage_snr] Done.\n")
