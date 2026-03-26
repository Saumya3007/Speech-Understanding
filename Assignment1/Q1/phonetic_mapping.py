import os
import re
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

BUNDLE   = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
BLANK_ID = 0

@dataclass
class Segment:
    label: str
    start: float
    end:   float
    score: float = 0.0

    @property
    def duration_ms(self):
        return round((self.end - self.start) * 1000, 1)


def get_emission(waveform: torch.Tensor, model, device: str) -> torch.Tensor:
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
    return emission.cpu()


def transcript_to_tokens(transcript: str, label2idx: dict):
   
    clean   = re.sub(r"[^A-Za-z' ]", "", transcript.upper()).strip()
    clean   = re.sub(r"\s+", " ", clean)
    tok_str = "|".join(clean.split())
    tok_ids = [label2idx[c] for c in tok_str if c in label2idx]
    return tok_str, tok_ids

def tokens_to_char_segments(aligned_tokens: torch.Tensor,
                             scores: torch.Tensor,
                             labels: tuple,
                             ratio: float) -> List[Segment]:
    token_spans = F.merge_tokens(aligned_tokens, scores)
    segments    = []
    for span in token_spans:
        lbl = labels[span.token]
        if lbl == "-":
            continue
        segments.append(Segment(lbl,
                                span.start * ratio,
                                span.end   * ratio,
                                float(span.score)))
    return segments


def segments_to_words(char_segs: List[Segment]) -> List[Segment]:
    words, buf, buf_start = [], [], None
    for seg in char_segs:
        if seg.label == "|":
            if buf:
                words.append(Segment("".join(s.label for s in buf),
                                     buf_start, buf[-1].end,
                                     float(np.mean([s.score for s in buf]))))
            buf, buf_start = [], None
        else:
            if buf_start is None:
                buf_start = seg.start
            buf.append(seg)
    if buf:
        words.append(Segment("".join(s.label for s in buf),
                              buf_start, buf[-1].end,
                              float(np.mean([s.score for s in buf]))))
    return words

def forced_align_wav2vec2(signal: np.ndarray, sr: int,
                           transcript: str,
                           device: str = "cpu") -> Tuple[List[Segment], List[Segment]]:
    model  = BUNDLE.get_model().to(device).eval()
    labels = BUNDLE.get_labels()
    l2i    = {c: i for i, c in enumerate(labels)}

    target_sr = BUNDLE.sample_rate
    waveform  = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    tok_str, tok_ids = transcript_to_tokens(transcript, l2i)
    if len(tok_ids) < 2:
        print("  ⚠  Transcript too short — empty alignment returned.")
        return [], []

    emission = get_emission(waveform, model, device)
    T        = emission.shape[1]
    ratio    = waveform.shape[1] / target_sr / T
    print(f"  Emission frames: {T}  |  Tokens: {len(tok_ids)}"
          f"  |  {ratio*1000:.2f} ms/frame")

    targets        = torch.tensor([tok_ids], dtype=torch.int32)
    aligned_tokens, ali_scores = F.forced_align(emission, targets, blank=BLANK_ID)
    aligned_tokens = aligned_tokens[0]
    ali_scores     = ali_scores[0].exp()

    char_segs = tokens_to_char_segments(aligned_tokens, ali_scores, labels, ratio)
    word_segs = segments_to_words(char_segs)
    return char_segs, word_segs


def compute_rmse(manual_bounds: np.ndarray, model_bounds: np.ndarray) -> dict:
    n_m, n_mod = len(manual_bounds), len(model_bounds)
    if n_m == 0 or n_mod == 0:
        return {"RMSE (s)": float("nan"), "MAE (s)": float("nan"),
                "Max Error (s)": float("nan"),
                "# Manual Bounds": n_m, "# Model Bounds": n_mod}
    errors = np.array([abs(t - model_bounds[np.argmin(np.abs(model_bounds - t))])
                       for t in manual_bounds])
    return {
        "RMSE (s)":        round(float(np.sqrt(np.mean(errors ** 2))), 6),
        "MAE (s)":         round(float(np.mean(errors)), 6),
        "Max Error (s)":   round(float(np.max(errors)), 6),
        "# Manual Bounds": n_m,
        "# Model Bounds":  n_mod,
    }

def plot_alignment(signal, sr, char_segs, word_segs, manual_bounds, outdir):
    duration = len(signal) / sr
    t_wave   = np.linspace(0, duration, len(signal))
    fig, axes = plt.subplots(3, 1, figsize=(15, 11))

    ax = axes[0]
    ax.plot(t_wave, signal, linewidth=0.3, color="black", alpha=0.7)
    for seg in char_segs:
        ax.axvline(seg.start, color="steelblue", linewidth=0.5, alpha=0.5)
    ax.set(title="Waveform + Wav2Vec2 Character Boundaries",
           ylabel="Amplitude", xlim=(0, duration))
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(t_wave, signal, linewidth=0.3, color="black", alpha=0.5)
    cmap_w = plt.cm.Set3(np.linspace(0, 1, max(len(word_segs), 1)))
    for i, seg in enumerate(word_segs):
        ax.axvspan(seg.start, seg.end, alpha=0.35, color=cmap_w[i % len(cmap_w)])
        mid = (seg.start + seg.end) / 2
        ylo, yhi = ax.get_ylim()
        ax.text(mid, (ylo + yhi) / 2, seg.label,
                ha="center", va="center", fontsize=7, fontweight="bold", clip_on=True)
    ax.set(title="Word-Level Alignment (Wav2Vec2)", ylabel="Amplitude",
           xlim=(0, duration))
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    model_bounds = np.array([s.start for s in char_segs])
    if len(manual_bounds):
        ax.vlines(manual_bounds, 0.55, 1.45, colors="crimson", linewidth=1.2,
                  label=f"Manual V/UV  ({len(manual_bounds)})")
    ax.vlines(model_bounds, -0.45, 0.45, colors="steelblue", linewidth=0.6,
              label=f"Wav2Vec2 chars  ({len(model_bounds)})")
    ax.set(title="Boundary Comparison  —  Manual (red) vs Wav2Vec2 (blue)",
           xlabel="Time (s)", ylim=(-1, 2), yticks=[], xlim=(0, duration))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.suptitle("Phonetic Forced Alignment  —  WAV2VEC2_ASR_BASE_960H  (LJSpeech)",
                 fontsize=13)
    plt.tight_layout()
    p = os.path.join(outdir, "q1_phonetic_alignment.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"  Saved {p}")


if __name__ == "__main__":
    os.makedirs("./outputs", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 62)
    print(f"  [phonetic_mapping] Step 4  |  device: {device}")
    print("=" * 62)

    signal = np.load("./outputs/signal.npy")
    sr     = int(np.load("./outputs/sample_rate.npy")[0])
    with open("./outputs/transcript.txt") as f:
        transcript = f.read().strip()

    manual_bounds = np.array([])
    if os.path.exists("./outputs/v_uv_boundaries.npy"):
        manual_bounds = np.load("./outputs/v_uv_boundaries.npy",
                                allow_pickle=True).astype(float)

    print(f"  Transcript    : {transcript[:80]}")
    print(f"  Manual bounds : {len(manual_bounds)}")
    print("  Loading WAV2VEC2_ASR_BASE_960H (~360 MB on first run) …")

    char_segs, word_segs = forced_align_wav2vec2(signal, sr, transcript, device)
    print(f"  Char segments : {len(char_segs)}  |  Word segments: {len(word_segs)}")

    model_bounds = np.array([s.start for s in char_segs])
    metrics      = compute_rmse(manual_bounds, model_bounds)

    print("\n  ┌────────────────────────────────────────┐")
    for k, v in metrics.items():
        print(f"  │  {k:<24}: {str(v):<10}")
    print("  └────────────────────────────────────────┘\n")

    pd.DataFrame([metrics]).to_csv("./outputs/q1_rmse_table.csv", index=False)
    print("  Saved ./outputs/q1_rmse_table.csv")

    with open("./outputs/rmse.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    rows = [{"Phone": s.label, "Start (s)": round(s.start, 4),
             "End (s)": round(s.end, 4), "Duration (ms)": s.duration_ms,
             "Score": round(s.score, 4)} for s in char_segs]
    df_ph = pd.DataFrame(rows)
    df_ph.to_csv("./outputs/q1_phoneme_segments.csv", index=False)
    print("  Saved ./outputs/q1_phoneme_segments.csv")
    print("\n  First 20 phoneme segments:")
    print(df_ph.head(20).to_string(index=False))

    plot_alignment(signal, sr, char_segs, word_segs, manual_bounds, "./outputs")
    print("[phonetic_mapping] Done.\n")
