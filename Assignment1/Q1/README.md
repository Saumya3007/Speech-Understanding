# Q1 — Multi-Stage Cepstral Feature Extraction & Phoneme Boundary Detection

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Dataset
Uses **`hf-internal-testing/librispeech_asr_demo`** (~5 MB, auto-downloads).  
Automatic fallback to `torchaudio LIBRISPEECH test-clean` (~346 MB) if HuggingFace is unavailable.

## Steps & Scripts

| # | Script | What it does |
|---|--------|-------------|
| 1 | `mfcc_manual.py` | Pre-emphasis → Framing → Window → FFT → Mel-fb → Log → DCT |
| 2 | `leakage_snr.py` | Spectral leakage ratio, peak sidelobe, SNR for 3 windows |
| 3 | `voiced_unvoiced.py` | STE + ZCR + Cepstrum pitch strength V/UV/Silence detector |
| 4 | `phonetic_mapping.py` | Wav2Vec2 forced alignment → char/word segments → RMSE |

## Outputs saved to `./outputs/`

| File | Description |
|------|-------------|
| `q1_waveform.png` | Raw waveform |
| `q1_mfcc_heatmap.png` | 13-coeff MFCC heatmap |
| `q1_mel_filterbank.png` | 26-band filterbank visualisation |
| `q1_cepstrum_heatmap.png` | Low vs High quefrency cepstrum |
| `q1_window_shapes.png` | Rect / Hamming / Hanning shapes |
| `q1_leakage_spectra.png` | Per-window magnitude spectra |
| `q1_leakage_overlay.png` | All three overlaid |
| `q1_window_analysis.csv` | Leakage ratio, PSL, MLW, SNR table |
| `q1_voiced_unvoiced.png` | Waveform with V/UV shading |
| `q1_vu_features.png` | STE / ZCR / Pitch-strength tracks |
| `q1_vu_timeline.png` | Horizontal label timeline |
| `q1_phonetic_alignment.png` | Char + word alignment plots |
| `q1_phoneme_segments.csv` | Per-phone start/end/score table |
| `q1_rmse_table.csv` | RMSE / MAE / MaxErr metrics |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 16 000 Hz |
| Frame length | 400 samples (25 ms) |
| Hop length | 160 samples (10 ms) |
| FFT size | 512 |
| Mel filters | 26 |
| MFCC coefficients | 13 |
| Pre-emphasis | 0.97 |
| F0 range (pitch) | 60–400 Hz |
| Wav2Vec2 model | WAV2VEC2_ASR_BASE_960H |
