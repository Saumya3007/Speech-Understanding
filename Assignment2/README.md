# Speech Understanding & Translation Pipeline

An end-to-end multi-language processing architecture explicitly designed to handle Hinglish code-switched lectures. The core objective translates educational audio from English/Hindi into Maithili while deploying zero-shot voice cloning to preserve the original lecturer's pedagogy and prosody.

## 🚀 Features

- **Audio Ingestion & Denoising**: Native FFMPEG down-sampling followed firmly by automated spectral subtraction routines.
- **Language Identification (LID)**: Frame-level classification boundaries accurately isolating Hinglish code-switching constraints (`wav2vec2-large-xlsr-53`).
- **Constrained ASR**: Transcription rigorously guided by N-gram structural terminology via targeted logit biasing (`whisper-large-v3`).
- **Translation Engine**: Cross-lingual phonetical representation mapping bridging IPA conversions directly onto `indictrans2-en-indic-1B`.
- **Zero-Shot Prosody Cloning**: Unconstrained voice cloning utilizing MMS-TTS coupled directly alongside fundamental frequency (F0) tracking and amplitude DTW temporal alignment. Consistently mimics complex classroom delivery pacing securely.
- **Anti-Spoofing Constraints**: An internally evaluated defense array mapping phase differences utilizing high-scale Linear Frequency Cepstral Coefficients (LFCC) tracked by a Bi-LSTM network. Flawlessly discriminates synthetic traits versus valid vocal inputs.
- **Adversarial Robustness Testing**: Employs physical threshold measurements tracking LID degradation using Fast Gradient Sign Methods (FGSM).

## 🗂️ Project Structure

- `data/` *(Ignored)* - Houses the primary uncompressed `.mp4` references alongside local baseline `student_voice_ref.wav` samples.
- `src/` - Active execution context.
  - `main.py` - Core execution handler mapping parameters concurrently across all evaluating Blocks.
  - `evaluate.py` - Mathematics backbone validating derived limits (MCD, LID-F1, global WER constraints).
  - `plots.py` - Explicit visualization engine routing data dependencies securely onto final PNG representations.
- `speech_pa2_outputs/outputs/` - Core system routing directory tracking execution JSONs, final TTS traces, and plot layouts seamlessly.

## 📦 System Requirements

A functional global binary instance of FFmpeg is strictly mandatory targeting audio extractions. 

If running explicitly via a local Windows context for local plot processing without the baseline HPC GPU dependency chains, simply install visualization and signal routing dependencies natively:

```bash
pip install librosa soundfile matplotlib scikit-learn
```

## ⚙️ Usage Overview

### Executing Full HPC Processing Stack
To target an original 10-minute code-switched sequence structurally against all translation and synthetic evasion bounds natively:
```bash
python src/main.py --input data/Attention.mp4 --output_dir speech_pa2_outputs/outputs --ref_voice data/student_voice_ref.wav
```

### Reproducing Visual Matrices
The plotting interface can be engaged directly without running heavy underlying transformer structures, accurately recreating visual outputs derived from a past execution log context organically:
```bash
python src/plots.py speech_pa2_outputs/outputs
```
