"""
Speech Understanding — Programming Assignment 2
Pipeline:
  Block 1:  Audio Ingestion (ffmpeg → 16kHz mono WAV)
  Block 2:  Denoising (Spectral Subtraction) + 10-min segmentation
  Block 3:  Frame-level LID (Wav2Vec2-XLSR + 2-class head)
  Block 4:  Constrained Whisper-large-v3 Transcription
  Block 5:  IPA Unified Representation (G2P)
  Block 6:  EN/HI → Maithili Translation (Google Translate)
  Block 7:  x-vector + DTW Prosody + MMS-TTS-mai Synthesis
  Block 8:  Evaluation Metrics (WER, MCD, LID F1)
  Block 9:  Anti-Spoof LFCC-LSTM + FGSM Adversarial Attack
  Block 10: All Plots
"""

import os
import sys
import json
import subprocess
import argparse

import numpy as np
import torch

# ── Config ────────────────────────────────────────────────────────────────────

def build_cfg(args) -> dict:
    out = os.path.abspath(args.output_dir)
    os.makedirs(out, exist_ok=True)
    cfg = {
        # paths
        "input_video":      os.path.abspath(args.input),
        "output_dir":       out,
        "raw_wav":          os.path.join(out, "raw_16k.wav"),
        "denoised_wav":     os.path.join(out, "denoised.wav"),
        "orig_segment":     os.path.join(out, "original_segment.wav"),
        "ref_voice":        os.path.abspath(args.ref_voice),
        "synth_wav":        os.path.join(out, "output_LRL_cloned.wav"),
        # json outputs
        "lid_json":         os.path.join(out, "lid_results.json"),
        "transcript_json":  os.path.join(out, "transcript.json"),
        "ipa_json":         os.path.join(out, "ipa_results.json"),
        "lrl_json":         os.path.join(out, "lrl_text.json"),
        "metrics_json":     os.path.join(out, "metrics.json"),
        "adv_json":         os.path.join(out, "adv_metrics.json"),
        # model weights
        "lid_weights":      os.path.join(out, "lid_head.pt"),
        "cm_weights":       os.path.join(out, "cm_head.pt"),
        # hyperparams
        "seg_len_s":        600,          # 10-min segments
        "target_sr":        22050,
        "lid_epochs":       50,
        "epochs":           50,
        "model_cache":      args.model_cache,
        # W&B / HF
        "wandb_project":    args.wandb_project,
        "hf_repo":          args.hf_repo,
        "hf_token":         args.hf_token,
    }
    # save config for plots.py standalone use
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    return cfg


# ── Block 1: Audio Ingestion ──────────────────────────────────────────────────

def ingest(cfg: dict) -> str:
    print("\n" + "=" * 60)
    print("BLOCK 1  Audio Ingestion (ffmpeg → 16kHz mono WAV)")
    print("=" * 60)
    out = cfg["raw_wav"]
    if os.path.exists(out):
        import librosa
        dur = librosa.get_duration(path=out)
        print(f"  [cache] {out}  ({dur:.1f}s)")
        return out
    cmd = [
        "ffmpeg", "-y", "-i", cfg["input_video"],
        "-ac", "1", "-ar", "16000", "-vn", out
    ]
    print(f"  {' '.join(cmd)}")
    ret = subprocess.run(cmd, capture_output=True)
    if ret.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{ret.stderr.decode()}")
    import librosa
    dur = librosa.get_duration(path=out)
    print(f"  ingested → {out}  ({dur:.1f}s)")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Speech PA2 Pipeline")
    parser.add_argument("--input",         default="/scratch/m25csa027/Assin/SpeechA/data/Attention.mp4",
                        help="Input video/audio file")
    parser.add_argument("--output_dir",    default="../speech_pa2_outputs/outputs",
                        help="Output directory")
    parser.add_argument("--ref_voice",     default="/scratch/m25csa027/Assin/SpeechA/data/student_voice_ref.wav",
                        help="60s student reference voice WAV")
    parser.add_argument("--model_cache",   default=None,
                        help="Directory to cache downloaded models")
    parser.add_argument("--wandb_project", default="speech_pa2",
                        help="W&B project name")
    parser.add_argument("--hf_repo",       default="",
                        help="HuggingFace repo for artifact upload")
    parser.add_argument("--hf_token",      default="",
                        help="HuggingFace token")
    args = parser.parse_args()

    cfg = build_cfg(args)

    # ── import all modules after sys.path is set ──────────────────────────────
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from tracking  import init_tracking
    from stt       import denoise, segment, lid, transcribe
    from ipa       import ipa_convert
    from translate import translate
    from tts       import synthesise
    from evaluate  import metrics, adversarial
    from plots     import all_plots

    # ── init W&B ─────────────────────────────────────────────────────────────
    init_tracking(cfg)

    # ── Block 1: ingest ───────────────────────────────────────────────────────
    raw_wav = ingest(cfg)

    # ── Block 2: denoise + segment ────────────────────────────────────────────
    denoised = denoise(raw_wav, cfg["output_dir"], cfg)
    segs     = segment(denoised, cfg["output_dir"], cfg)

    # copy first segment as "original_segment.wav" for submission
    import shutil, librosa
    if not os.path.exists(cfg["orig_segment"]):
        shutil.copy(segs[0]["seg_wav"], cfg["orig_segment"])
    dur_total = sum(librosa.get_duration(path=s["seg_wav"]) for s in segs)
    print(f"\n  {dur_total:.1f}s total  →  {len(segs)} segment(s) of {cfg['seg_len_s']}s")

    # ── Block 3: LID ──────────────────────────────────────────────────────────
    lid_res = lid(segs, cfg)

    # ── Block 4: transcription ────────────────────────────────────────────────
    transcript = transcribe(segs, lid_res, cfg)

    # ── Block 5: IPA ──────────────────────────────────────────────────────────
    ipa_res = ipa_convert(segs, transcript, cfg)

    # ── Block 6: translate → Maithili ─────────────────────────────────────────
    lrl_res = translate(segs, cfg)

    # ── Block 7: TTS (x-vector + DTW prosody + MMS-TTS-mai) ──────────────────
    synth_wav = synthesise(segs, lrl_res, cfg)

    # ── Block 8: metrics ──────────────────────────────────────────────────────
    m = metrics(segs, transcript, lid_res, synth_wav, cfg)

    # ── Block 9: anti-spoof + adversarial ─────────────────────────────────────
    adv = adversarial(segs, lid_res, synth_wav, cfg)

    # ── Block 10: all plots ───────────────────────────────────────────────────
    try:
        all_plots(cfg)
    except Exception as e:
        print(f"  [plots] WARNING: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  WER_EN  = {m.get('wer_en', -1):.4f}  (pass: {m.get('pass_wer_en', False)})")
    print(f"  WER_HI  = {m.get('wer_hi', -1):.4f}  (pass: {m.get('pass_wer_hi', False)})")
    print(f"  MCD     = {m.get('mcd',    -1):.4f}  (pass: {m.get('pass_mcd',    False)})")
    print(f"  LID_F1  = {m.get('lid_f1', -1):.4f}  (pass: {m.get('pass_lid_f1', False)})")
    print(f"  EER     = {adv.get('eer_pct', -1):.2f}%  (pass: {adv.get('pass_eer', False)})")
    print(f"  FGSM ε  = {adv.get('fgsm_epsilon', 0):.2e}  "
          f"SNR={adv.get('fgsm_snr_db', 0):.1f}dB  "
          f"flip={adv.get('fgsm_flip_rate_pct', 0):.1f}%")
    print(f"\n  Outputs → {cfg['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())