"""
config.py — single source of truth for all paths, keys, hyperparams.
All sensitive values read from .env via python-dotenv.
"""
import os, time, pathlib
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # set env vars manually if dotenv not installed

# ── API keys (from .env) ──────────────────────────────────────────────────
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
HF_TOKEN      = os.getenv("HF_TOKEN",      "")

# ── W&B / HF names ────────────────────────────────────────────────────────
WB_PROJECT    = os.getenv("WANDB_PROJECT", "speech_pa2")
WB_RUN        = os.getenv("WANDB_RUN",     f"run_{int(time.time())}")
HF_REPO       = os.getenv("HF_REPO",       "")

# ── Paths ─────────────────────────────────────────────────────────────────
OUTDIR        = os.getenv("OUTDIR",       "./speech_pa2_outputs")
LOCAL_FILE    = os.getenv("LOCAL_FILE",   "")
YOUTUBE_URL   = os.getenv("YOUTUBE_URL",  "https://www.youtube.com/watch?v=wjZofJX0v4M")
REF_VOICE_ENV = os.getenv("REF_VOICE",    "")

# ── Model / training params ───────────────────────────────────────────────
SEG_DUR       = int(os.getenv("SEG_DUR",       "600"))
LRL           = os.getenv("LRL",               "maithili")
WHISPER_MODEL = os.getenv("WHISPER_MODEL",     "openai/whisper-large-v3")
EPOCHS        = int(os.getenv("EPOCHS",        "30"))
BATCH_SIZE    = int(os.getenv("BATCH_SIZE",    "32"))
TARGET_SR     = 22050


def make_cfg(outdir=None, local_file=None, youtube_url=None,
             ref_voice=None, seg_dur=None, only_seg=None,
             lrl=None, whisper_model=None,
             epochs=None, batch_size=None) -> dict:
    """Build master config dict and create all required directories."""
    base = outdir or OUTDIR
    ref  = ref_voice or REF_VOICE_ENV or f"{base}/outputs/student_voice_ref.wav"
    cfg  = dict(
        youtube_url      = youtube_url    or YOUTUBE_URL,
        local_file       = local_file     or LOCAL_FILE,
        seg_dur          = seg_dur        or SEG_DUR,
        only_seg         = only_seg,
        lrl              = (lrl or LRL).lower(),
        whisper_model    = whisper_model  or WHISPER_MODEL,
        wb_project       = WB_PROJECT,
        wb_run           = WB_RUN,
        hf_repo          = HF_REPO,
        # dirs
        output_dir       = f"{base}/outputs",
        plots_dir        = f"{base}/plots",
        data_dir         = f"{base}/data",
        segs_dir         = f"{base}/outputs/segments",
        # files
        raw_wav          = f"{base}/outputs/raw_audio.wav",
        ref_voice        = ref,
        synth_wav        = f"{base}/outputs/output_LRL_cloned.wav",
        transcript_json  = f"{base}/outputs/transcript.json",
        ipa_json         = f"{base}/outputs/ipa_transcript.json",
        lrl_json         = f"{base}/outputs/lrl_text.json",
        metrics_json     = f"{base}/outputs/metrics.json",
        adv_json         = f"{base}/outputs/adv_metrics.json",
        lid_weights      = f"{base}/outputs/lid_weights.pt",
        cm_weights       = f"{base}/outputs/antispoof_cm.pt",
        epochs           = epochs      or EPOCHS,
        batch_size       = batch_size  or BATCH_SIZE,
        target_sr        = TARGET_SR,
    )
    for k in ["output_dir", "plots_dir", "data_dir", "segs_dir"]:
        pathlib.Path(cfg[k]).mkdir(parents=True, exist_ok=True)
    return cfg