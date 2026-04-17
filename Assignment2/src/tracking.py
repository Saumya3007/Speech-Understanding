"""
tracking.py — W&B + HuggingFace Hub logging utilities
Exports: init_tracking, wb_log, wb_save, hf_upload
"""

import os

_run = None  # W&B run object


def init_tracking(cfg: dict):
    """Initialise Weights & Biases run. Silently skips if wandb not installed."""
    global _run
    try:
        import wandb
        _run = wandb.init(
            project=cfg.get("wandb_project", "speech_pa2"),
            config={k: v for k, v in cfg.items()
                    if isinstance(v, (str, int, float, bool))},
            resume="allow",
        )
        print(f"  [W&B] run initialised: {_run.name}")
    except Exception as e:
        print(f"  [W&B] skipped ({e})")
        _run = None


def wb_log(data: dict):
    """Log a dict of scalars to W&B (no-op if not initialised)."""
    if _run is not None:
        try:
            import wandb
            wandb.log(data)
        except Exception:
            pass


def wb_save(path: str):
    """Upload a file artifact to W&B (no-op if not initialised)."""
    if _run is not None and os.path.exists(path):
        try:
            import wandb
            wandb.save(path)
        except Exception:
            pass


def hf_upload(local_path: str, repo_path: str, cfg: dict):
    """Upload a file to a HuggingFace Hub repository. Skips if repo not set."""
    repo = cfg.get("hf_repo", "").strip()
    token = cfg.get("hf_token", "").strip()
    if not repo or repo in ("", "YOUR_HF_REPO"):
        return
    if not os.path.exists(local_path):
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token or None)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=repo,
            repo_type="dataset",
        )
        print(f"  [HF] uploaded {local_path} → {repo}/{repo_path}")
    except Exception as e:
        print(f"  [HF] upload skipped ({e})")