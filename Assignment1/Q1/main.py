import os, sys, time, subprocess

STEPS = [
    ("Step 1 — Manual MFCC, Mel-Spec & Cepstrum",  "mfcc_manual.py"),
    ("Step 2 — Spectral Leakage & SNR",              "leakage_snr.py"),
    ("Step 3 — Voiced / Unvoiced Detection",         "voiced_unvoiced.py"),
    ("Step 4 — Phonetic Mapping & RMSE",             "phonetic_mapping.py"),
]

def banner(msg, w=62):
    print(f"\n{'═'*w}\n  {msg}\n{'═'*w}")

def run_step(label, script):
    banner(label)
    if not os.path.exists(script):
        print(f"  ✗  {script} not found."); sys.exit(1)
    t0     = time.time()
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n  ✗  FAILED (exit {result.returncode})"); sys.exit(result.returncode)
    print(f"\n  ✓  Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    os.makedirs("./data",    exist_ok=True)
    os.makedirs("./outputs", exist_ok=True)

    print("\n╔" + "═"*60 + "╗")
    print("║   Q1 · Multi-Stage Cepstral Feature Extraction Pipeline   ║")
    print("║   Dataset: LJSpeech-1.1  (~2.6 GB, auto-downloads)        ║")
    print("╚" + "═"*60 + "╝")

    t0 = time.time()
    for label, script in STEPS:
        run_step(label, script)

    banner(f"ALL STEPS COMPLETE  ({time.time()-t0:.1f}s total)")
    print("\n  Saved outputs:\n")
    for f in sorted(os.listdir("./outputs")):
        kb = os.path.getsize(f"./outputs/{f}") / 1024
        print(f"    {'./outputs/'+f:<48}  {kb:>7.1f} KB")
    print()
