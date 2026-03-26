# Q3 — Ethical Auditing & Documentation Debt Mitigation

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in HF_USERNAME and HF_TOKEN
python main.py              
Environment
text
HF_USERNAME=your_huggingface_username
HF_TOKEN=hf_your_write_token
Configuration
All parameters are in config.yaml. Key settings:

text
dataset:
  url: "test-clean"           # 334MB, 2620 utterances, auto-downloads
  max_train_samples: 2000

training:
  model_name: "facebook/wav2vec2-base-960h"
  n_epochs: 3
  batch_size: 4
  lambda_fair: 0.5
File Structure
text
Q3/
├── main.py                          # runs everything
├── audit.py                         # Part 1: bias audit
├── privacymodule.py                 # Part 2: VoiceBiometricObfuscator
├── pp_demo.py                       # Part 2: audio pair generation
├── train_fair.py                    # Part 3: fairness ASR training
├── evaluation_scripts/
│   └── fad_eval.py                  # Part 4: FAD + DNSMOS validation
├── config.yaml
├── config_loader.py
├── .env
├── requirements.txt
└── Results/
    ├── audit_plots.png / .pdf       ← Part 1 output
    ├── speakers_audit.csv
    ├── gender_time_audit.csv
    ├── documentation_debt.csv
    ├── spectrogram_pairs.png        ← Part 2 output
    ├── transformation_metadata.csv
    ├── training_results.png         ← Part 3 output
    ├── training_losses.csv
    ├── wer_by_gender.csv
    ├── evaluation/
    │   ├── fad_dnsmos_validation.png  ← Part 4 output
    │   ├── validation_results.csv
    │   ├── dnsmos_original.csv
    │   └── dnsmos_transformed.csv
    └── examples/
        ├── pair_XXX_orig_M.wav      ← original audio
        └── pair_XXX_trans_M2F.wav   ← transformed audio
Checkpoint
The best model checkpoint is saved to ./q3/best_model/ during Part 3 training and uploaded to HuggingFace at {HF_USERNAME}/wav2vec2-fairness-q3.

Checkpoint corresponds to the lowest total loss (CTC + fairness penalty) step across all epochs

Trained on test-clean, 2000 samples, 3 epochs, λ=0.5

Base model: facebook/wav2vec2-base-960h with feature encoder frozen

Results Location
All experiment outputs are in the Results/ folder. Results/evaluation/validation_results.csv contains the final FAD, DNSMOS, and SNR scores.