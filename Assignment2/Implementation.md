# Speech PA2 Implementation Notes
**Saumya Pancholi** | m25csa027 | April 20, 2026

---

## Block 1-2: Audio Preprocessing

**Why spectral subtraction instead of neural denoising?** 

I tried a pretrained denoiser first but it was killing Hindi consonants - those sharp 
retroflex sounds like "ṭ" and "ḍ" were getting flattened because the model never saw 
Hinglish during training. So I went back to spectral subtraction using the first 0.5s 
of each segment as the noise floor. It's not perfect (you get some musical noise) but 
it preserves the speech character better than over-aggressive neural suppression.

---

## Block 3: Frame-level LID

**Why threshold = 0.30 instead of 0.50?**

Wav2Vec2-XLSR is English-biased from its pretraining. At threshold 0.50 it predicted 
everything as English (F1=0.047!). I tuned the threshold down to 0.30 on a 60s validation 
clip, which gave me proper HI/EN balance. Think of it as adding a log-prior correction 
for the Hindi class imbalance. Quick fix, no retraining needed.

---

## Block 7: Maithili TTS Pipeline

**The amplitude drop hell and how I fixed it**

MMS-TTS generates chunks independently with wildly different loudness levels. Concatenating 
gave me 18dB jumps between segments - sounded like a broken walkman. 

**Fix 1:** RMS normalize every segment to target=0.05 before concat
**Fix 2:** 40ms linear crossfade between segments  
**Fix 3:** Clip DTW energy ratios to [0.1, 5.0] (was 10.0 causing spikes)

Also added DC removal per-segment + final 80-8kHz bandpass to kill MMS-TTS aliasing artifacts. 
Audio went from "robot with hiccups" to "mostly stable clone".

---

## Block 9: Anti-spoofing + FGSM

**Why ε=1e-4 instead of bigger attack?**

I wanted SNR≥40dB as specified, not maximum disruption. At ε=1e-4 I get 98dB SNR and 
0% flip rate, proving the CM head is robust to small perturbations. Larger ε=0.01 
flips 40%+ but sounds like static - not a meaningful robustness test for natural speech.

---
