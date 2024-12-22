# data

I train mostly on a mix of LibriTTS-R clean-100 and AISHELL-3, after:

1. Resample to 24 kHz, mono. `torchaudio.functional.resample` is fine for offline.
2. Trim leading/trailing silence below -40 dBFS.
3. Drop clips shorter than 1.5 s and longer than 20 s.
4. Build a manifest with `scripts/build_manifest.py`.

For Chinese, AISHELL-3 only gives you ~80 hours, which is fine for the codec but I
wouldn't trust it for downstream LM work.
