# notes

random ideas / things to try, not necessarily in order

- [ ] multi-scale STFT loss (probably need it, L1 alone sounds muffled)
- [ ] EMA codebook? or just use the straight-through trick from VQ-VAE
- [ ] codebook collapse on small batches — need to track usage
- [ ] try AISHELL-3 for some Chinese coverage
- [ ] streaming / causal mode
- [ ] longer segments at train time? 1s might be too short

things that already broke at least once:
- torchaudio resample on the cluster — fixed by pinning to 2.1
- nan in commitment loss when batch size = 1 (no longer support b=1)

- 2023-08: first overfit on a 4s clip after ~2000 steps. very muffled. probably need stft loss.
- 2023-09: ~50% of codes unused after 5k steps. need something smarter than vanilla VQ.
- 2024-01: stft+l1 sounds much better. Still no consonant clarity above ~6kHz.
- 2024-04: trained 50k steps on libritts-r clean-100. recognisable speech, lots of metallic artefacts.
