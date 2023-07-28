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
