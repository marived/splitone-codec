# Changelog

## 0.3.1 — 2025-08
- Fix off-by-one in `flatten_rvq` when n_q doesn't divide T evenly (it never did but the
  bug was hiding behind the assertion).
- Update default lr in `configs/base.yaml`, the old one was too high for n_codebooks=8.

## 0.3.0 — 2025-05
- Switch to EMA codebook updates + dead-code restarts. Codebook usage went from ~40% to
  ~94% on the same data.
- Add `splitone.tokenize.flatten_rvq` / `unflatten_rvq` helpers.

## 0.2.0 — 2024-12
- Residual VQ (was single-stage).
- Multi-scale STFT loss.
- AISHELL-3 added to the default mixture.

## 0.1.0 — 2024-06
- First trainable version. Encoder + single-stage VQ + decoder, L1 only.
