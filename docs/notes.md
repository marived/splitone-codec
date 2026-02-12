# notes

scratchpad. ideas, post-mortems, things I keep forgetting.

## things that worked

- multi-scale STFT loss with log-magnitude term was the single biggest quality win
- EMA codebook updates fixed the codebook-collapse problem mostly. dead-code restart
  pushed usage from ~40% to ~94%.
- adding AISHELL-3 didn't hurt LibriTTS-R quality, slightly helped mandarin

## things that didn't work

- multi-period discriminator (Hifi-GAN style). on speech it tended to colour the output
  weirdly. left it in `disc.py` but disabled.
- bigger codebook (4096). codebook usage dropped sharply, PESQ basically unchanged.
- a "global" prior on top of the per-frame indices. fun experiment, didn't help.

## things I keep forgetting

- the encoder strides multiply to 320 (= 2*4*5*8). if you change them you also need to
  pad the wav to a multiple of the new product.
- `torchaudio.functional.resample` is very slow on CPU for long clips. resample on GPU
  if you can.
- when `n_codebooks=8` and `T=200` and `vocab=1024`, the flattened sequence has 1600
  tokens. that's the budget you have for an LM input.

## streaming

Receptive field with default strides (2,4,5,8) and kernel-3 res-units is ~320*2 ~= 640
samples ~= 27 ms. The dominant latency comes from the stem (kernel=7) plus the
encoder's downsample blocks. With causal=True it's all one-sided. Cumulative latency
is around 80 ms in practice.

- 2026-01: spent 2 days debugging a 0.1 PESQ drop. eval was using torch's default window (rectangular) while train used hann. moved the window into the loss module.
