# SplitOne-Codec

A small residual-vector-quantised audio codec for speech (24 kHz mono), built while I
tried to wrap my head around SoundStream / Encodec / DAC.

The eventual goal is to feed the discrete tokens into a small speech-LM, so the codec
is tuned for low token rate (~50 Hz) over absolute fidelity.

## Quick start

```bash
pip install -e .
python -m splitone.train --config configs/base.yaml
python -m splitone.encode some_clip.wav out.npy
python -m splitone.decode out.npy recon.wav
```

## Notes

- single codebook of 1024 entries, 320x downsampling -> ~75 tokens/sec
- Encoder is a 1d-conv stack, decoder mirrors it
- Reconstruction loss: L1 + multi-scale STFT magnitude + log-magnitude
