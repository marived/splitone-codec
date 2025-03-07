# SplitOne-Codec :sound:

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![tests](https://img.shields.io/badge/tests-pytest-success.svg)](tests/)

> A small, hackable residual-vector-quantised neural codec for 24 kHz speech, built for
> downstream speech-LM experiments. Optimised for low token rate (~50 Hz) and easy
> editing, not SOTA fidelity.

`splitone-codec` started as a side project while I was trying to understand SoundStream,
EnCodec, and DAC well enough to plug discrete speech tokens into a small autoregressive
language model. It now serves as my default tokenizer for that kind of work.

It is intentionally a few thousand lines of PyTorch. Nothing fancy, nothing magic.

---

## Highlights

- **~50 tokens/sec** at 24 kHz mono — friendly to LM context budgets
- **8x1024 RVQ codebooks** with simple EMA codebook updates (no straight-through gradients)
- **Multi-scale STFT + L1** reconstruction loss, plus an optional adversarial term
- **Pure PyTorch**, no custom CUDA kernels, runs on a single 24 GB GPU
- **Streaming inference** via a causal variant of the encoder (`--causal`)
- **Pretrained checkpoint** on LibriTTS-R + AISHELL-3 mixture (small, ~30M params)
- **Token-level utilities** for feeding the output into a HuggingFace `transformers` LM

## Installation

```bash
git clone https://github.com/marived/splitone-codec.git
cd splitone-codec
pip install -e ".[train]"
```

For inference only:

```bash
pip install -e .
```

The hard deps are `torch`, `torchaudio`, `einops`, `numpy`. Training additionally needs
`accelerate`, `wandb`, and `pyyaml`.

## Quick start

### Encode / decode a clip

```python
import torch, torchaudio
from splitone import SplitOneCodec

codec = SplitOneCodec.from_pretrained("checkpoints/splitone-base.pt").eval()
wav, sr = torchaudio.load("clip.wav")          # (1, T) at 24 kHz, please
assert sr == 24000

with torch.no_grad():
    tokens = codec.encode(wav.unsqueeze(0))    # (1, n_q=8, T_tok)
    recon  = codec.decode(tokens)              # (1, 1, T)

torchaudio.save("recon.wav", recon[0], 24000)
```

### CLI

```bash
python -m splitone.encode  clip.wav   tokens.npy
python -m splitone.decode  tokens.npy recon.wav
python -m splitone.eval    --manifest data/test.tsv --checkpoint checkpoints/splitone-base.pt
```

### As an LLM tokenizer

The output of `encode` is `(B, n_q, T)` int64. For an autoregressive LM I usually flatten
along the codebook axis with an offset:

```python
from splitone.tokenize import flatten_rvq, unflatten_rvq

flat = flatten_rvq(tokens, vocab_size=1024)    # (B, n_q * T)
# ... feed `flat` into a normal transformers model ...
out  = unflatten_rvq(predicted_flat, n_q=8, vocab_size=1024)
```

## Training

A reasonably small run on LibriTTS-R clean-100 + AISHELL-3:

```bash
accelerate launch -m splitone.train \
    --config configs/base.yaml \
    --output_dir checkpoints/run01
```

The default config trains for 400k steps on a single A100 in ~3 days. There's a
`configs/tiny.yaml` that does 50k steps in a few hours, mostly useful for debugging the
loss curves.

Loss components, in order of how much I trust them:

1. Multi-scale STFT magnitude loss (n_fft in {2048, 1024, 512, 256, 128})
2. L1 on the waveform
3. Codebook commitment loss
4. (optional) MS-STFT discriminator hinge loss — adds clarity, also adds instability

I keep #4 off by default. It helps a lot at higher token rates and helps very little here.

## Configs

| File                | Token rate | Params | n_q | Notes                       |
| ------------------- | ---------- | ------ | --- | --------------------------- |
| `configs/tiny.yaml` | 75 Hz      | ~8M    | 4   | debug, trains fast          |
| `configs/base.yaml` | 50 Hz      | ~30M   | 8   | default                     |
| `configs/wide.yaml` | 50 Hz      | ~58M   | 8   | wider channels, better PESQ |

## Results

Internal test set (held-out LibriTTS-R dev-clean, 200 utts):

| Model            | Bitrate  | PESQ | STOI | Token-rate |
| ---------------- | -------- | ---- | ---- | ---------- |
| splitone-tiny    | 3.0 kbps | 2.81 | 0.89 | 75 Hz      |
| splitone-base    | 4.0 kbps | 3.24 | 0.93 | 50 Hz      |
| splitone-wide    | 4.0 kbps | 3.38 | 0.94 | 50 Hz      |

These are not state of the art. EnCodec and DAC are both better at the same bitrate; the
point of this repo is the small size and hackability, not the number.

## Repository layout

```
splitone-codec/
├── splitone/
│   ├── model/         # encoder, decoder, RVQ
│   ├── data/          # dataset + manifest utils
│   ├── losses.py      # STFT + L1 + adversarial
│   ├── train.py       # main loop
│   ├── encode.py      # CLI encode
│   ├── decode.py      # CLI decode
│   └── tokenize.py    # helpers for plugging into HF LMs
├── configs/           # YAML run configs
├── tests/             # pytest, mostly shape / round-trip tests
└── examples/          # short notebooks
```

## FAQ

**Why not just use EnCodec / DAC?**
Use them, they're better. I built this to learn, and to have something small enough to
modify without fear.

**Does it work on Chinese?**
Yes — the released checkpoint includes AISHELL-3. Mandarin sounds fine to my ear; the
PESQ on AISHELL-3 dev is around 3.1.

**Will you support 16 kHz / 48 kHz?**
There's a `--target_sr` flag on the encoder for 16 kHz. 48 kHz is not planned: the model
would need to be much bigger to be useful at that rate.

**Streaming?**
There's a causal variant; pass `--causal` at training time and use `encode_streaming`.
The receptive field is about 80 ms so the latency is small but the quality drop is real.

## Citing

If this is somehow useful in academic work:

```bibtex
@misc{xu2024splitone,
  author = {Xu, Mingrui},
  title  = {SplitOne-Codec: a small RVQ speech codec for LM experiments},
  year   = {2024},
  howpublished = {\url{https://github.com/marived/splitone-codec}}
}
```

## Acknowledgements

The encoder/decoder design borrows heavily from SoundStream and DAC. The RVQ
implementation owes a lot to `vector-quantize-pytorch`. The MS-STFT discriminator is
adapted from EnCodec.

## License

Apache 2.0. See [LICENSE](LICENSE).
