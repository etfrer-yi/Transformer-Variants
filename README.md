# Transformer Variants

Implementations of encoder, decoder, and encoder-decoder Transformer architectures in PyTorch, following "Attention Is All You Need" (Vaswani et al., 2017).

## Installation

```bash
pip install transformers-from-scratch
```

PyTorch is required but not installed automatically (to allow users to choose their CUDA build). Install it first from [pytorch.org](https://pytorch.org/get-started/locally/).

## Development Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Modules

The `transformer_variants` package provides three stand-alone models and the building blocks they are composed of.

| Class | Description |
|---|---|
| `EncoderTransformer` | BERT-style bidirectional encoder |
| `DecoderOnlyTransformer` | GPT-style autoregressive decoder |
| `EncoderDecoderTransformer` | Sequence-to-sequence encoder-decoder |
| `DecoderTransformer` | Decoder component — not for stand-alone use |

## Usage

```python
import torch
from transformer_variants import EncoderTransformer, DecoderOnlyTransformer, EncoderDecoderTransformer

VOCAB_SIZE, MAX_SEQ_LEN = 32000, 512
D_MODEL, D_FF, N_HEADS, N_BLOCKS = 512, 2048, 8, 6
```

### Encoder

```python
model = EncoderTransformer(VOCAB_SIZE, MAX_SEQ_LEN, D_MODEL, D_FF, N_HEADS, N_BLOCKS)

src = torch.randint(0, VOCAB_SIZE, (B, T_src))                        # [B, T_src]
src_mask = (src == pad_id).unsqueeze(1).expand(-1, T_src, -1)         # [B, T_src, T_src]  True = padding

out = model(src, src_mask)                                            # [B, T_src, d_model]
```

### Decoder-only

```python
model = DecoderOnlyTransformer(VOCAB_SIZE, MAX_SEQ_LEN, D_MODEL, D_FF, N_HEADS, N_BLOCKS)

tgt = torch.randint(0, VOCAB_SIZE, (B, T))              # [B, T]
pad_mask = (tgt == pad_id).unsqueeze(1)                 # [B, 1, T]  True = padding (optional)

logits = model(tgt, pad_mask)                           # [B, T, vocab_size]
```

A causal mask is generated internally — no need to pass one.

### Encoder-decoder

```python
model = EncoderDecoderTransformer(VOCAB_SIZE, MAX_SEQ_LEN, D_MODEL, D_FF, N_HEADS, N_BLOCKS, N_BLOCKS)

src = torch.randint(0, VOCAB_SIZE, (B, T_src))          # [B, T_src]
tgt = torch.randint(0, VOCAB_SIZE, (B, T_tgt))          # [B, T_tgt]

src_mask = (src == pad_id).unsqueeze(1)                 # [B, 1, T_src]
T_tgt = tgt.size(1)
causal = torch.ones(T_tgt, T_tgt, dtype=torch.bool).triu(1).unsqueeze(0)  # [1, T_tgt, T_tgt]
tgt_mask = causal | (tgt == pad_id).unsqueeze(1)        # [B, T_tgt, T_tgt]

logits = model(src, tgt, src_mask, tgt_mask)            # [B, T_tgt, vocab_size]
```

## Running tests

```bash
python -m unittest test -v
```
