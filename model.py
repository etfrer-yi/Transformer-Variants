"""
Implementations of encoder, decoder, and encoder-decoder Transformer architectures,
following the design of "Attention Is All You Need" (Vaswani et al., 2017).

Notation used in shape comments throughout this module:
  B       — batch size
  T       — sequence length (generic)
  T_src   — source sequence length
  T_tgt   — target sequence length
  d_model — model embedding dimension
  d_ff    — feed-forward hidden dimension
  d_key   — per-head key/query dimension  (= d_model // n_heads)
  d_val   — per-head value dimension      (= d_model // n_heads)

Mask convention: BoolTensor where True marks positions to ignore (filled with -inf).
"""

import math
import torch
from torch import Tensor, BoolTensor
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        pos = torch.arange(max_seq_len).unsqueeze(1)  # [T, 1]
        i = torch.arange(0, d_model, 2)               # [d_model/2]
        pe = torch.zeros(1, max_seq_len, d_model)      # [1, T, d_model]
        pe[0, :, 0::2] = torch.sin(pos / 10000 ** (i / d_model))
        pe[0, :, 1::2] = torch.cos(pos / 10000 ** (i / d_model))
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, d_model]
        return x + self.pe[:, :x.size(1)]  # [B, T, d_model]


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gelu = nn.GELU()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor):
        # x: [B, T, d_model]
        x = self.linear1(x)  # [B, T, d_ff]
        x = self.gelu(x)     # [B, T, d_ff]
        x = self.linear2(x)  # [B, T, d_model]
        return x


class SingleHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, d_key: int, d_val: int):
        super().__init__()
        self.softmax = nn.Softmax(-1)
        self.q_proj = nn.Linear(d_model, d_key, bias=False)
        self.k_proj = nn.Linear(d_model, d_key, bias=False)
        self.v_proj = nn.Linear(d_model, d_val, bias=False)

    def forward(self, src: Tensor, tgt: Tensor, mask: BoolTensor = None):
        # src: [B, T_src, d_model], tgt: [B, T_tgt, d_model], mask: [B, 1, T_src] or [B, T_tgt, T_src]
        Q = self.q_proj(tgt)                                           # [B, T_tgt, d_key]
        K = self.k_proj(src)                                           # [B, T_src, d_key]
        V = self.v_proj(src)                                           # [B, T_src, d_val]
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(K.size(-1))  # [B, T_tgt, T_src]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        return torch.matmul(self.softmax(attn_scores), V)              # [B, T_tgt, d_val]


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, d_key: int, d_val: int):
        super().__init__()
        self.softmax = nn.Softmax(-1)
        self.q_proj = nn.Linear(d_model, d_key, bias=False)
        self.k_proj = nn.Linear(d_model, d_key, bias=False)
        self.v_proj = nn.Linear(d_model, d_val, bias=False)

    def forward(self, x: Tensor, mask: BoolTensor = None):
        # x: [B, T, d_model], mask: [B, T, T] or [1, T, T]
        Q = self.q_proj(x)                                             # [B, T, d_key]
        K = self.k_proj(x)                                             # [B, T, d_key]
        V = self.v_proj(x)                                             # [B, T, d_val]
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(K.size(-1))  # [B, T, T]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        return torch.matmul(self.softmax(attn_scores), V)              # [B, T, d_val]


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        d_head = d_model // n_heads
        self.heads = nn.ModuleList([
            SingleHeadCrossAttention(d_model, d_head, d_head) for _ in range(n_heads)
        ])
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, src: Tensor, tgt: Tensor, mask: BoolTensor = None):
        # src: [B, T_src, d_model], tgt: [B, T_tgt, d_model], mask: [B, 1, T_src] or [B, T_tgt, T_src]
        return self.out_proj(torch.cat([head(src, tgt, mask) for head in self.heads], dim=-1))  # [B, T_tgt, d_model]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        d_head = d_model // n_heads
        self.heads = nn.ModuleList([
            SingleHeadSelfAttention(d_model, d_head, d_head) for _ in range(n_heads)
        ])
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, mask: BoolTensor = None):
        # x: [B, T, d_model], mask: [B, T, T] or [1, T, T]
        return self.out_proj(torch.cat([head(x, mask) for head in self.heads], dim=-1))  # [B, T, d_model]


class CrossAttentionTransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.multi_head_self_attn = MultiHeadSelfAttention(d_model, n_heads)
        self.multi_head_cross_attn = MultiHeadCrossAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: BoolTensor = None, tgt_mask: BoolTensor = None):
        # src: [B, T_src, d_model], tgt: [B, T_tgt, d_model], src_mask: [B, 1, T_src], tgt_mask: [B, T_tgt, T_tgt]
        carry = self.multi_head_self_attn(self.layer_norm1(tgt), tgt_mask)    # [B, T_tgt, d_model]
        tgt = carry + tgt                                                     # [B, T_tgt, d_model]
        carry = self.multi_head_cross_attn(src, self.layer_norm2(tgt), src_mask)  # [B, T_tgt, d_model]
        tgt = carry + tgt                                                     # [B, T_tgt, d_model]
        carry = self.feed_forward(self.layer_norm3(tgt))                      # [B, T_tgt, d_model]
        tgt = carry + tgt                                                     # [B, T_tgt, d_model]
        return tgt


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.multi_head_attn = MultiHeadSelfAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, x: Tensor, mask: BoolTensor = None):
        # x: [B, T, d_model], mask: [B, T, T] or [1, T, T]
        carry = self.multi_head_attn(self.layer_norm1(x), mask)  # [B, T, d_model]
        x = carry + x                                            # [B, T, d_model]
        carry = self.feed_forward(self.layer_norm2(x))           # [B, T, d_model]
        x = carry + x                                            # [B, T, d_model]
        return x


class DecoderOnlyTransformer(nn.Module):
    """
    An autoregressive, GPT-style, decoder-only transformer for text generation purposes.
    Importantly, returns outputs with dimension vocab_size, i.e. logits.
    Stand-alone, can be used on its own.
    """
    def __init__(
            self,
            vocab_size: int,
            max_seq_len: int,
            d_model: int,
            d_ff: int,
            n_heads: int,
            n_blocks: int,
        ):
        super().__init__()
        assert d_model % n_heads == 0
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_seq_len, d_model)
        self.attn_blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff, n_heads) for blk in range(n_blocks)
        ])
        self.final_layer_norm = nn.LayerNorm(d_model)
        # In decoder-only Transformer, we use the unembedding matrix to map to logits of the vocabulary
        self.unembedding = nn.Linear(d_model, vocab_size, bias=False)
        self.unembedding.weight = self.embedding.weight

    def forward(self, x: Tensor, pad_mask: BoolTensor = None):
        # x: [B, T] (token indices), pad_mask: [B, 1, T] (optional, True = padding)
        T = x.size(1)
        causal_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1).unsqueeze(0)  # [1, T, T]
        mask = causal_mask | pad_mask if pad_mask is not None else causal_mask                  # [B, T, T]
        x = self.pos_encoding(self.embedding(x))  # [B, T, d_model]
        for attn_blk in self.attn_blocks:
            x = attn_blk(x, mask)                 # [B, T, d_model]
        x = self.final_layer_norm(x)              # [B, T, d_model]
        x = self.unembedding(x)                   # [B, T, vocab_size]
        return x


class EncoderTransformer(nn.Module):
    """
    An encoder Transformer, with BERT-style bidirectionality.
    Importantly, returns outputs with dimension d_model, i.e. same as the input embeddings.
    Stand-alone, can be used on its own.
    """
    def __init__(
            self,
            vocab_size: int,
            max_seq_len: int,
            d_model: int,
            d_ff: int,
            n_heads: int,
            n_blocks: int,
        ):
        super().__init__()
        assert d_model % n_heads == 0
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_seq_len, d_model)
        self.attn_blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff, n_heads) for blk in range(n_blocks)
        ])
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: BoolTensor = None):
        # x: [B, T] (token indices), mask: [B, 1, T] (optional, True = padding)
        x = self.pos_encoding(self.embedding(x))  # [B, T, d_model]
        for attn_blk in self.attn_blocks:
            x = attn_blk(x, mask)                 # [B, T, d_model]
        x = self.final_layer_norm(x)              # [B, T, d_model]
        return x


class DecoderTransformer(nn.Module):
    """
    A decoder Transformer, which assumes that the src argument already consists of encoded vectors.
    Importantly, returns outputs with dimension vocab_size, i.e. logits.
    Not for stand-alone use.
    """
    def __init__(
            self,
            vocab_size: int,
            max_seq_len: int,
            d_model: int,
            d_ff: int,
            n_heads: int,
            n_blocks: int,
        ):
        super().__init__()
        assert d_model % n_heads == 0
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_seq_len, d_model)
        self.attn_blocks = nn.ModuleList([
            CrossAttentionTransformerBlock(d_model, d_ff, n_heads) for blk in range(n_blocks)
        ])
        self.final_layer_norm = nn.LayerNorm(d_model)
        # In decoder Transformer, we use the unembedding matrix to map to logits of the vocabulary
        self.unembedding = nn.Linear(d_model, vocab_size, bias=False)
        self.unembedding.weight = self.embedding.weight

    def forward(self, src: Tensor, tgt: Tensor, src_mask: BoolTensor = None, tgt_mask: BoolTensor = None):
        # src: [B, T_src, d_model] (encoded), tgt: [B, T_tgt] (token indices), src_mask: [B, 1, T_src], tgt_mask: [B, T_tgt, T_tgt]
        tgt = self.pos_encoding(self.embedding(tgt))      # [B, T_tgt, d_model]
        for attn_blk in self.attn_blocks:
            tgt = attn_blk(src, tgt, src_mask, tgt_mask)  # [B, T_tgt, d_model]
        tgt = self.final_layer_norm(tgt)                  # [B, T_tgt, d_model]
        tgt = self.unembedding(tgt)                       # [B, T_tgt, vocab_size]
        return tgt


class EncoderDecoderTransformer(nn.Module):
    """
    An encoder-decoder Transformer.
    Importantly, returns outputs with dimension vocab_size, i.e. logits.
    Stand-alone, can be used on its own.
    """
    def __init__(
            self,
            vocab_size: int,
            max_seq_len: int,
            d_model: int,
            d_ff: int,
            n_heads: int,
            n_blocks_encoder: int,
            n_blocks_decoder: int,
        ):
        super().__init__()
        assert d_model % n_heads == 0
        self.encoder = EncoderTransformer(vocab_size, max_seq_len, d_model, d_ff, n_heads, n_blocks_encoder)
        self.decoder = DecoderTransformer(vocab_size, max_seq_len, d_model, d_ff, n_heads, n_blocks_decoder)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: BoolTensor = None, tgt_mask: BoolTensor = None):
        # src: [B, T_src] (token indices), tgt: [B, T_tgt] (token indices), src_mask: [B, 1, T_src], tgt_mask: [B, T_tgt, T_tgt]
        return self.decoder(self.encoder(src, src_mask), tgt, src_mask, tgt_mask)  # [B, T_tgt, vocab_size]
