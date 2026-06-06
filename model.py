
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
        # x:           [B, T, d_model]
        # pe[:, :T]:   [1, T, d_model] → broadcasts over B
        return x + self.pe[:, :x.size(1)]
        # output:      [B, T, d_model]


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gelu = nn.GELU()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor):
        # x:        [B, T, d_model]
        x = self.linear1(x)   # [B, T, d_ff]
        x = self.gelu(x)      # [B, T, d_ff]
        x = self.linear2(x)   # [B, T, d_model]
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        d_head = d_model // n_heads
        self.heads = nn.ModuleList([
            SingleHeadCrossAttention(d_model, d_head, d_head) for _ in range(n_heads)
        ])
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, src: Tensor, tgt: Tensor, mask: BoolTensor=None):
        # src, tgt: [B, T, d_model], [B, T, d_model]
        return self.out_proj(torch.cat([head(src, tgt, mask) for head in self.heads], dim=-1))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        d_head = d_model // n_heads
        self.heads = nn.ModuleList([
            SingleHeadSelfAttention(d_model, d_head, d_head) for _ in range(n_heads)
        ])
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, mask: BoolTensor=None):
        # x:               [B, T, d_model]
        # each head(x, mask): [B, T, d_key]   where d_key = d_model // n_heads
        # cat(..., dim=-1): [B, T, n_heads * d_key] = [B, T, d_model]
        # out_proj:         [B, T, d_model]
        return self.out_proj(torch.cat([head(x, mask) for head in self.heads], dim=-1))


class SingleHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, d_key: int, d_val: int):
        super().__init__()
        self.softmax = nn.Softmax(-1)

        self.q_proj = nn.Linear(d_model, d_key, bias=False)
        self.k_proj = nn.Linear(d_model, d_key, bias=False)
        self.v_proj = nn.Linear(d_model, d_val, bias=False)

    def forward(self, src: Tensor, tgt: Tensor, mask: BoolTensor = None):
        # src, tgt: [B, T, d_model], [B, T, d_model]
        # mask: [B, 1, T_src] or [B, T_tgt, T_src], True = ignore
        Q, K, V = self.q_proj(tgt), self.k_proj(src), self.v_proj(src)
        d_key, d_val = K.size(-1), V.size(-1)
        attn_scores = torch.matmul(Q, torch.transpose(K, -1, -2)) / math.sqrt(d_key)
        # attn_scores: [B, T_tgt, T_src]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        # softmax(attn_scores): [B, T_tgt, T_src]
        return torch.matmul(self.softmax(attn_scores), V)


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, d_key: int, d_val: int):
        super().__init__()
        self.softmax = nn.Softmax(-1)

        self.q_proj = nn.Linear(d_model, d_key, bias=False)
        self.k_proj = nn.Linear(d_model, d_key, bias=False)
        self.v_proj = nn.Linear(d_model, d_val, bias=False)

    def forward(self, x: Tensor, mask: BoolTensor = None):
        # x:           [B, T, d_model]
        # mask: [B, T, T] or [1, T, T], True = ignore
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x) # Q, K, V: [B, T, d_key], [B, T, d_key], [B, T, d_val]
        d_key, d_val = K.size(-1), V.size(-1)
        attn_scores = torch.matmul(Q, torch.transpose(K, -1, -2)) / math.sqrt(d_key)
        # attn_scores: [B, T, T]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        # softmax(attn_scores): [B, T, T]
        return torch.matmul(self.softmax(attn_scores), V)
        # output:      [B, T, d_val]

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
        # tgt_mask: [B, T_tgt, T_tgt] causal+padding mask for self-attention over tgt
        # src_mask: [B, 1, T_src] padding mask for cross-attention over src
        carry = self.multi_head_self_attn(self.layer_norm1(tgt), tgt_mask)  # [B, T_tgt, d_model]
        tgt = carry + tgt
        carry = self.multi_head_cross_attn(src, self.layer_norm2(tgt), src_mask)
        tgt = carry + tgt
        carry = self.feed_forward(self.layer_norm3(tgt))
        tgt = carry + tgt
        return tgt


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.multi_head_attn = MultiHeadSelfAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, x: Tensor, mask: BoolTensor = None):
        # x:     [B, T, d_model]
        carry = self.multi_head_attn(self.layer_norm1(x), mask)  # [B, T, d_model]
        x = carry + x                                            # [B, T, d_model]
        carry = self.feed_forward(self.layer_norm2(x))           # [B, T, d_model]
        x = carry + x                                            # [B, T, d_model]
        return x



class DecoderOnlyTransformer(nn.Module):
    """
    An autoregressive, GPT-style, decoder-only transformer for text generation purposes.
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
        # x:        [B, T]  (token indices)
        # pad_mask: [B, 1, T], True = padding position (optional)
        T = x.size(1)
        # causal mask: upper triangle is True (future positions to ignore), shape [1, T, T]
        causal_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1).unsqueeze(0)
        mask = causal_mask | pad_mask if pad_mask is not None else causal_mask
        x = self.pos_encoding(self.embedding(x))    # [B, T, d_model]
        for attn_blk in self.attn_blocks:
            x = attn_blk(x, mask)   # [B, T, d_model]
        x = self.final_layer_norm(x)
        x = self.unembedding(x)     # [B, T, vocab_size]
        return x


class EncoderTransformer(nn.Module):
    """
    An encoder Transformer, with BERT-style bidirectionality.
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
        # x:                [B, T]  (token indices)
        x = self.pos_encoding(self.embedding(x))    # [B, T, d_model]
        for attn_blk in self.attn_blocks:  # repeated n_blocks times
            x = attn_blk(x, mask)   # [B, T, d_model]
        x = self.final_layer_norm(x)  # [B, T, d_model]
        return x

class DecoderTransformer(nn.Module):
    """
    A decoder Transformer, which assumes that the src argument already consists of encoded vectors.
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
        tgt = self.pos_encoding(self.embedding(tgt))    # [B, T_tgt, d_model]
        for attn_blk in self.attn_blocks:
            tgt = attn_blk(src, tgt, src_mask, tgt_mask)   # [B, T_tgt, d_model]
        tgt = self.final_layer_norm(tgt)
        tgt = self.unembedding(tgt)         # [B, T_tgt, vocab_size]
        return tgt

class EncoderDecoderTransformer(nn.Module):
    """
    An encoder-decoder Transformer.
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
        # src_mask: [B, 1, T_src] padding mask applied in cross-attention and encoder self-attention
        # tgt_mask: [B, T_tgt, T_tgt] causal+padding mask for decoder self-attention
        return self.decoder(self.encoder(src, src_mask), tgt, src_mask, tgt_mask)
    