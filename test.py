import unittest
import torch
from model import EncoderTransformer, DecoderOnlyTransformer, EncoderDecoderTransformer

VOCAB_SIZE  = 1000
MAX_SEQ_LEN = 512
D_MODEL     = 512
D_FF        = 2048
N_HEADS     = 8
N_BLOCKS    = 6

def make_tokens(B, T):
    return torch.randint(0, VOCAB_SIZE, (B, T))

def make_src_mask(tokens, pad_id=0):
    return (tokens == pad_id).unsqueeze(1)  # [B, 1, T]

def make_tgt_mask(tokens, pad_id=0):
    B, T = tokens.shape
    causal = torch.ones(T, T, dtype=torch.bool).triu(1).unsqueeze(0)  # [1, T, T]
    pad    = (tokens == pad_id).unsqueeze(1)                           # [B, 1, T]
    return causal | pad  # [B, T, T]


class TestEncoderTransformer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = EncoderTransformer(VOCAB_SIZE, MAX_SEQ_LEN, D_MODEL, D_FF, N_HEADS, N_BLOCKS)
        self.model.eval()

    def _run(self, B, T):
        src = make_tokens(B, T)
        out = self.model(src, make_src_mask(src))
        self.assertEqual(out.shape, (B, T, D_MODEL))

    def test_single_short(self):   self._run(1, 10)
    def test_batch_medium(self):   self._run(4, 64)
    def test_batch_max_len(self):  self._run(2, 512)


class TestDecoderOnlyTransformer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = DecoderOnlyTransformer(VOCAB_SIZE, MAX_SEQ_LEN, D_MODEL, D_FF, N_HEADS, N_BLOCKS)
        self.model.eval()

    def _run(self, B, T):
        tgt = make_tokens(B, T)
        out = self.model(tgt, make_src_mask(tgt))
        self.assertEqual(out.shape, (B, T, VOCAB_SIZE))

    def test_single_short(self):   self._run(1, 10)
    def test_batch_medium(self):   self._run(4, 64)
    def test_batch_max_len(self):  self._run(2, 512)

    def test_causal_mask(self):
        """Changing token at position t must not affect logits at positions < t."""
        x = make_tokens(1, 20)
        x_mod = x.clone()
        x_mod[0, 10] = (x_mod[0, 10] + 1) % VOCAB_SIZE
        with torch.no_grad():
            out1 = self.model(x)
            out2 = self.model(x_mod)
        self.assertTrue(torch.allclose(out1[0, :10], out2[0, :10]))


class TestEncoderDecoderTransformer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = EncoderDecoderTransformer(VOCAB_SIZE, MAX_SEQ_LEN, D_MODEL, D_FF, N_HEADS, N_BLOCKS, N_BLOCKS)
        self.model.eval()

    def _run(self, B, T_src, T_tgt):
        src = make_tokens(B, T_src)
        tgt = make_tokens(B, T_tgt)
        out = self.model(src, tgt, make_src_mask(src), make_tgt_mask(tgt))
        self.assertEqual(out.shape, (B, T_tgt, VOCAB_SIZE))

    def test_single_short(self):          self._run(1, 10, 8)
    def test_batch_unequal_lengths(self):  self._run(4, 64, 32)
    def test_batch_equal_lengths(self):    self._run(2, 128, 128)


if __name__ == "__main__":
    unittest.main()
