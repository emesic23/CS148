import unittest
import torch
import os
import pickle as pkl

from positional_encoder.sin_cos_encoder import SinCosPosEncoder
from positional_encoder.sin_cos_concat_encoder import SinCosConcatTextEncoder, SinCosConcatPosEncoder
from positional_encoder.index_encoder import IndexTextEncoder, IndexPosEncoder
from positional_encoder.learned_encoder import LearnedPosEncoder


class TestPosEncoder(unittest.TestCase):
    seed = 0

    def test_a_sincos1(self):
        seed = TestPosEncoder.seed
        max_seq_len = 1000
        seq_len = 100
        d_model = 512
        batch_size = 64

        torch.manual_seed(seed)
        with torch.no_grad():
            encoder = SinCosPosEncoder(d_model=d_model, max_seq_len=max_seq_len)
            input = torch.rand([seq_len, batch_size, d_model], dtype=torch.float32)
            output = encoder(input)

        with open('part_2_test_files/sincos1.pkl', 'rb') as f:
            corr_outputs = pkl.load(f)

        self.assertTrue(torch.allclose(output, corr_outputs, atol=1e-5), f"error: {torch.max(torch.abs(output-corr_outputs))}")

    def test_a_sincos2(self):
        seed = TestPosEncoder.seed
        max_seq_len = 5000
        seq_len = 100
        d_model = 128
        batch_size = 32

        torch.manual_seed(seed)
        with torch.no_grad():
            encoder = SinCosPosEncoder(d_model=d_model, max_seq_len=max_seq_len)
            input = torch.rand([seq_len, batch_size, d_model], dtype=torch.float32)
            output = encoder(input)

        with open('part_2_test_files/sincos2.pkl', 'rb') as f:
            corr_outputs = pkl.load(f)

        self.assertTrue(torch.allclose(output, corr_outputs, atol=1e-5), f"error: {torch.max(torch.abs(output-corr_outputs))}")

    def test_a_sincos3_not_parameter(self):
        seed = TestPosEncoder.seed
        max_seq_len = 5000
        d_model = 128

        torch.manual_seed(seed)
        with torch.no_grad():
            encoder = SinCosPosEncoder(d_model=d_model, max_seq_len=max_seq_len)

        self.assertFalse(isinstance(encoder.positional_encoding, torch.nn.parameter.Parameter))

    def test_b_sincos_concat_text(self):
        seed = TestPosEncoder.seed
        seq_len = 100
        d_model = 512
        batch_size = 64
        n_tokens = 28782
        init_range = 0.1

        torch.manual_seed(seed)
        with torch.no_grad():
            encoder = SinCosConcatTextEncoder(n_tokens=n_tokens, d_model=d_model, init_range=init_range)
            input = torch.randint(low=0, high=n_tokens, size=[seq_len, batch_size], dtype=torch.int32)
            output = encoder(input)

        with open('part_2_test_files/sincos_concat_text.pkl', 'rb') as f:
            corr_outputs = pkl.load(f)

        self.assertTrue(torch.allclose(output, corr_outputs, atol=1e-5))

    def test_b_sincos_concat_pos(self):
        seed = TestPosEncoder.seed
        max_seq_len = 5000
        seq_len = 100
        d_model = 512
        batch_size = 64

        torch.manual_seed(seed)
        with torch.no_grad():
            encoder = SinCosConcatPosEncoder(d_model=d_model, max_seq_len=max_seq_len)
            input = torch.rand([seq_len, batch_size, d_model], dtype=torch.float32)
            output = encoder(input)

        with open('part_2_test_files/sincos_concat_pos.pkl', 'rb') as f:
            corr_outputs = pkl.load(f)

        self.assertTrue(torch.allclose(output, corr_outputs, atol=1e-5))

    def test_b_sincos_concat_pos_not_parameter(self):
        seed = TestPosEncoder.seed
        max_seq_len = 5000
        d_model = 128

        torch.manual_seed(seed)
        with torch.no_grad():
            encoder = SinCosConcatPosEncoder(d_model=d_model, max_seq_len=max_seq_len)

        self.assertFalse(isinstance(encoder.positional_encoding, torch.nn.parameter.Parameter))

    def test_c_idx_text(self):
        seed = TestPosEncoder.seed
        seq_len = 100
        d_model = 512
        batch_size = 64
        n_tokens = 28782
        init_range = 0.1

        torch.manual_seed(seed)
        with torch.no_grad():
            encoder = IndexTextEncoder(n_tokens=n_tokens, d_model=d_model, init_range=init_range)
            input = torch.randint(low=0, high=n_tokens, size=[seq_len, batch_size], dtype=torch.int32)
            output = encoder(input)

        with open('part_2_test_files/idx_text.pkl', 'rb') as f:
            corr_outputs = pkl.load(f)

        self.assertTrue(torch.allclose(output, corr_outputs, atol=1e-5))

    def test_c_idx_pos(self):
        seed = TestPosEncoder.seed
        max_seq_len = 5000
        seq_len = 100
        d_model = 512
        batch_size = 64

        torch.manual_seed(seed)
        with torch.no_grad():
            encoder = IndexPosEncoder(d_model=d_model, max_seq_len=max_seq_len)
            input = torch.rand([seq_len, batch_size, d_model], dtype=torch.float32)
            output = encoder(input)

        with open('part_2_test_files/idx_pos.pkl', 'rb') as f:
            corr_outputs = pkl.load(f)

        self.assertTrue(torch.allclose(output, corr_outputs, atol=1e-5))

    def test_c_idx_pos_not_parameter(self):
        seed = TestPosEncoder.seed
        max_seq_len = 5000
        d_model = 128

        torch.manual_seed(seed)
        with torch.no_grad():
            encoder = IndexPosEncoder(d_model=d_model, max_seq_len=max_seq_len)

        self.assertFalse(isinstance(encoder.positional_encoding, torch.nn.parameter.Parameter))

    def test_d_learned_pos_is_uniform(self):
        seed = TestPosEncoder.seed
        max_seq_len = 5000
        d_model = 512

        torch.manual_seed(seed)
        with torch.no_grad():
            encoder = LearnedPosEncoder(d_model=d_model, max_seq_len=max_seq_len)

        self.assertTrue(torch.allclose(encoder.positional_encoding.mean(), torch.tensor([0.0]), atol=1e-2))
        self.assertTrue(torch.allclose(encoder.positional_encoding.std(), torch.tensor([0.577]), atol=1e-2))

    def test_d_learned_pos_is_parameter(self):
        seed = TestPosEncoder.seed
        max_seq_len = 5000
        d_model = 512

        torch.manual_seed(seed)
        with torch.no_grad():
            encoder = LearnedPosEncoder(d_model=d_model, max_seq_len=max_seq_len)

        self.assertTrue(isinstance(encoder.positional_encoding, torch.nn.parameter.Parameter))


if __name__ == '__main__':
    unittest.main()
