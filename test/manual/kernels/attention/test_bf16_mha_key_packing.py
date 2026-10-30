"""Bit-exact BF16 key packing for Kimi K3 TP8 prefill layouts.

Run in a CUDA SGLang environment:
    python test/manual/kernels/attention/test_bf16_mha_key_packing.py
"""

import unittest

import torch

from sglang.kernels.ops.attention.mla_qk_prep import concat_and_cast_mha_k_triton


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestBF16MHAKeyPacking(unittest.TestCase):
    @torch.inference_mode()
    def test_strided_inputs_match_two_copies_bitwise(self):
        codes = torch.arange(65536, device="cuda", dtype=torch.int32)
        patterns = (
            codes[(codes & 0x7F80) != 0x7F80].to(torch.int16).view(torch.bfloat16)
        )
        self.assertEqual(patterns.numel(), 65280)
        cases = [(t, 2112) for t in (1024, 8103, 9447, 16384)]
        cases += [(t, 64) for t in range(16384, 131073, 16384)]
        for tokens, rope_stride in cases:
            with self.subTest(tokens=tokens, rope_stride=rope_stride):
                self.check_layout(tokens, rope_stride, patterns)

    def check_layout(self, tokens, rope_stride, patterns):
        generator = torch.Generator(device="cuda").manual_seed(251025 + tokens)
        kv = torch.randn(
            (tokens, 12, 256), device="cuda", dtype=torch.bfloat16, generator=generator
        )
        nope = kv[..., :128]
        rope_storage = torch.randn(
            (tokens, rope_stride),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        rope = rope_storage[:, :64].view(tokens, 1, 64)
        # Include every finite BF16 pattern, including signed zero and subnormals.
        nope[:43].copy_(patterns.repeat(2)[: 43 * 12 * 128].view(43, 12, 128))
        rope[:1020].copy_(patterns.view(1020, 1, 64))
        expected = torch.empty((tokens, 12, 192), device="cuda", dtype=torch.bfloat16)
        expected[..., :128].copy_(nope)
        expected[..., 128:].copy_(rope)
        actual = torch.full_like(expected, float("nan"))
        concat_and_cast_mha_k_triton(actual, nope, rope)
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        )


if __name__ == "__main__":
    unittest.main()
