import importlib.util
import unittest

from src.official_tfm import _tfm_stft


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is supplied by Colab")
class STFTTests(unittest.TestCase):
    def test_paper_stft_shape_and_magnitude(self):
        import torch

        eeg = torch.zeros(2, 4, 400)
        spectral = _tfm_stft(eeg, sampling_rate=200)
        self.assertEqual(tuple(spectral.shape), (2, 4, 101, 3))
        self.assertFalse(torch.is_complex(spectral))
        self.assertTrue(torch.isfinite(spectral).all())


if __name__ == "__main__":
    unittest.main()
