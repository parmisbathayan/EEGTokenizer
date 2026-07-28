import unittest

import numpy as np

from src.config import PreprocessConfig
from src.preprocess import preprocess_eeg


class PreprocessTests(unittest.TestCase):
    def test_resamples_drops_reference_and_repairs_nan(self):
        rng = np.random.default_rng(4)
        eeg = rng.normal(size=(105, 1000)).astype(np.float32)
        eeg[3, 20:25] = np.nan
        result = preprocess_eeg(eeg, PreprocessConfig())
        self.assertEqual(result.shape, (104, 400))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())

    def test_rejects_transposed_recording(self):
        with self.assertRaisesRegex(ValueError, "transposed"):
            preprocess_eeg(np.ones((1000, 105)), PreprocessConfig())


if __name__ == "__main__":
    unittest.main()

