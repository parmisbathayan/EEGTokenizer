import unittest

import numpy as np

from src.config import PreprocessConfig
from src.preprocess import preprocess_eeg
from src.zuco_io import orient_eeg


class PreprocessTests(unittest.TestCase):
    def test_preprocessing_shape_scale_and_finiteness(self):
        rng = np.random.default_rng(4)
        eeg = rng.normal(scale=20, size=(105, 1250)).astype(np.float32)
        eeg[0, 20:23] = np.nan
        result = preprocess_eeg(eeg, PreprocessConfig())
        self.assertEqual(result.shape, (104, 400))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())

    def test_short_and_transposed_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "too_short"):
            preprocess_eeg(np.ones((105, 499)), PreprocessConfig())
        with self.assertRaisesRegex(ValueError, "transposed"):
            preprocess_eeg(np.ones((1000, 105)), PreprocessConfig())

    def test_orientation_prioritizes_known_channel_axis(self):
        self.assertEqual(orient_eeg(np.ones((105, 81))).shape, (105, 81))
        self.assertEqual(orient_eeg(np.ones((81, 105))).shape, (105, 81))


if __name__ == "__main__":
    unittest.main()
