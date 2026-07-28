import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.features import build_sentence_features


class FeatureAggregationTests(unittest.TestCase):
    def test_readers_are_averaged_at_sentence_level(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for subject, value in (("A", 1.0), ("B", 3.0)):
                folder = root / subject
                folder.mkdir()
                np.savez_compressed(
                    folder / "sentence_0007.npz",
                    feature=np.full(6, value, dtype=np.float16),
                    subject=np.asarray(subject), sentence_id=np.int64(7),
                    label=np.int64(1), channels=np.int64(104),
                    seconds=np.int64(2), patches=np.int64(208),
                )
            X, y, metadata, diagnostics = build_sentence_features(root)
            np.testing.assert_allclose(X, 2.0)
            self.assertEqual(y.tolist(), [1])
            self.assertEqual(metadata.n_subjects.tolist(), [2])
            self.assertEqual(diagnostics["n_recordings"], 2)


if __name__ == "__main__":
    unittest.main()
