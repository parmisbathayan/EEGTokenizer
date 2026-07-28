import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.features import build_sentence_histograms


class FeatureTests(unittest.TestCase):
    def test_subjects_receive_equal_weight_per_sentence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples = [
                ("S1", 10, -1, [[0, 0, 1]]),
                ("S2", 10, -1, [[1, 1]]),
                ("S1", 20, 1, [[2, 3]]),
            ]
            for subject, sentence_id, label, tokens in examples:
                path = root / subject / f"sentence_{sentence_id:04d}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    path,
                    tokens=np.asarray(tokens, dtype=np.uint16),
                    subject=np.asarray(subject),
                    sentence_id=np.int64(sentence_id),
                    label=np.int64(label),
                )
            X, y, metadata, diagnostics = build_sentence_histograms(root, codebook_size=4)
            np.testing.assert_allclose(X[0], [1 / 3, 2 / 3, 0, 0])
            np.testing.assert_array_equal(y, [-1, 1])
            self.assertEqual(metadata.loc[0, "n_subjects"], 2)
            self.assertEqual(diagnostics["used_tokens"], 4)


if __name__ == "__main__":
    unittest.main()

