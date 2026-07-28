import tempfile
import unittest

import numpy as np

from src.config import EvaluationConfig
from src.evaluation import bootstrap_alignment_delta, evaluate_histograms


class EvaluationTests(unittest.TestCase):
    def test_writes_one_oof_prediction_per_sentence_and_setup(self):
        rng = np.random.default_rng(7)
        y = np.repeat([-1, 0, 1], 12)
        X = rng.random((len(y), 18)).astype(np.float32)
        X[np.arange(len(y)), (y + 1) * 4] += 3
        config = EvaluationConfig(
            seeds=(3,), n_splits=3, inner_splits=2, c_values=(1.0,), bootstrap_samples=20
        )
        with tempfile.TemporaryDirectory() as output:
            metrics, predictions, _ = evaluate_histograms(
                X, y, np.arange(len(y)), output, config
            )
            self.assertEqual(len(metrics), 9)
            counts = predictions.groupby("setup").size().to_dict()
            self.assertEqual(counts["tfm_histogram"], len(y))
            self.assertEqual(counts["tfm_histogram_shuffled"], len(y))
            delta = bootstrap_alignment_delta(predictions, samples=20, seed=9)
            self.assertIn("ci_95_low", delta)


if __name__ == "__main__":
    unittest.main()

