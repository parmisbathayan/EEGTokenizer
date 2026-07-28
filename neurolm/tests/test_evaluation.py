import tempfile
import unittest

import numpy as np

from src.config import EvaluationConfig
from src.evaluation import (
    ALIGNED,
    SHUFFLED,
    bootstrap_alignment_delta,
    evaluate_features,
    viability_gate,
)


class EvaluationTests(unittest.TestCase):
    def test_evaluation_outputs_paired_predictions(self):
        rng = np.random.default_rng(8)
        y = np.repeat(np.array([-1, 0, 1]), 12)
        X = rng.normal(size=(len(y), 8))
        X[:, 0] += y * 2.5
        sentence_ids = np.arange(len(y))
        config = EvaluationConfig(
            seeds=(42,), n_splits=3, inner_splits=2, c_values=(0.1,),
            bootstrap_samples=20, minimum_positive_seeds=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            metrics, predictions, _ = evaluate_features(
                X, y, sentence_ids, directory, config
            )
        self.assertEqual(set(metrics.setup), {ALIGNED, SHUFFLED, "majority"})
        self.assertEqual(
            len(predictions[predictions.setup == ALIGNED]), len(sentence_ids)
        )
        delta = bootstrap_alignment_delta(
            predictions, samples=20, confidence=config.bootstrap_ci
        )
        gate = viability_gate(metrics, delta, config)
        self.assertIn("seed_deltas", delta)
        self.assertIn("decision", gate)


if __name__ == "__main__":
    unittest.main()
