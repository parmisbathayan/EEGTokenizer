import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.text_gpt2 import (
    ALIGNED,
    MODEL_NAME,
    MODEL_REVISION,
    SHUFFLED,
    TextGPT2Config,
    evaluate_text_features,
    load_feature_cache,
    save_feature_cache,
    sentence_fingerprint,
)


def _table():
    return pd.DataFrame(
        {
            "sentence_id": [2, 1, 3],
            "sentence": ["bad", "fine", "great"],
            "sentiment_label": [-1, 0, 1],
        }
    )


def _report(table):
    return {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "pooling": "final_non_padding_token_last_hidden_state",
        "max_length": 128,
        "dataset_fingerprint": sentence_fingerprint(table),
    }


class TextGPT2Tests(unittest.TestCase):
    def test_sentence_fingerprint_is_order_invariant_and_text_sensitive(self):
        table = _table()
        self.assertEqual(sentence_fingerprint(table), sentence_fingerprint(table.iloc[::-1]))
        changed = table.copy()
        changed.loc[0, "sentence"] = "very bad"
        self.assertNotEqual(sentence_fingerprint(table), sentence_fingerprint(changed))

    def test_feature_cache_round_trip_and_guards(self):
        table = _table()
        X = np.arange(12, dtype=np.float32).reshape(3, 4)
        y = table.sort_values("sentence_id")["sentiment_label"].to_numpy()
        sentence_ids = np.sort(table["sentence_id"].to_numpy())
        with tempfile.TemporaryDirectory() as directory:
            from pathlib import Path

            path = Path(directory) / "features.npz"
            save_feature_cache(path, X, y, sentence_ids, _report(table))
            loaded = load_feature_cache(
                path,
                expected_fingerprint=sentence_fingerprint(table),
                max_length=128,
            )
            np.testing.assert_array_equal(loaded[0], X)
            np.testing.assert_array_equal(loaded[1], y)
            np.testing.assert_array_equal(loaded[2], sentence_ids)

            bad = _report(table)
            bad["model_revision"] = "different"
            save_feature_cache(path, X, y, sentence_ids, bad)
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                load_feature_cache(path)

    def test_nested_text_evaluation_writes_sentence_level_results(self):
        rng = np.random.default_rng(5)
        y = np.repeat(np.asarray([-1, 0, 1]), 18)
        sentence_ids = np.arange(len(y))
        X = rng.normal(scale=0.1, size=(len(y), 6)).astype(np.float32)
        X[:, 0] += y * 3.0
        config = TextGPT2Config(
            seeds=(42,),
            n_splits=3,
            inner_splits=2,
            c_values=(0.1, 1.0),
            bootstrap_samples=25,
        )
        with tempfile.TemporaryDirectory() as directory:
            from pathlib import Path

            output_dir = Path(directory)
            metrics, predictions, _, delta = evaluate_text_features(
                X,
                y,
                sentence_ids,
                output_dir,
                extraction_report={"test": True},
                config=config,
            )
            self.assertEqual(set(metrics["setup"]), {ALIGNED, SHUFFLED, "majority"})
            self.assertEqual(len(predictions[predictions["setup"] == ALIGNED]), len(y))
            self.assertGreater(
                metrics.loc[metrics["setup"] == ALIGNED, "macro_f1"].mean(), 0.9
            )
            self.assertEqual(delta["comparison"], f"{ALIGNED}_minus_{SHUFFLED}")
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertEqual(
                json.loads((output_dir / "extraction_report.json").read_text()),
                {"test": True},
            )


if __name__ == "__main__":
    unittest.main()
