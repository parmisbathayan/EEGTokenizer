import importlib.util
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is supplied by Colab")
class TokenCacheTests(unittest.TestCase):
    def test_loads_subject_records_and_builds_a_content_fingerprint(self):
        import numpy as np
        from pathlib import Path

        from src.token_map import (
            TokenMapConfig,
            _integer_targets,
            load_or_pack_token_records,
            load_token_records,
        )

        config = TokenMapConfig(
            codebook_size=16,
            embedding_size=8,
            expected_channels=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            packed = Path(directory) / "packed"
            for subject, value in (("A", 1), ("B", 2)):
                path = root / subject / "sentence_0001.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    path,
                    tokens=np.full((4, 3), value, dtype=np.uint16),
                    subject=np.asarray(subject),
                    sentence_id=np.int64(1),
                    label=np.int64(0),
                    preprocess_hash=np.asarray("abc"),
                )
            records, metadata, report = load_token_records(root, config=config)
            self.assertEqual(len(records), 2)
            self.assertEqual(len(metadata), 2)
            self.assertEqual(report["n_sentences"], 1)
            self.assertEqual(report["minimum_readers_per_sentence"], 2)
            self.assertEqual(len(report["dataset_fingerprint"]), 64)
            self.assertEqual(records[0].tokens.dtype, np.uint16)
            packed_records, packed_metadata, packed_report = load_or_pack_token_records(
                root,
                packed,
                config=config,
                workers=2,
            )
            self.assertEqual(len(list(packed.glob("*.npz"))), 2)
            self.assertEqual(len(packed_records), len(records))
            self.assertEqual(len(packed_metadata), len(metadata))
            self.assertEqual(
                packed_report["dataset_fingerprint"],
                report["dataset_fingerprint"],
            )
            for original, reconstructed in zip(records, packed_records):
                np.testing.assert_array_equal(original.tokens, reconstructed.tokens)
            reused_records, _, reused_report = load_or_pack_token_records(
                root,
                packed,
                config=config,
                workers=2,
            )
            self.assertEqual(len(reused_records), len(records))
            self.assertEqual(
                reused_report["dataset_fingerprint"],
                report["dataset_fingerprint"],
            )
            object_targets = np.asarray([-1, 0, 1], dtype=object)
            normalized_targets = _integer_targets(object_targets)
            self.assertEqual(normalized_targets.dtype, np.dtype(np.int64))
            np.testing.assert_array_equal(normalized_targets, [-1, 0, 1])

    def test_bootstrap_normalizes_object_dtype_targets(self):
        import sys
        import types
        from unittest import mock

        import numpy as np
        import pandas as pd

        from src.token_map import TokenMapConfig, _bootstrap_delta

        metrics_module = types.ModuleType("sklearn.metrics")

        def checked_f1(y_true, y_pred, **kwargs):
            del kwargs
            self.assertEqual(np.asarray(y_true).dtype, np.dtype(np.int64))
            self.assertEqual(np.asarray(y_pred).dtype, np.dtype(np.int64))
            return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

        metrics_module.f1_score = checked_f1
        sklearn_module = types.ModuleType("sklearn")
        sklearn_module.metrics = metrics_module
        rows = []
        for setup, predicted in (
            ("token_map", [-1, 0, 1]),
            ("token_map_shuffled", [0, 0, 1]),
        ):
            for sentence_id, (label, prediction) in enumerate(
                zip([-1, 0, 1], predicted)
            ):
                rows.append(
                    {
                        "setup": setup,
                        "seed": 42,
                        "sentence_id": sentence_id,
                        "label": label,
                        "prediction": prediction,
                    }
                )
        predictions = pd.DataFrame(rows).astype(object)
        with mock.patch.dict(
            sys.modules,
            {"sklearn": sklearn_module, "sklearn.metrics": metrics_module},
        ):
            result = _bootstrap_delta(
                predictions,
                TokenMapConfig(seeds=(42,), bootstrap_samples=3),
            )
        self.assertEqual(result["bootstrap_samples"], 3)


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),
    "PyTorch and NumPy are supplied by Colab",
)
class TokenMapTests(unittest.TestCase):
    def test_extracts_codebook_directly_from_checkpoint(self):
        import torch
        from pathlib import Path

        from src.token_map import (
            TokenMapConfig,
            extract_frozen_codebook_from_checkpoint,
        )

        config = TokenMapConfig(
            codebook_size=16,
            embedding_size=8,
            expected_channels=4,
        )
        expected = torch.randn(16, 8)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "tokenizer.pt"
            torch.save(
                {"state_dict": {"vqvae.quantize.embedding.weight": expected}},
                checkpoint,
            )
            codebook, report = extract_frozen_codebook_from_checkpoint(
                checkpoint, config=config
            )
        self.assertTrue(torch.equal(codebook, expected))
        self.assertEqual(report["state_key"], "quantize.embedding.weight")
        self.assertEqual(report["source"], "checkpoint_state")

    def test_model_accepts_variable_length_masked_token_maps(self):
        import torch

        from src.token_map import TokenMapConfig, build_token_map_model

        config = TokenMapConfig(
            codebook_size=16,
            embedding_size=8,
            expected_channels=4,
            hidden_size=6,
        )
        codebook = torch.randn(16, 8)
        model = build_token_map_model(codebook, config=config)
        tokens = torch.randint(0, 16, (3, 4, 7))
        mask = torch.tensor(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, False, False, False],
                [True, True, False, False, False, False, False],
            ]
        )
        logits = model(tokens, mask)
        self.assertEqual(tuple(logits.shape), (3, 3))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertFalse(model.embedding.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
