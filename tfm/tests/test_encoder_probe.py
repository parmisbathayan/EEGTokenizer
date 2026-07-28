import importlib.util
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is supplied by Colab")
class EncoderFeatureCacheTests(unittest.TestCase):
    def test_channel_groups_use_all_channels_with_count_weighting(self):
        import numpy as np

        from src.encoder_probe import EncoderProbeConfig, OfficialFrozenTFMEncoder

        config = EncoderProbeConfig(
            codebook_size=100,
            embedding_size=2,
            expected_channels=10,
            channel_group_size=4,
            official_max_sequence_length=100,
        )
        encoder = OfficialFrozenTFMEncoder.__new__(OfficialFrozenTFMEncoder)
        encoder.config = config
        seen_channels = []

        def fake_encode_groups(groups):
            seen_channels.append(groups.shape[1])
            means = groups.mean(axis=(1, 2))
            return np.stack([means, means], axis=1)

        encoder._encode_groups = fake_encode_groups
        tokens = np.arange(10, dtype=np.uint16).reshape(1, 10, 1)
        features = encoder.encode(tokens)
        self.assertEqual(seen_channels, [4, 2])
        np.testing.assert_allclose(features, [[4.5, 4.5]])

    def test_extracts_reuses_and_sentence_pools_frozen_features(self):
        import numpy as np
        from pathlib import Path

        from src.encoder_probe import (
            EncoderProbeConfig,
            build_sentence_features,
            extract_or_load_encoder_features,
        )
        from src.token_map import TokenRecord

        config = EncoderProbeConfig(
            codebook_size=32,
            embedding_size=3,
            expected_channels=4,
            channel_group_size=2,
            extraction_batch_size=2,
            official_max_sequence_length=32,
            seeds=(7,),
            n_splits=2,
            inner_splits=2,
            c_values=(1.0,),
            bootstrap_samples=3,
        )

        class FakeEncoder:
            def __init__(self):
                self.report = {"checkpoint_sha256": "a" * 64}
                self.calls = 0

            def encode(self, token_maps):
                self.calls += 1
                means = token_maps.mean(axis=(1, 2))
                return np.stack([means, means + 1, means + 2], axis=1).astype(
                    np.float32
                )

        records = []
        for subject, offset in (("A", 0), ("B", 10)):
            for sentence_id, label in ((1, -1), (2, 1)):
                records.append(
                    TokenRecord(
                        subject=subject,
                        sentence_id=sentence_id,
                        label=label,
                        tokens=np.full(
                            (4, sentence_id + 1),
                            offset + sentence_id,
                            dtype=np.uint16,
                        ),
                        preprocess_hash="abc",
                        source_path=f"{subject}/{sentence_id}.npz",
                    )
                )
        encoder = FakeEncoder()
        with tempfile.TemporaryDirectory() as directory:
            features, metadata, report = extract_or_load_encoder_features(
                records,
                encoder,
                Path(directory),
                dataset_fingerprint="b" * 64,
                config=config,
            )
            first_call_count = encoder.calls
            reused_features, reused_metadata, reused_report = (
                extract_or_load_encoder_features(
                    records,
                    encoder,
                    Path(directory),
                    dataset_fingerprint="b" * 64,
                    config=config,
                )
            )
            self.assertEqual(encoder.calls, first_call_count)
            self.assertEqual(len(list(Path(directory).glob("*.npz"))), 2)
        np.testing.assert_array_equal(features, reused_features)
        self.assertTrue(metadata.equals(reused_metadata))
        self.assertEqual(
            report["feature_fingerprint"],
            reused_report["feature_fingerprint"],
        )
        sentence_features, y, sentence_metadata = build_sentence_features(
            features,
            metadata,
        )
        self.assertEqual(sentence_features.shape, (2, 3))
        np.testing.assert_array_equal(y, [-1, 1])
        np.testing.assert_array_equal(sentence_metadata["n_readers"], [2, 2])
        np.testing.assert_allclose(sentence_features[:, 0], [6, 7])

    def test_checkpoint_prefixes_are_removed_repeatedly(self):
        from src.encoder_probe import _unwrap_encoder_checkpoint

        value = object()
        state = _unwrap_encoder_checkpoint(
            {"state_dict": {"module.tfm_token.transformer.weight": value}}
        )
        self.assertIs(state["transformer.weight"], value)


if __name__ == "__main__":
    unittest.main()
