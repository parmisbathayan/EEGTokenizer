import unittest

import numpy as np

from src.config import EncoderConfig
from src.official_neurolm import OfficialNeuroLMEncoder, select_tokenizer_state


class OfficialAdapterTests(unittest.TestCase):
    def test_only_tokenizer_subtree_is_selected(self):
        state = {
            "_orig_mod.tokenizer.layer.weight": 1,
            "tokenizer.layer.bias": 2,
            "GPT2.layer.weight": 3,
        }
        selected = select_tokenizer_state(state)
        self.assertEqual(list(selected), ["layer.weight", "layer.bias"])

    def test_pooling_has_fixed_three_part_shape(self):
        values = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
        feature = OfficialNeuroLMEncoder.pool_embeddings(values)
        self.assertEqual(feature.shape, (15,))
        self.assertTrue(np.isfinite(feature).all())

    def test_one_second_has_zero_slope(self):
        values = np.ones((1, 4, 3), dtype=np.float32)
        feature = OfficialNeuroLMEncoder.pool_embeddings(values)
        np.testing.assert_array_equal(feature[-3:], 0)

    def test_structured_tokens_preserve_v1_pooling_input(self):
        encoder = OfficialNeuroLMEncoder.__new__(OfficialNeuroLMEncoder)
        encoder.config = EncoderConfig(block_size=4, patch_samples=2)
        encoder.channel_ids = np.asarray([10, 20], dtype=np.int64)
        encoder.zuco_indices = np.asarray([0, 1], dtype=np.int64)

        def fake_encode(patches, channel_ids, time_ids):
            return np.stack(
                [patches.mean(axis=1), channel_ids, time_ids], axis=1
            ).astype(np.float32)

        encoder._encode_block = fake_encode
        eeg = np.zeros((104, 6), dtype=np.float32)
        eeg[0] = np.arange(6)
        eeg[1] = np.arange(6) + 10
        tokens, details = encoder.encode_recording_tokens(eeg)
        feature, pooled_details = encoder.encode_recording(eeg)
        self.assertEqual(tokens.shape, (3, 2, 3))
        self.assertEqual(details["seconds"], 3)
        np.testing.assert_allclose(feature, encoder.pool_embeddings(tokens))
        self.assertEqual(pooled_details["feature_dim"], 9)


if __name__ == "__main__":
    unittest.main()
