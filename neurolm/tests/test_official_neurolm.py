import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
