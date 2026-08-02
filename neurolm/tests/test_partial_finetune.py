import unittest

import numpy as np

from src.partial_finetune import (
    PartialFinetuneConfig,
    one_reader_per_sentence,
    selected_trainable_blocks,
)
from src.raw_cache import RawEEGRecord
from src.raw_eegnet import make_bundle_examples


def records():
    rows = []
    for sentence_id, label in enumerate((-1, 0, 1), start=1):
        for reader in range(3):
            rows.append(
                RawEEGRecord(
                    subject=f"S{reader}",
                    sentence_id=sentence_id,
                    label=label,
                    eeg=np.zeros((104, 200), dtype=np.float16),
                )
            )
    return rows


class PartialFinetuneTests(unittest.TestCase):
    def test_final_block_selection(self):
        self.assertEqual(selected_trainable_blocks(12, 2), (10, 11))
        with self.assertRaises(ValueError):
            selected_trainable_blocks(12, 13)

    def test_reader_resampling_keeps_one_full_weight_row_per_sentence(self):
        examples = make_bundle_examples(records(), [1, 2, 3], [-1, 0, 1])
        sampled = one_reader_per_sentence(examples, seed=5)
        self.assertEqual(len(sampled), 3)
        self.assertEqual(
            [row.target_sentence_id for row in sampled],
            [1, 2, 3],
        )
        self.assertTrue(all(abs(row.weight - 1.0) < 1e-7 for row in sampled))

    def test_locked_resource_configuration(self):
        config = PartialFinetuneConfig()
        self.assertEqual(config.top_gpt2_blocks, 2)
        self.assertEqual(config.seeds, (42, 52, 62))
        self.assertEqual(config.n_splits, 5)
        self.assertEqual(config.confidence, 0.95)


if __name__ == "__main__":
    unittest.main()
