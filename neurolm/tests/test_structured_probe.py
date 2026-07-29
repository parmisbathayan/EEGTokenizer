import unittest
import importlib.util

import numpy as np

from src.structured_cache import StructuredRecord
from src.structured_probe import (
    StructuredCollator,
    StructuredProbeConfig,
    build_structured_probe,
    make_bundle_examples,
)


def example_records(channels=4, embedding=6):
    records = []
    labels = (-1, 0, 1, -1, 0, 1)
    for sentence_id, label in enumerate(labels, start=1):
        for reader in range(2):
            base = sentence_id * 100 + reader * 10
            channel = np.arange(channels * embedding, dtype=np.float32).reshape(
                channels, embedding
            ) + base
            time = np.arange((sentence_id % 3 + 1) * embedding, dtype=np.float32).reshape(
                sentence_id % 3 + 1, embedding
            ) + base
            records.append(
                StructuredRecord(f"S{reader}", sentence_id, label, channel, time)
            )
    return records


class StructuredProbeTests(unittest.TestCase):
    def test_shuffled_examples_keep_complete_reader_bundles(self):
        records = example_records()
        sentence_ids = np.arange(1, 7)
        labels = np.asarray([-1, 0, 1, -1, 0, 1])
        examples = make_bundle_examples(
            records, sentence_ids, labels, shuffled=True, seed=9
        )
        for target in sentence_ids:
            rows = [row for row in examples if row.target_sentence_id == target]
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row.record.sentence_id for row in rows}), 1)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is supplied by Colab")
    def test_structure_shuffle_preserves_token_values(self):
        records = example_records()
        config = StructuredProbeConfig(
            expected_channels=4, embedding_size=6, hidden_size=8, attention_size=4
        )
        examples = make_bundle_examples(records, [1, 2, 3], [-1, 0, 1])[:2]
        ordinary = StructuredCollator(config)(examples)
        shuffled = StructuredCollator(config, structure_shuffle=True, seed=5)(examples)
        np.testing.assert_allclose(
            np.sort(ordinary["channel_tokens"].numpy(), axis=1),
            np.sort(shuffled["channel_tokens"].numpy(), axis=1),
        )
        for row, example in enumerate(examples):
            length = example.record.time_tokens.shape[0]
            np.testing.assert_allclose(
                np.sort(ordinary["time_tokens"][row, :length].numpy(), axis=0),
                np.sort(shuffled["time_tokens"][row, :length].numpy(), axis=0),
            )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is supplied by Colab")
    def test_model_returns_three_logits(self):
        import torch
        config = StructuredProbeConfig(
            expected_channels=4, embedding_size=6, hidden_size=8, attention_size=4
        )
        model = build_structured_probe(config).eval()
        with torch.inference_mode():
            logits = model(
                torch.zeros(2, 4, 6),
                torch.zeros(2, 3, 6),
                torch.ones(2, 3, dtype=torch.bool),
            )
        self.assertEqual(tuple(logits.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
