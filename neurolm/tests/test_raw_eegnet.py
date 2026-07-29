import unittest

import numpy as np

from src.raw_cache import RawEEGRecord
from src.raw_eegnet import (
    RawEEGNetConfig,
    build_raw_eegnet,
    channel_statistics,
    make_bundle_examples,
    sentence_table,
    window_recording,
)


def example_records():
    records = []
    labels = (-1, 0, 1, -1, 0, 1)
    for sentence_id, label in enumerate(labels, start=1):
        for reader in range(2):
            base = sentence_id * 10 + reader
            eeg = np.arange(104 * 400, dtype=np.float32).reshape(104, 400)
            eeg = eeg / 1000 + base
            records.append(RawEEGRecord(f"S{reader}", sentence_id, label, eeg))
    return records


class RawEEGNetDataTests(unittest.TestCase):
    def test_bundle_shuffle_keeps_targets_and_permutes_whole_sentences(self):
        records = example_records()
        sentence_ids, labels = sentence_table(records)
        examples = make_bundle_examples(
            records, sentence_ids, labels, shuffled=True, seed=17
        )
        targets = sorted({example.target_sentence_id for example in examples})
        sources = sorted({example.record.sentence_id for example in examples})
        self.assertEqual(targets, sentence_ids.tolist())
        self.assertEqual(sources, sentence_ids.tolist())
        for target in sentence_ids:
            rows = [row for row in examples if row.target_sentence_id == target]
            self.assertEqual(len({row.record.sentence_id for row in rows}), 1)
            self.assertEqual(len(rows), 2)

    def test_windowing_and_temporal_shuffle_preserve_values(self):
        records = example_records()
        examples = make_bundle_examples(records, [1, 2, 3], [-1, 0, 1])
        mean, std = channel_statistics(examples)
        config = RawEEGNetConfig(normalization_clip=1e6)
        ordinary = window_recording(records[0], mean, std, config)
        shuffled = window_recording(
            records[0], mean, std, config, temporal_shuffle=True, shuffle_seed=9
        )
        self.assertEqual(ordinary.shape, (2, 104, 200))
        self.assertEqual(shuffled.shape, ordinary.shape)
        np.testing.assert_allclose(
            np.sort(ordinary, axis=-1), np.sort(shuffled, axis=-1), rtol=0, atol=1e-6
        )
        self.assertFalse(np.array_equal(ordinary, shuffled))

    def test_model_returns_one_logit_vector_per_window(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is supplied by Colab")
        config = RawEEGNetConfig()
        model = build_raw_eegnet(config).eval()
        with torch.inference_mode():
            logits = model(torch.zeros(3, 104, 200))
        self.assertEqual(tuple(logits.shape), (3, 3))


if __name__ == "__main__":
    unittest.main()
