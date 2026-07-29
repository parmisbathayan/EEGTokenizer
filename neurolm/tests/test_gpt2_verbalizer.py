import importlib.util
import unittest

import numpy as np

from src.gpt2_cache import GPT2Record
from src.gpt2_verbalizer import (
    GPT2Collator,
    GPT2VerbalizerConfig,
    build_gpt2_verbalizer,
)
from src.raw_eegnet import make_bundle_examples


def example_records(embedding=6):
    records = []
    labels = (-1, 0, 1, -1, 0, 1)
    for sentence_id, label in enumerate(labels, start=1):
        for reader in range(2):
            hidden = np.arange(embedding, dtype=np.float32) + sentence_id + reader
            records.append(GPT2Record(f"S{reader}", sentence_id, label, hidden))
    return records


class GPT2VerbalizerTests(unittest.TestCase):
    def test_shuffled_examples_keep_reader_bundles(self):
        records = example_records()
        ids = np.arange(1, 7)
        labels = np.asarray([-1, 0, 1, -1, 0, 1])
        examples = make_bundle_examples(records, ids, labels, shuffled=True, seed=4)
        for target in ids:
            rows = [row for row in examples if row.target_sentence_id == target]
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row.record.sentence_id for row in rows}), 1)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is supplied by Colab")
    def test_adapter_returns_three_verbalizer_logits(self):
        import torch

        config = GPT2VerbalizerConfig(embedding_size=6, adapter_size=2)
        vectors = np.arange(18, dtype=np.float32).reshape(3, 6)
        model = build_gpt2_verbalizer(vectors, config).eval()
        examples = make_bundle_examples(example_records(), [1, 2, 3], [-1, 0, 1])[:2]
        batch = GPT2Collator(config)(examples)
        with torch.inference_mode():
            logits = model(batch["hidden"])
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            27,
        )


if __name__ == "__main__":
    unittest.main()
