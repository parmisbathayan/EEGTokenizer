import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is supplied by Colab")
class FinetuneProbeTests(unittest.TestCase):
    def test_shuffled_examples_are_a_within_split_derangement(self):
        import numpy as np

        from src.finetune_probe import _examples_for_split, _sentence_groups
        from src.token_map import TokenRecord

        records = []
        for sentence_id, label in ((1, -1), (2, 0), (3, 1), (4, -1)):
            for subject in ("A", "B"):
                records.append(
                    TokenRecord(
                        subject=subject,
                        sentence_id=sentence_id,
                        label=label,
                        tokens=np.full(
                            (4, sentence_id),
                            sentence_id,
                            dtype=np.uint16,
                        ),
                        preprocess_hash="abc",
                        source_path=f"{subject}/{sentence_id}.npz",
                    )
                )
        groups, sentence_ids, labels = _sentence_groups(records)
        self.assertEqual(sentence_ids.tolist(), [1, 2, 3, 4])
        self.assertEqual(labels.tolist(), [-1, 0, 1, -1])
        examples = _examples_for_split(
            groups,
            sentence_ids,
            shuffled=True,
            rng=np.random.default_rng(7),
        )
        targets = {example.sentence_id for example in examples}
        sources = {example.feature_sentence_id for example in examples}
        self.assertEqual(targets, sources)
        self.assertTrue(
            all(
                example.sentence_id != example.feature_sentence_id
                for example in examples
            )
        )
        self.assertTrue(
            all(
                all(
                    record.sentence_id == example.feature_sentence_id
                    for record in example.records
                )
                for example in examples
            )
        )

    def test_aligned_examples_preserve_sentence_and_label(self):
        import numpy as np

        from src.finetune_probe import _examples_for_split, _sentence_groups
        from src.token_map import TokenRecord

        records = [
            TokenRecord(
                subject="A",
                sentence_id=sentence_id,
                label=label,
                tokens=np.ones((4, 2), dtype=np.uint16),
                preprocess_hash="abc",
                source_path=f"A/{sentence_id}.npz",
            )
            for sentence_id, label in ((10, -1), (20, 1))
        ]
        groups, sentence_ids, _ = _sentence_groups(records)
        examples = _examples_for_split(
            groups,
            sentence_ids,
            shuffled=False,
            rng=np.random.default_rng(1),
        )
        self.assertEqual(
            [(item.sentence_id, item.feature_sentence_id, item.label) for item in examples],
            [(10, 10, -1), (20, 20, 1)],
        )

    def test_duplicate_subject_within_sentence_is_rejected(self):
        import numpy as np

        from src.finetune_probe import _sentence_groups
        from src.token_map import TokenRecord

        records = [
            TokenRecord(
                subject="A",
                sentence_id=1,
                label=0,
                tokens=np.ones((4, 2), dtype=np.uint16),
                preprocess_hash="abc",
                source_path=f"duplicate-{index}.npz",
            )
            for index in range(2)
        ]
        with self.assertRaisesRegex(ValueError, "duplicate subject"):
            _sentence_groups(records)


if __name__ == "__main__":
    unittest.main()
