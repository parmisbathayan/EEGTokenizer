import csv
from pathlib import Path
import tempfile
import unittest

from src.labels import load_label_lookup
from src.text import normalize_text, text_hash


class TextTests(unittest.TestCase):
    def test_normalization_handles_quotes_punctuation_and_spacing(self):
        left = '  “A Movie”—isn’t... BAD!  '
        right = "a movie isn't bad"
        self.assertEqual(normalize_text(left), normalize_text(right))
        self.assertEqual(text_hash(left), text_hash(right))

    def test_label_lookup_uses_normalized_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sentence_id", "sentence", "sentiment_label"],
                )
                writer.writeheader()
                writer.writerow(
                    {"sentence_id": 7, "sentence": "A fine film.", "sentiment_label": 1}
                )
            lookup = load_label_lookup(path)
        self.assertEqual(lookup["a fine film"], (7, 1))

    def test_conflicting_duplicate_labels_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            path.write_text(
                "sentence_id,sentence,sentiment_label\n"
                "1,Same sentence,1\n"
                "1,Same sentence,-1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "conflicting labels"):
                load_label_lookup(path)


if __name__ == "__main__":
    unittest.main()
