import unittest

from src.splits import assert_disjoint_text_groups


class SplitTests(unittest.TestCase):
    def test_disjoint_partitions_pass(self):
        self.assertTrue(
            assert_disjoint_text_groups({"train": ["a", "b"], "test": ["c"]})
        )

    def test_duplicate_text_across_partitions_fails(self):
        with self.assertRaisesRegex(ValueError, "text groups cross partitions"):
            assert_disjoint_text_groups({"train": ["a", "b"], "test": ["b", "c"]})


if __name__ == "__main__":
    unittest.main()
