import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.gpt2_cache import (
    _load_gpt2_pack,
    _write_subject_pack,
    prepare_eeg_tokens,
    selected_seconds,
)


class GPT2CacheTests(unittest.TestCase):
    def test_uniform_seconds_cover_start_middle_end(self):
        np.testing.assert_array_equal(selected_seconds(1, 3), [0])
        np.testing.assert_array_equal(selected_seconds(3, 3), [0, 1, 2])
        np.testing.assert_array_equal(selected_seconds(29, 3), [0, 14, 28])

    def test_token_preparation_is_time_major_and_padded(self):
        eeg = np.zeros((104, 5 * 200), dtype=np.float32)
        for second in range(5):
            eeg[0, second * 200 : (second + 1) * 200] = second
            eeg[1, second * 200 : (second + 1) * 200] = second + 10
        patches, channels, times, valid, chosen = prepare_eeg_tokens(
            eeg, [0, 1], [7, 9], maximum_seconds=3
        )
        np.testing.assert_array_equal(chosen, [0, 2, 4])
        self.assertEqual(patches.shape, (6, 200))
        np.testing.assert_array_equal(channels, [7, 9, 7, 9, 7, 9])
        np.testing.assert_array_equal(times, [0, 0, 2, 2, 4, 4])
        self.assertTrue(valid.all())
        self.assertEqual(float(patches[2, 0]), 2.0)
        self.assertEqual(float(patches[5, 0]), 14.0)

    def test_subject_pack_round_trip(self):
        rows = [
            (1, -1, np.ones(4, dtype=np.float32)),
            (2, 1, np.full(4, 2, dtype=np.float32)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.npz"
            raw.write_bytes(b"raw")
            path = root / "v4.npz"
            _write_subject_pack(path, "S01", rows, "signature", raw, 4)
            records, hidden = _load_gpt2_pack(path, "signature")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].hidden.shape, (4,))
        self.assertEqual(hidden.dtype, np.float16)


if __name__ == "__main__":
    unittest.main()
