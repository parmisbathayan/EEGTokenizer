import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.structured_cache import _load_structured_pack, _write_subject_pack


class StructuredCacheTests(unittest.TestCase):
    def test_subject_pack_round_trip_preserves_both_token_axes(self):
        rows = [
            (1, -1, np.ones((3, 4), dtype=np.float32), np.ones((2, 4))),
            (2, 1, np.full((3, 4), 2, dtype=np.float32), np.full((5, 4), 3)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_pack = root / "raw.npz"
            raw_pack.write_bytes(b"raw-pack-metadata-test")
            path = root / "structured.npz"
            _write_subject_pack(
                path,
                "S01",
                rows,
                "signature",
                raw_pack,
                expected_channels=3,
                embedding_size=4,
            )
            records, channel_backing, time_backing = _load_structured_pack(
                path, "signature"
            )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].channel_tokens.shape, (3, 4))
        self.assertEqual(records[0].time_tokens.shape, (2, 4))
        self.assertEqual(records[1].time_tokens.shape, (5, 4))
        self.assertEqual(channel_backing.dtype, np.float16)
        self.assertEqual(time_backing.dtype, np.float16)


if __name__ == "__main__":
    unittest.main()
