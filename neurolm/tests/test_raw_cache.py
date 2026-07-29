import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config import PreprocessConfig
from src.raw_cache import (
    _cache_signature,
    _json_equivalent,
    _load_subject_pack,
    _write_subject_pack,
)


class RawCacheTests(unittest.TestCase):
    def test_json_round_trip_config_is_equivalent(self):
        original = PreprocessConfig().to_dict()
        restored = json.loads(json.dumps(original))
        self.assertNotEqual(original, restored)
        self.assertTrue(_json_equivalent(original, restored))

    def test_subject_pack_round_trip_preserves_variable_lengths(self):
        config = PreprocessConfig()
        signature = _cache_signature(config)
        recordings = [
            (1, -1, np.arange(104 * 200, dtype=np.float32).reshape(104, 200)),
            (2, 1, np.ones((104, 400), dtype=np.float32)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "S01.npz"
            _write_subject_pack(path, "S01", recordings, signature, config)
            loaded, loaded_signature, backing = _load_subject_pack(
                path, expected_signature=signature
            )
        self.assertEqual(loaded_signature, signature)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].eeg.shape, (104, 200))
        self.assertEqual(loaded[1].eeg.shape, (104, 400))
        self.assertEqual(loaded[0].sentence_id, 1)
        self.assertEqual(loaded[1].label, 1)
        self.assertEqual(backing.dtype, np.float16)


if __name__ == "__main__":
    unittest.main()
