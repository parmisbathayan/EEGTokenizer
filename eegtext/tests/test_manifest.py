from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.config import AuditConfig
from src.manifest import (
    _row_from_record,
    combine_manifests,
    read_manifest,
    summarize_manifest,
    write_audit,
)
from src.zuco_io import SentenceRecord


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.config = AuditConfig(task="NR", pattern="results*_NR.mat")

    def record(self, ordinal, text, eeg):
        return SentenceRecord("ZAB", ordinal, text, eeg, "/data/resultsZAB_NR.mat")

    def test_invalid_audit_threshold_fails_early(self):
        with self.assertRaisesRegex(ValueError, "must lie"):
            AuditConfig(maximum_nonfinite_fraction=1.1)

    def test_valid_record_has_stable_group_and_duration(self):
        eeg = np.zeros((105, 750), dtype=np.float32)
        row = _row_from_record(self.record(1, "A sentence.", eeg), self.config, {})
        duplicate = _row_from_record(
            self.record(2, "  a SENTENCE ", eeg), self.config, {}
        )
        self.assertTrue(row.usable)
        self.assertEqual(row.duration_seconds, 1.5)
        self.assertEqual(row.sentence_group_id, duplicate.sentence_group_id)

    def test_exclusions_are_explicit_and_composable(self):
        eeg = np.zeros((104, 300), dtype=np.float32)
        row = _row_from_record(self.record(1, "Text", eeg), self.config, {})
        self.assertFalse(row.usable)
        self.assertEqual(row.exclusion_reason, "unexpected_channels;too_short")

    def test_nonfinite_threshold_is_recorded(self):
        eeg = np.zeros((105, 500), dtype=np.float32)
        eeg[:, :125] = np.nan
        config = replace(self.config, maximum_nonfinite_fraction=0.20)
        row = _row_from_record(self.record(1, "Text", eeg), config, {})
        self.assertIn("too_many_nonfinite", row.exclusion_reason)

    def test_summary_counts_cross_task_text_once(self):
        eeg = np.zeros((105, 500), dtype=np.float32)
        nr = _row_from_record(self.record(1, "Repeated text", eeg), self.config, {})
        tsr_config = replace(self.config, task="TSR", pattern="results*_TSR.mat")
        tsr_record = SentenceRecord(
            "ZAB", 1, "Repeated text.", eeg, "/data/resultsZAB_TSR.mat"
        )
        tsr = _row_from_record(tsr_record, tsr_config, {})
        summary = summarize_manifest([nr, tsr])
        self.assertEqual(summary["unique_text_groups"], 1)
        self.assertEqual(summary["cross_context_text_groups"], 1)

    def test_audit_reports_are_written_atomically(self):
        eeg = np.zeros((105, 500), dtype=np.float32)
        row = _row_from_record(self.record(1, "Text", eeg), self.config, {})
        with tempfile.TemporaryDirectory() as directory:
            summary = write_audit([row], self.config, directory)
            paths = {path.name for path in Path(directory).iterdir()}
        self.assertEqual(
            paths, {"recordings.csv", "summary.json", "audit_config.json"}
        )
        self.assertEqual(summary["usable_recordings"], 1)

    def test_combined_manifest_recomputes_cross_task_duplicates(self):
        eeg = np.zeros((105, 500), dtype=np.float32)
        nr = _row_from_record(self.record(1, "Repeated text", eeg), self.config, {})
        tsr_config = replace(self.config, task="TSR", pattern="results*_TSR.mat")
        tsr = _row_from_record(
            SentenceRecord("ZAB", 1, "Repeated text", eeg, "/data/tsr.mat"),
            tsr_config,
            {},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_audit([nr], self.config, root / "nr")
            write_audit([tsr], tsr_config, root / "tsr")
            summary = combine_manifests(
                [root / "nr/recordings.csv", root / "tsr/recordings.csv"],
                root / "combined",
            )
            combined = read_manifest(root / "combined/recordings.csv")
        self.assertEqual(len(combined), 2)
        self.assertEqual(summary["cross_context_text_groups"], 1)


if __name__ == "__main__":
    unittest.main()
