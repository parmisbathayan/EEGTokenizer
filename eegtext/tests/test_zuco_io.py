from pathlib import Path
import unittest

import numpy as np

from src.zuco_io import orient_eeg, subject_from_path


class ZuCoIOTests(unittest.TestCase):
    def test_subject_from_standard_task_names(self):
        self.assertEqual(subject_from_path(Path("resultsZAB_SR.mat")), "ZAB")
        self.assertEqual(subject_from_path(Path("resultsYTL_TSR.mat")), "YTL")

    def test_orientation_prefers_known_channel_axis(self):
        channels_first = np.zeros((105, 700))
        samples_first = np.zeros((700, 105))
        self.assertEqual(orient_eeg(channels_first).shape, (105, 700))
        self.assertEqual(orient_eeg(samples_first).shape, (105, 700))

    def test_short_recording_is_not_accidentally_transposed(self):
        short = np.zeros((105, 39))
        self.assertEqual(orient_eeg(short).shape, (105, 39))

    def test_non_matrix_is_missing(self):
        self.assertIsNone(orient_eeg(np.zeros(20)))


if __name__ == "__main__":
    unittest.main()
