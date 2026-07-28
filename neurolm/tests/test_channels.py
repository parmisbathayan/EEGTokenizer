import unittest

import numpy as np

from src.channels import NEUROLM_CHANNELS, build_spatial_mapping, zuco_signal_channel_names


class ChannelMappingTests(unittest.TestCase):
    def test_zuco_retained_channel_count(self):
        names = zuco_signal_channel_names()
        self.assertEqual(len(names), 104)
        self.assertNotIn("E1", names)
        self.assertIn("E2", names)

    def test_assignment_is_one_to_one_and_ordered(self):
        sources = zuco_signal_channel_names()
        targets = [
            name for name in NEUROLM_CHANNELS
            if "-" not in name and name not in {"PAD", "I1", "I2"}
        ]
        angles = np.linspace(0, 2 * np.pi, len(targets), endpoint=False)
        target_positions = {
            name: np.array([np.cos(angle), np.sin(angle), 0.5])
            for name, angle in zip(targets, angles)
        }
        source_positions = {
            name: target_positions[targets[index]] for index, name in enumerate(sources)
        }
        mapping = build_spatial_mapping(source_positions, target_positions)
        self.assertEqual(mapping.zuco_channel.tolist(), list(sources))
        self.assertEqual(mapping.neurolm_index.nunique(), 104)
        self.assertLess(mapping.angular_distance_deg.max(), 1e-5)


if __name__ == "__main__":
    unittest.main()
