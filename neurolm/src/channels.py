"""Auditable mapping from ZuCo's EGI montage to NeuroLM channel embeddings."""

import numpy as np
import pandas as pd


# Exact order published in the official NeuroLM dataset loader.
NEUROLM_CHANNELS = (
    "FP1", "FPZ", "FP2", "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2",
    "AF4", "AF6", "AF8", "AF10", "F9", "F7", "F5", "F3", "F1", "FZ", "F2",
    "F4", "F6", "F8", "F10", "FT9", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "FT10", "T9", "T7", "C5", "C3", "C1", "CZ", "C2",
    "C4", "C6", "T8", "T10", "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2",
    "CP4", "CP6", "TP8", "TP10", "P9", "P7", "P5", "P3", "P1", "PZ", "P2",
    "P4", "P6", "P8", "P10", "PO9", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2",
    "PO4", "PO6", "PO8", "PO10", "O1", "OZ", "O2", "O9", "CB1", "CB2", "IZ",
    "O10", "T3", "T5", "T4", "T6", "M1", "M2", "A1", "A2", "CFC1", "CFC2",
    "CFC3", "CFC4", "CFC5", "CFC6", "CFC7", "CFC8", "CCP1", "CCP2", "CCP3",
    "CCP4", "CCP5", "CCP6", "CCP7", "CCP8", "T1", "T2", "FTT9H", "TTP7H",
    "TPP9H", "FTT10H", "TPP8H", "TPP10H", "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "PAD", "I1", "I2",
)

# ZuCo retains these 104 numbered EGI sensors; its exported 105th channel is flat Cz.
ZUCO_EXCLUDED_EGI = frozenset(
    (1, 8, 14, 17, 21, 25, 32, 48, 49, 56, 63, 68, 73, 81, 88, 94, 99,
     107, 113, 119, 125, 126, 127, 128)
)
MAX_MAPPING_DISTANCE_DEG = 30.0
MIN_MAPPED_CHANNELS = 80


def zuco_signal_channel_names():
    names = tuple(f"E{index}" for index in range(1, 129) if index not in ZUCO_EXCLUDED_EGI)
    if len(names) != 104:
        raise AssertionError(f"expected 104 retained ZuCo signal channels, got {len(names)}")
    return names


def _unit(position):
    value = np.asarray(position, dtype=np.float64)
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"invalid channel position {position}")
    return value / norm


def build_spatial_mapping(source_positions, target_positions):
    """Assign every ZuCo EGI sensor to a unique nearest NeuroLM-supported name."""

    from scipy.optimize import linear_sum_assignment

    source_names = zuco_signal_channel_names()
    missing_source = [name for name in source_names if name not in source_positions]
    if missing_source:
        raise ValueError(f"missing EGI positions: {missing_source[:5]}")

    target_by_upper = {str(name).upper(): name for name in target_positions}
    candidates = [name for name in NEUROLM_CHANNELS if name in target_by_upper]
    candidates = [name for name in candidates if "-" not in name and name not in {"PAD", "I1", "I2"}]
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) < len(source_names):
        raise ValueError(
            f"only {len(candidates)} positioned NeuroLM channels for {len(source_names)} ZuCo sensors"
        )

    source_xyz = np.stack([_unit(source_positions[name]) for name in source_names])
    target_xyz = np.stack([_unit(target_positions[target_by_upper[name]]) for name in candidates])
    cosine = np.clip(source_xyz @ target_xyz.T, -1.0, 1.0)
    source_index, target_index = linear_sum_assignment(1.0 - cosine)
    rows = []
    for src, dst in zip(source_index, target_index):
        target_name = candidates[dst]
        rows.append(
            {
                "zuco_index": int(src),
                "zuco_channel": source_names[src],
                "neurolm_channel": target_name,
                "neurolm_index": int(NEUROLM_CHANNELS.index(target_name)),
                "angular_distance_deg": float(np.degrees(np.arccos(cosine[src, dst]))),
            }
        )
    mapping = pd.DataFrame(rows).sort_values("zuco_index").reset_index(drop=True)
    if len(mapping) != 104 or mapping["neurolm_index"].duplicated().any():
        raise ValueError("spatial assignment must contain 104 unique target channels")
    return mapping


def build_mne_spatial_mapping():
    """Use MNE's bundled montages; no EEG data or positions are downloaded."""

    import mne

    source = mne.channels.make_standard_montage("GSN-HydroCel-128").get_positions()["ch_pos"]
    target = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    return build_spatial_mapping(source, target)


def select_usable_mapping(
    mapping,
    max_distance_deg=MAX_MAPPING_DISTANCE_DEG,
    min_channels=MIN_MAPPED_CHANNELS,
):
    """Keep spatially credible assignments and mark rejected rows for audit."""

    required = {
        "zuco_index",
        "zuco_channel",
        "neurolm_channel",
        "neurolm_index",
        "angular_distance_deg",
    }
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"mapping is missing columns: {sorted(missing)}")
    audited = mapping.copy()
    audited["use_for_encoder"] = (
        np.isfinite(audited["angular_distance_deg"])
        & (audited["angular_distance_deg"] <= max_distance_deg)
    )
    used = audited[audited["use_for_encoder"]].copy().reset_index(drop=True)
    if len(used) < min_channels:
        raise RuntimeError(
            f"only {len(used)} channels satisfy the {max_distance_deg:g}° mapping limit; "
            f"minimum is {min_channels}"
        )
    if used["zuco_index"].duplicated().any() or used["neurolm_index"].duplicated().any():
        raise ValueError("usable mapping must remain one-to-one")
    return audited, used
