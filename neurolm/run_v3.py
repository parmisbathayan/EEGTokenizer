"""Command-line helper for V3; the dedicated Colab notebook is primary."""

import argparse
import json

from src.channels import build_mne_spatial_mapping, select_usable_mapping
from src.official_neurolm import OfficialNeuroLMEncoder
from src.structured_cache import (
    extract_structured_subject_packs,
    load_structured_records,
)
from src.structured_probe import (
    StructuredProbeConfig,
    evaluate_structured_probe,
    smoke_test_structured_probe,
)


def parser():
    root = argparse.ArgumentParser(description="Structured frozen NeuroLM V3 probe")
    commands = root.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract")
    extract.add_argument("--raw-pack-dir", required=True)
    extract.add_argument("--feature-pack-dir", required=True)
    extract.add_argument("--neurolm-repo", required=True)
    extract.add_argument("--checkpoint", required=True)
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--overwrite", action="store_true")

    smoke = commands.add_parser("smoke")
    smoke.add_argument("--feature-pack-dir", required=True)
    smoke.add_argument("--device", default="cuda")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--feature-pack-dir", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--device", default="cuda")
    return root


def main():
    args = parser().parse_args()
    if args.command == "extract":
        _, mapping = select_usable_mapping(build_mne_spatial_mapping())
        encoder = OfficialNeuroLMEncoder(
            args.neurolm_repo,
            args.checkpoint,
            mapping.neurolm_index.to_numpy(),
            zuco_indices=mapping.zuco_index.to_numpy(),
            device=args.device,
        )
        manifest = extract_structured_subject_packs(
            args.raw_pack_dir,
            args.feature_pack_dir,
            encoder,
            overwrite=args.overwrite,
        )
        print(json.dumps(manifest, indent=2))
        return

    records, _, report = load_structured_records(args.feature_pack_dir)
    config = StructuredProbeConfig(
        expected_channels=report["channels"],
        embedding_size=report["embedding_size"],
    )
    if args.command == "smoke":
        print(json.dumps(smoke_test_structured_probe(records, config, args.device), indent=2))
        return
    _, _, summary, _, gate = evaluate_structured_probe(
        records,
        args.output_dir,
        report["dataset_fingerprint"],
        config,
        args.device,
    )
    print(summary.to_string(index=False))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
