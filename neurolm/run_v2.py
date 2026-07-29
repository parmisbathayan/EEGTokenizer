"""Command-line helper for V2; the dedicated Colab notebook is the main interface."""

import argparse
import json

from src.config import PreprocessConfig
from src.raw_cache import build_raw_subject_packs, load_raw_records
from src.raw_eegnet import RawEEGNetConfig, evaluate_raw_eegnet, smoke_test_raw_eegnet


def parser():
    root = argparse.ArgumentParser(description="Raw EEGNet V2 screen on ZuCo")
    commands = root.add_subparsers(dest="command", required=True)

    cache = commands.add_parser("cache")
    cache.add_argument("--raw-dir", required=True)
    cache.add_argument("--labels-csv", required=True)
    cache.add_argument("--pack-dir", required=True)
    cache.add_argument("--overwrite", action="store_true")

    smoke = commands.add_parser("smoke")
    smoke.add_argument("--pack-dir", required=True)
    smoke.add_argument("--device", default="cuda")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--pack-dir", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--device", default="cuda")
    return root


def main():
    args = parser().parse_args()
    preprocess = PreprocessConfig()
    if args.command == "cache":
        result = build_raw_subject_packs(
            args.raw_dir,
            args.labels_csv,
            args.pack_dir,
            preprocess,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, indent=2))
        return

    records, _, report = load_raw_records(args.pack_dir, preprocess)
    if args.command == "smoke":
        print(
            json.dumps(
                smoke_test_raw_eegnet(records, RawEEGNetConfig(), args.device),
                indent=2,
            )
        )
        return

    _, _, summary, _, gate = evaluate_raw_eegnet(
        records,
        args.output_dir,
        report["dataset_fingerprint"],
        RawEEGNetConfig(),
        args.device,
    )
    print(summary.to_string(index=False))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
