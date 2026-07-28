"""Command-line entry point for the TFM-to-ZuCo transfer test."""

import argparse
import json
from pathlib import Path

from src.config import EvaluationConfig, PreprocessConfig
from src.evaluation import bootstrap_alignment_delta, evaluate_histograms
from src.extraction import extract_token_cache
from src.features import build_sentence_histograms
from src.official_tfm import OfficialTFMTokenizer, discover_checkpoint
from src.zuco_io import inspect_zuco


def parser():
    root = argparse.ArgumentParser(description="Frozen TFM tokenizer transfer test on ZuCo.")
    commands = root.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect")
    inspect.add_argument("--raw-dir", required=True)
    inspect.add_argument("--labels-csv", required=True)

    extract = commands.add_parser("extract")
    extract.add_argument("--raw-dir", required=True)
    extract.add_argument("--labels-csv", required=True)
    extract.add_argument("--cache-dir", required=True)
    extract.add_argument("--tfm-repo", required=True)
    extract.add_argument("--checkpoint")
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--limit", type=int)
    extract.add_argument("--overwrite", action="store_true")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--cache-dir", required=True)
    evaluate.add_argument("--output-dir", required=True)
    return root


def main():
    args = parser().parse_args()
    if args.command == "inspect":
        print(json.dumps(inspect_zuco(args.raw_dir, args.labels_csv), indent=2))
        return
    if args.command == "extract":
        checkpoint = args.checkpoint or discover_checkpoint(args.tfm_repo)
        tokenizer = OfficialTFMTokenizer(args.tfm_repo, checkpoint, device=args.device)
        report = extract_token_cache(
            args.raw_dir,
            args.labels_csv,
            args.cache_dir,
            tokenizer,
            preprocess_config=PreprocessConfig(),
            overwrite=args.overwrite,
            limit=args.limit,
        )
        print(json.dumps(report, indent=2))
        return

    X, y, metadata, diagnostics = build_sentence_histograms(args.cache_dir)
    metrics, predictions, summary = evaluate_histograms(
        X,
        y,
        metadata["sentence_id"].to_numpy(),
        args.output_dir,
        EvaluationConfig(),
    )
    delta = bootstrap_alignment_delta(predictions)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with (Path(args.output_dir) / "alignment_delta.json").open("w") as handle:
        json.dump(delta, handle, indent=2)
    print(summary.to_string(index=False))
    print(json.dumps(delta, indent=2))


if __name__ == "__main__":
    main()

