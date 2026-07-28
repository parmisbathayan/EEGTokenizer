"""Command-line entry point; Colab remains the intended runtime."""

import argparse
import json
from pathlib import Path

from src.channels import build_mne_spatial_mapping
from src.config import EvaluationConfig, PreprocessConfig
from src.evaluation import (
    bootstrap_alignment_delta,
    evaluate_features,
    viability_gate,
)
from src.extraction import extract_feature_cache
from src.features import build_sentence_features
from src.official_neurolm import OfficialNeuroLMEncoder
from src.zuco_io import inspect_zuco


def parser():
    root = argparse.ArgumentParser(description="Frozen NeuroLM-B transfer test on ZuCo")
    commands = root.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--raw-dir", required=True)
    inspect.add_argument("--labels-csv", required=True)

    extract = commands.add_parser("extract")
    extract.add_argument("--raw-dir", required=True)
    extract.add_argument("--labels-csv", required=True)
    extract.add_argument("--cache-dir", required=True)
    extract.add_argument("--neurolm-repo", required=True)
    extract.add_argument("--checkpoint", required=True)
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--limit", type=int)

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
        mapping = build_mne_spatial_mapping()
        encoder = OfficialNeuroLMEncoder(
            args.neurolm_repo,
            args.checkpoint,
            mapping["neurolm_index"].to_numpy(),
            device=args.device,
        )
        manifest = extract_feature_cache(
            args.raw_dir,
            args.labels_csv,
            args.cache_dir,
            encoder,
            PreprocessConfig(),
            limit=args.limit,
        )
        print(json.dumps(manifest, indent=2))
        return

    X, y, metadata, diagnostics = build_sentence_features(args.cache_dir)
    config = EvaluationConfig()
    metrics, predictions, summary = evaluate_features(
        X, y, metadata["sentence_id"].to_numpy(), args.output_dir, config
    )
    delta = bootstrap_alignment_delta(
        predictions,
        samples=config.bootstrap_samples,
        confidence=config.bootstrap_ci,
    )
    gate = viability_gate(metrics, delta, config)
    output = Path(args.output_dir)
    (output / "alignment_delta.json").write_text(json.dumps(delta, indent=2))
    (output / "viability_gate.json").write_text(json.dumps(gate, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps({k: v for k, v in diagnostics.items() if k != "records"}, indent=2))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
