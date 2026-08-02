"""Command-line entry point for the frozen GPT-2 text-only reference."""

import argparse
import json

from src.text_gpt2 import TextGPT2Config, run_text_reference


def parser():
    root = argparse.ArgumentParser(description="Frozen GPT-2 text-only ZuCo reference")
    root.add_argument("--labels-csv", required=True)
    root.add_argument("--feature-cache", required=True)
    root.add_argument("--model-cache-dir", required=True)
    root.add_argument("--output-dir", required=True)
    root.add_argument("--device", default="cuda")
    root.add_argument("--batch-size", type=int, default=32)
    root.add_argument("--max-length", type=int, default=128)
    root.add_argument("--force-extract", action="store_true")
    return root


def main():
    args = parser().parse_args()
    config = TextGPT2Config(batch_size=args.batch_size, max_length=args.max_length)
    _, _, summary, delta = run_text_reference(
        labels_csv=args.labels_csv,
        feature_cache=args.feature_cache,
        model_cache_dir=args.model_cache_dir,
        output_dir=args.output_dir,
        device=args.device,
        config=config,
        force_extract=args.force_extract,
    )
    print(summary.to_string(index=False))
    print(json.dumps(delta, indent=2))


if __name__ == "__main__":
    main()
