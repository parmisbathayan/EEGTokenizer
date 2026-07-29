"""Command-line helper for V4; the dedicated Colab notebook is primary."""

import argparse
import json

from src.channels import build_mne_spatial_mapping, select_usable_mapping
from src.gpt2_cache import (
    OfficialNeuroLMGPT2,
    extract_gpt2_subject_packs,
    load_gpt2_records,
)
from src.gpt2_verbalizer import (
    GPT2VerbalizerConfig,
    evaluate_gpt2_verbalizer,
    smoke_test_gpt2_verbalizer,
)


def parser():
    root = argparse.ArgumentParser(description="Frozen NeuroLM/GPT-2 V4 verbalizer")
    commands = root.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract")
    extract.add_argument("--raw-pack-dir", required=True)
    extract.add_argument("--feature-pack-dir", required=True)
    extract.add_argument("--neurolm-repo", required=True)
    extract.add_argument("--checkpoint", required=True)
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--batch-size", type=int, default=4)
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
        encoder = OfficialNeuroLMGPT2(
            args.neurolm_repo,
            args.checkpoint,
            mapping.neurolm_index.to_numpy(),
            mapping.zuco_index.to_numpy(),
            device=args.device,
        )
        report = extract_gpt2_subject_packs(
            args.raw_pack_dir,
            args.feature_pack_dir,
            encoder,
            overwrite=args.overwrite,
            batch_size=args.batch_size,
        )
        print(json.dumps(report, indent=2))
        return

    records, vectors, _, report = load_gpt2_records(args.feature_pack_dir)
    config = GPT2VerbalizerConfig(embedding_size=report["embedding_size"])
    if args.command == "smoke":
        print(
            json.dumps(
                smoke_test_gpt2_verbalizer(records, vectors, config, args.device),
                indent=2,
            )
        )
        return
    _, _, summary, _, gate = evaluate_gpt2_verbalizer(
        records,
        vectors,
        args.output_dir,
        report["dataset_fingerprint"],
        config,
        args.device,
    )
    print(summary.to_string(index=False))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
