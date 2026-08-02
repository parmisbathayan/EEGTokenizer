"""Command-line helper for V5; the dedicated Colab notebook is primary."""

import argparse
import hashlib
import json

from src.channels import build_mne_spatial_mapping, select_usable_mapping
from src.config import PreprocessConfig
from src.gpt2_cache import OfficialNeuroLMGPT2
from src.partial_finetune import (
    PartiallyUnfrozenNeuroLM,
    PartialFinetuneConfig,
    evaluate_partial_finetune,
    smoke_test_partial_finetune,
)
from src.raw_cache import load_raw_records


def parser():
    root = argparse.ArgumentParser(description="Partially unfrozen NeuroLM V5")
    root.add_argument("--raw-pack-dir", required=True)
    root.add_argument("--neurolm-repo", required=True)
    root.add_argument("--checkpoint", required=True)
    root.add_argument("--output-dir", required=True)
    root.add_argument("--device", default="cuda")
    root.add_argument("--smoke-only", action="store_true")
    return root


def file_sha256(path, chunk_size=2**20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parser().parse_args()
    config = PartialFinetuneConfig()
    records, _, report = load_raw_records(args.raw_pack_dir, PreprocessConfig())
    _, mapping = select_usable_mapping(build_mne_spatial_mapping())
    base = OfficialNeuroLMGPT2(
        args.neurolm_repo,
        args.checkpoint,
        mapping.neurolm_index.to_numpy(),
        mapping.zuco_index.to_numpy(),
        device=args.device,
        maximum_seconds=config.maximum_seconds,
    )
    model = PartiallyUnfrozenNeuroLM(base, config)
    smoke = smoke_test_partial_finetune(model, records, config)
    print(json.dumps(smoke, indent=2))
    if args.smoke_only:
        return
    _, _, summary, _, gate = evaluate_partial_finetune(
        model=model,
        records=records,
        output_dir=args.output_dir,
        dataset_fingerprint=report["dataset_fingerprint"],
        checkpoint_fingerprint=file_sha256(args.checkpoint),
        config=config,
    )
    print(summary.to_string(index=False))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
