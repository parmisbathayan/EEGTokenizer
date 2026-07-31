#!/usr/bin/env python3
"""Command-line interface for EEGText corpus preparation."""

import argparse
import json

from src.config import AuditConfig
from src.manifest import audit_zuco, combine_manifests
from src.osf_inventory import inventory_osf


def _inventory_command(args):
    return inventory_osf(args.node, args.output_dir)


def _audit_command(args):
    config = AuditConfig(
        dataset=args.dataset,
        release=args.release,
        task=args.task,
        pattern=args.pattern,
        source_hz=args.source_hz,
        expected_channels=args.expected_channels,
        minimum_samples=args.minimum_samples,
        maximum_nonfinite_fraction=args.maximum_nonfinite_fraction,
        recursive=not args.no_recursive,
        labels_csv=args.labels_csv,
    )
    return audit_zuco(args.raw_dir, config, args.output_dir)


def _combine_command(args):
    return combine_manifests(args.manifest, args.output_dir)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser(
        "inventory-osf", description="List public OSF files without downloading them."
    )
    inventory.add_argument("--node", required=True, help="Public OSF node identifier.")
    inventory.add_argument("--output-dir", required=True)
    inventory.set_defaults(handler=_inventory_command)

    audit = commands.add_parser(
        "audit-zuco", description="Audit one directory of ZuCo MATLAB files."
    )
    audit.add_argument("--raw-dir", required=True)
    audit.add_argument("--dataset", default="zuco")
    audit.add_argument("--release", default="1.0")
    audit.add_argument("--task", required=True)
    audit.add_argument("--pattern", required=True)
    audit.add_argument("--output-dir", required=True)
    audit.add_argument("--labels-csv")
    audit.add_argument("--source-hz", type=float, default=500.0)
    audit.add_argument("--expected-channels", type=int, default=105)
    audit.add_argument("--minimum-samples", type=int, default=500)
    audit.add_argument("--maximum-nonfinite-fraction", type=float, default=0.20)
    audit.add_argument("--no-recursive", action="store_true")
    audit.set_defaults(handler=_audit_command)

    combine = commands.add_parser(
        "combine-manifests",
        description="Merge task manifests and recompute cross-task duplicate groups.",
    )
    combine.add_argument("--manifest", action="append", required=True)
    combine.add_argument("--output-dir", required=True)
    combine.set_defaults(handler=_combine_command)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
