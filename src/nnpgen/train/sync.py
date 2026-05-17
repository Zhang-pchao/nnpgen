"""Configurable dataset sync helper for fine-tuning workflows."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import List


def _run(cmd: List[str], dry_run: bool) -> None:
    print(" ".join(shlex.quote(part) for part in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def _remote_target(host: str, path: str) -> str:
    return f"{host}:{path}" if host else path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync DP train/test data and optional fine-tuning assets")
    parser.add_argument("--train-source", required=True, help="Local train dataset root")
    parser.add_argument("--test-source", required=True, help="Local test dataset root")
    parser.add_argument("--target-root", required=True, help="Target root on local or remote machine")
    parser.add_argument("--target-host", default="", help="Optional SSH host for remote rsync target")
    parser.add_argument("--model", default="", help="Optional model checkpoint to sync")
    parser.add_argument("--input-json", default="", help="Optional fine-tuning input JSON to sync")
    parser.add_argument("--delete", action="store_true", help="Delete stale files in train/test target directories")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    target_root = str(args.target_root).rstrip("/")
    common = ["rsync", "-az"]
    if args.delete:
        common.append("--delete")

    transfers = [
        (args.train_source.rstrip("/") + "/", f"{target_root}/train_data/"),
        (args.test_source.rstrip("/") + "/", f"{target_root}/test_data/"),
    ]
    if args.model:
        transfers.append((args.model, f"{target_root}/model/"))
    if args.input_json:
        transfers.append((args.input_json, f"{target_root}/finetune/"))

    for source, target in transfers:
        _run(common + [source, _remote_target(args.target_host, target)], dry_run=args.dry_run)

    manifest = {
        "schema_version": "nnpgen.sync.v1",
        "target_host": args.target_host,
        "target_root": target_root,
        "train_target": f"{target_root}/train_data",
        "test_target": f"{target_root}/test_data",
        "model_synced": bool(args.model),
        "input_json_synced": bool(args.input_json),
    }
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    parser = build_parser()
    run(parser.parse_args())


if __name__ == "__main__":
    main()
