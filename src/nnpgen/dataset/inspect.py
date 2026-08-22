"""CLI for inspecting and indexing DPData/NPY dataset roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..utils import write_json
from .manifest import build_manifest_rows, inspect_roots, write_manifest


def add_inspect_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", action="append", required=True, help="Dataset root or DPData group; repeat for multiple roots.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON inspection report.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional TSV manifest generated from valid groups.")
    parser.add_argument("--block", default="", help="Block name written to the optional manifest.")
    parser.add_argument("--label", default="", help="Provenance label written to the optional manifest.")


def run_inspect(args: argparse.Namespace) -> Dict[str, Any]:
    roots = [Path(value) for value in args.root]
    report = inspect_roots(roots)
    if args.manifest and report["valid"]:
        rows, _ = build_manifest_rows(roots, block=str(args.block), label=str(args.label))
        write_manifest(args.manifest, rows)
        report["manifest"] = str(Path(args.manifest).expanduser().resolve())
        report["manifest_rows"] = len(rows)
    if args.output:
        write_json(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise ValueError("dataset inspection failed: " + "; ".join(report["errors"]))
    return report


def add_manifest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", action="append", required=True, help="Dataset root or DPData group; repeat for multiple roots.")
    parser.add_argument("--output", type=Path, required=True, help="Output TSV manifest.")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON inspection report.")
    parser.add_argument("--block", default="", help="Block name; defaults to each root directory name.")
    parser.add_argument("--label", default="", help="Provenance label written to every row.")


def run_build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    roots = [Path(value) for value in args.root]
    rows, report = build_manifest_rows(roots, block=str(args.block), label=str(args.label))
    write_manifest(args.output, rows)
    report["manifest"] = str(Path(args.output).expanduser().resolve())
    report["manifest_rows"] = len(rows)
    if args.report:
        write_json(Path(args.report), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect and index Deep Potential NPY datasets.")
    add_inspect_arguments(parser)
    run_inspect(parser.parse_args(argv))


if __name__ == "__main__":
    main()
