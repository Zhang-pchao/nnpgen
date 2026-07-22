"""Portable status summaries for manifests, frame runs, and XYZ datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..utils import write_json


def count_statuses(records: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    counts = Counter(str(record.get("status", "unknown") or "unknown").lower() for record in records)
    return dict(sorted(counts.items()))


def _records_from_manifest(data: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    records = data.get("entries")
    if not isinstance(records, list):
        records = data.get("records")
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, Mapping)]


def summarize_manifest(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = _records_from_manifest(data)
    counts = count_statuses(records)
    terminal = sum(counts.get(status, 0) for status in ("finished", "failed", "completed"))
    return {
        "path": str(path),
        "kind": data.get("kind", ""),
        "schema_version": data.get("schema_version", ""),
        "total": len(records),
        "terminal": terminal,
        "unfinished": max(len(records) - terminal, 0),
        "status_counts": counts,
    }


def _status_from_json(data: Mapping[str, Any]) -> str:
    value = data.get("status")
    if value is not None and str(value).strip():
        return str(value).strip().lower()
    counts = data.get("status_counts")
    if isinstance(counts, Mapping):
        if counts.get("failed"):
            return "failed"
        if counts.get("finished") and not counts.get("running") and not counts.get("submitted"):
            return "finished"
    return "unknown"


def summarize_run_root(root: Path) -> Dict[str, Any]:
    root = root.expanduser().resolve()
    candidates: List[Path] = []
    for name in ("frame_info.json", "task_info.json", "generation_summary.json"):
        candidates.extend(root.rglob(name))
    unique = sorted(set(candidates))
    statuses = Counter()
    unreadable = []
    for path in unique:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            statuses[_status_from_json(data)] += 1
        except (OSError, ValueError, TypeError):
            unreadable.append(str(path))
    return {
        "path": str(root),
        "files_scanned": len(unique),
        "status_counts": dict(sorted(statuses.items())),
        "unreadable_files": unreadable,
    }


def count_xyz_frames(path: Path) -> int:
    """Count complete XYZ frames without loading coordinates into memory."""

    frames = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            while True:
                first = handle.readline()
                if not first:
                    break
                if not first.strip():
                    continue
                try:
                    atom_count = int(first.strip())
                except ValueError:
                    break
                if not handle.readline():
                    break
                complete = True
                for _ in range(atom_count):
                    if not handle.readline():
                        complete = False
                        break
                if not complete:
                    break
                frames += 1
    except OSError:
        return 0
    return frames


def summarize_xyz(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "frames": count_xyz_frames(path)}


def build_summary(
    manifests: Sequence[Path],
    run_roots: Sequence[Path],
    xyz_files: Sequence[Path],
) -> Dict[str, Any]:
    return {
        "schema_version": "nnpgen.monitor_summary.v1",
        "manifests": [summarize_manifest(path.expanduser().resolve()) for path in manifests],
        "run_roots": [summarize_run_root(path) for path in run_roots],
        "xyz": [summarize_xyz(path.expanduser().resolve()) for path in xyz_files],
    }


def add_summary_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", action="append", type=Path, default=[], help="Manifest JSON path; repeatable.")
    parser.add_argument("--run-root", action="append", type=Path, default=[], help="Run root to scan for status JSON; repeatable.")
    parser.add_argument("--xyz", dest="xyz_files", action="append", type=Path, default=[], help="XYZ file to count; repeatable.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")


def run_summary(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.manifest and not args.run_root and not args.xyz_files:
        raise ValueError("Provide at least one --manifest, --run-root, or --xyz")
    summary = build_summary(args.manifest, args.run_root, args.xyz_files)
    if args.output:
        output = args.output.expanduser().resolve()
        write_json(output, summary)
        summary["output"] = str(output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize workflow status and XYZ frame coverage.")
    add_summary_arguments(parser)
    run_summary(parser.parse_args(argv))


if __name__ == "__main__":
    main()
