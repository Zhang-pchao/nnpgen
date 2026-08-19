"""Integrity checks for Deep Potential RAW/NPY datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..utils import write_json


ARRAY_FILES = ("energy", "box", "coord", "force", "virial")


def composition_key(group: Mapping[str, Any]) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    """Return a stable identity independent of generated group directory names."""

    return tuple(str(value) for value in group.get("elements", [])), tuple(int(value) for value in group.get("counts", []))


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def validate_dp_dataset(root: Path) -> Dict[str, Any]:
    """Validate converter summaries, RAW files, NPY arrays, and type metadata."""

    import numpy as np

    root = Path(root).expanduser().resolve()
    errors: List[str] = []
    summary_path = root / "convert_summary.json"
    if not summary_path.is_file():
        return {"valid": False, "root": str(root), "errors": ["missing convert_summary.json"], "groups": []}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"valid": False, "root": str(root), "errors": ["invalid convert_summary.json: {0}".format(exc)], "groups": []}

    groups = summary.get("groups")
    if not isinstance(groups, list) or not groups:
        groups = [summary] if summary.get("elements") else []
    if int(summary.get("frames_skipped", 0) or 0):
        errors.append("converter skipped OUTCAR frames")
    if int(summary.get("frames_skipped_invalid_poscar", 0) or 0):
        errors.append("converter skipped invalid POSCAR frames")

    group_reports: List[Dict[str, Any]] = []
    total_frames = 0
    seen_compositions = set()
    multi_group = len(groups) > 1
    for index, group in enumerate(groups, start=1):
        key = composition_key(group)
        if not key[0] or len(key[0]) != len(key[1]):
            errors.append("group {0} has invalid composition".format(index))
        if key in seen_compositions:
            errors.append("duplicate composition: {0}".format(key))
        seen_compositions.add(key)
        group_dir = root / Path(str(group.get("output_dir", ""))).name if multi_group else root
        expected = int(group.get("frames_converted", 0) or 0)
        selected = int(group.get("frames_selected", expected) or 0)
        total_frames += expected
        group_errors: List[str] = []
        if expected <= 0 or selected != expected or int(group.get("frames_skipped", 0) or 0):
            group_errors.append("selected/converted frame counts do not match")
        for name in ARRAY_FILES:
            raw_path = group_dir / (name + ".raw")
            npy_path = group_dir / "set.000" / (name + ".npy")
            if not raw_path.is_file() or not npy_path.is_file():
                group_errors.append("missing {0}.raw or set.000/{0}.npy".format(name))
                continue
            try:
                if _line_count(raw_path) != expected:
                    group_errors.append("{0}.raw frame count mismatch".format(name))
                if int(np.load(str(npy_path), mmap_mode="r").shape[0]) != expected:
                    group_errors.append("{0}.npy frame count mismatch".format(name))
            except (OSError, ValueError) as exc:
                group_errors.append("cannot read {0}: {1}".format(name, exc))
        type_path = group_dir / "type.raw"
        type_map_path = group_dir / "type_map.raw"
        atom_count = sum(key[1])
        if not type_path.is_file() or not type_map_path.is_file():
            group_errors.append("missing type.raw or type_map.raw")
        else:
            try:
                types = [int(line) for line in type_path.read_text(encoding="utf-8").split()]
                type_map = type_map_path.read_text(encoding="utf-8").split()
                if len(types) != atom_count or not type_map or any(value < 0 or value >= len(type_map) for value in types):
                    group_errors.append("type metadata mismatch")
            except (OSError, ValueError) as exc:
                group_errors.append("cannot read type metadata: {0}".format(exc))
        errors.extend("group {0}: {1}".format(index, message) for message in group_errors)
        group_reports.append(
            {
                "composition": [list(key[0]), list(key[1])],
                "directory": str(group_dir),
                "frames": expected,
                "valid": not group_errors,
            }
        )

    if int(summary.get("frames_converted", total_frames) or 0) != total_frames:
        errors.append("top-level frame count mismatch")
    if int(summary.get("group_count", len(groups)) or 0) != len(groups):
        errors.append("top-level group count mismatch")
    return {"valid": not errors, "root": str(root), "errors": errors, "frames": total_frames, "groups": group_reports}


def require_valid_dp_dataset(root: Path) -> Dict[str, Any]:
    report = validate_dp_dataset(root)
    if not report["valid"]:
        raise ValueError("DP dataset validation failed: {0}".format("; ".join(report["errors"])))
    return report


def add_validation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True, help="DP dataset root to validate.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON validation report.")


def run_validation(args: argparse.Namespace) -> Dict[str, Any]:
    report = validate_dp_dataset(args.dataset_root)
    if args.output:
        write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise ValueError("DP dataset validation failed")
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Validate RAW, NPY, set.000, and type metadata in DP data.")
    add_validation_arguments(parser)
    run_validation(parser.parse_args(argv))


if __name__ == "__main__":
    main()
