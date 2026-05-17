#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set

TASK_KEY_RE = re.compile(r"system_[^/]+__frame_\d+")
POSCAR_KEY_RE = re.compile(r"POSCAR_(system_[^_]+(?:_[^_]+)*)_(frame_\d+)\.vasp$")


def _task_key_from_entry(entry: Dict[str, object]) -> str:
    tk = str(entry.get("stage1_task_name", "")).strip()
    if tk:
        return tk

    sp = str(entry.get("source_poscar_path_11", "")).strip()
    if sp:
        m = POSCAR_KEY_RE.search(sp)
        if m:
            return "{0}__{1}".format(m.group(1), m.group(2))
        m2 = re.search(r"(system_[^/]+)/(frame_\d+)", sp)
        if m2:
            return "{0}__{1}".format(m2.group(1), m2.group(2))

    sx = str(entry.get("source_xyz_path", "")).strip()
    if sx:
        m = re.search(r"(system_[^/]+)/(frame_\d+)", sx)
        if m:
            return "{0}__{1}".format(m.group(1), m.group(2))

    return ""


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", action="append", required=True, help="Stage-2 manifest JSON path; can pass multiple")
    parser.add_argument("--output", required=True, help="Output exclude-list text file")
    parser.add_argument(
        "--statuses",
        default="",
        help="Optional comma-separated status filter (e.g. finished,running). Empty means all statuses.",
    )


def run(args: argparse.Namespace) -> None:
    manifests = [Path(p) for p in args.manifest]
    statuses = {x.strip().lower() for x in str(args.statuses).split(",") if x.strip()}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_keys: Set[str] = set()
    rows: List[str] = []

    for mp in manifests:
        if not mp.is_file():
            raise ValueError("manifest not found: {0}".format(mp))
        js = json.loads(mp.read_text())
        entries = js.get("entries", []) if isinstance(js, dict) else []
        if not isinstance(entries, list):
            entries = []

        rows.append("# manifest: {0}".format(mp))
        rows.append("# entries: {0}".format(len(entries)))

        kept = 0
        for e in entries:
            if not isinstance(e, dict):
                continue
            st = str(e.get("status", "")).strip().lower()
            if statuses and st and st not in statuses:
                continue
            tk = _task_key_from_entry(e)
            if tk and TASK_KEY_RE.fullmatch(tk):
                all_keys.add(tk)
                kept += 1
        rows.append("# accepted_task_keys: {0}".format(kept))

    lines = rows + [""] + sorted(all_keys)
    out.write_text("\n".join(lines) + "\n")

    summary = {
        "manifests": [str(p) for p in manifests],
        "statuses_filter": sorted(statuses),
        "task_key_count": len(all_keys),
        "output": str(out),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build explicit exclude-list task keys from Stage-2 manifest(s)")
    add_args(parser)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
