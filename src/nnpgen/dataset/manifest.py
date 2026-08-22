"""Discovery and validation helpers for Deep Potential NPY datasets."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


REQUIRED_ARRAYS = ("coord", "box", "energy", "force")
OPTIONAL_ARRAYS = ("virial",)
MANIFEST_FIELDS = (
    "index",
    "system_path",
    "block",
    "label",
    "frames",
    "natoms",
    "elements",
    "counts",
    "type_map",
    "type_map_sha256",
    "relative_path",
)


def _read_type_metadata(group: Path) -> Dict[str, Any]:
    type_path = group / "type.raw"
    type_map_path = group / "type_map.raw"
    if not type_path.is_file() or not type_map_path.is_file():
        raise ValueError(f"missing type.raw or type_map.raw: {group}")

    type_map = [line.strip() for line in type_map_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not type_map or len(type_map) != len(set(type_map)):
        raise ValueError(f"type_map.raw must contain unique non-empty names: {group}")
    types = np.loadtxt(str(type_path), dtype=np.int64, ndmin=1).reshape(-1)
    if types.size and (int(types.min()) < 0 or int(types.max()) >= len(type_map)):
        raise ValueError(f"type.raw index out of bounds: {group}")

    counts = np.bincount(types, minlength=len(type_map)).astype(int).tolist()
    present = [(name, count) for name, count in zip(type_map, counts) if count]
    return {
        "type_map": type_map,
        "types": types,
        "natoms": int(types.size),
        "elements": [name for name, _ in present],
        "counts": [count for _, count in present],
        "type_map_sha256": hashlib.sha256(type_map_path.read_bytes()).hexdigest(),
    }


def discover_groups(roots: Sequence[Path]) -> List[Path]:
    """Find DPData group roots below one or more explicit roots."""

    groups: Dict[str, Path] = {}
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"dataset root not found: {root}")
        candidates = [root] if (root / "type.raw").is_file() else [p.parent for p in root.rglob("type.raw")]
        for group in candidates:
            group = group.resolve()
            if not (group / "type_map.raw").is_file():
                continue
            if any((set_dir / "coord.npy").is_file() for set_dir in group.glob("set.*") if set_dir.is_dir()):
                groups[str(group)] = group
    return sorted(groups.values(), key=lambda path: str(path))


def _validate_set_arrays(set_dir: Path, natoms: int) -> Dict[str, Any]:
    arrays: Dict[str, Any] = {}
    for name in REQUIRED_ARRAYS:
        path = set_dir / f"{name}.npy"
        if not path.is_file():
            raise ValueError(f"missing {name}.npy: {set_dir}")
        arrays[name] = np.load(str(path), mmap_mode="r")

    frames = int(arrays["energy"].shape[0])
    if frames <= 0:
        raise ValueError(f"empty set: {set_dir}")
    if any(int(array.shape[0]) != frames for array in arrays.values()):
        raise ValueError(f"inconsistent frame count: {set_dir}")
    if int(np.prod(arrays["coord"].shape[1:])) != natoms * 3:
        raise ValueError(f"coord/type mismatch: {set_dir}")
    if arrays["force"].shape != arrays["coord"].shape:
        raise ValueError(f"force/coord mismatch: {set_dir}")
    if int(np.prod(arrays["box"].shape[1:])) != 9:
        raise ValueError(f"invalid box shape: {set_dir}")
    if int(np.prod(arrays["energy"].shape[1:])) != 1:
        raise ValueError(f"invalid energy shape: {set_dir}")

    virial_path = set_dir / "virial.npy"
    virial = None
    if virial_path.is_file():
        virial = np.load(str(virial_path), mmap_mode="r")
        if int(virial.shape[0]) != frames or int(np.prod(virial.shape[1:])) != 9:
            raise ValueError(f"invalid virial shape: {set_dir}")
        arrays["virial"] = virial

    nonfinite = {
        name: int(np.count_nonzero(~np.isfinite(array)))
        for name, array in arrays.items()
    }
    return {
        "path": str(set_dir.resolve()),
        "name": set_dir.name,
        "frames": frames,
        "shapes": {name: list(array.shape) for name, array in arrays.items()},
        "dtypes": {name: str(array.dtype) for name, array in arrays.items()},
        "bytes": int(sum((set_dir / f"{name}.npy").stat().st_size for name in arrays if (set_dir / f"{name}.npy").is_file())),
        "nonfinite": nonfinite,
        "virial_frames": frames if virial is not None else 0,
    }


def inspect_group(group: Path) -> Dict[str, Any]:
    """Return deterministic metadata and integrity information for one group."""

    group = Path(group).expanduser().resolve()
    type_info = _read_type_metadata(group)
    set_dirs = sorted(path for path in group.glob("set.*") if path.is_dir())
    if not set_dirs:
        raise ValueError(f"no set.* directories: {group}")

    sets = [_validate_set_arrays(set_dir, type_info["natoms"]) for set_dir in set_dirs]
    frames = int(sum(item["frames"] for item in sets))
    nonfinite = {
        name: int(sum(item["nonfinite"].get(name, 0) for item in sets))
        for name in set(REQUIRED_ARRAYS + OPTIONAL_ARRAYS)
    }
    return {
        "system_path": str(group),
        "relative_path": group.name,
        "frames": frames,
        "natoms": type_info["natoms"],
        "elements": type_info["elements"],
        "counts": type_info["counts"],
        "type_map": type_info["type_map"],
        "type_map_sha256": type_info["type_map_sha256"],
        "sets": sets,
        "set_count": len(sets),
        "virial_frames": int(sum(item["virial_frames"] for item in sets)),
        "bytes": int(sum(item["bytes"] for item in sets)),
        "nonfinite": nonfinite,
    }


def inspect_roots(roots: Sequence[Path]) -> Dict[str, Any]:
    groups: List[Dict[str, Any]] = []
    errors: List[str] = []
    for group in discover_groups(roots):
        try:
            groups.append(inspect_group(group))
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{group}: {exc}")

    return _summarize_groups(roots, groups, errors)


def _summarize_groups(roots: Sequence[Path], groups: Sequence[Mapping[str, Any]], errors: Sequence[str]) -> Dict[str, Any]:
    type_maps = sorted({tuple(str(value) for value in group.get("type_map", [])) for group in groups})
    return {
        "schema": "nnpgen.dpdata-inspection.v1",
        "roots": [str(Path(root).expanduser().resolve()) for root in roots],
        "valid": not errors and bool(groups),
        "errors": list(errors),
        "systems": len(groups),
        "frames": int(sum(int(group.get("frames", 0)) for group in groups)),
        "atoms": int(sum(int(group.get("frames", 0)) * int(group.get("natoms", 0)) for group in groups)),
        "virial_frames": int(sum(int(group.get("virial_frames", 0)) for group in groups)),
        "bytes": int(sum(int(group.get("bytes", 0)) for group in groups)),
        "type_maps": [list(type_map) for type_map in type_maps],
        "groups": [dict(group) for group in groups],
    }


def _default_block(root: Path) -> str:
    return root.name or "dataset"


def build_manifest_rows(
    roots: Sequence[Path],
    block: str = "",
    label: str = "",
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Build conversion rows and the associated inspection report."""

    report = inspect_roots(roots)
    if not report["valid"]:
        raise ValueError("dataset inspection failed: " + "; ".join(report["errors"]))
    rows: List[Dict[str, str]] = []
    for index, group in enumerate(report["groups"]):
        root = next(
            (Path(raw_root).expanduser().resolve() for raw_root in roots if Path(group["system_path"]).is_relative_to(Path(raw_root).expanduser().resolve())),
            Path(group["system_path"]).parent,
        )
        rows.append(
            {
                "index": str(index),
                "system_path": str(group["system_path"]),
                "block": block or _default_block(root),
                "label": label,
                "frames": str(group["frames"]),
                "natoms": str(group["natoms"]),
                "elements": json_list(group["elements"]),
                "counts": json_list(group["counts"]),
                "type_map": json_list(group["type_map"]),
                "type_map_sha256": str(group["type_map_sha256"]),
                "relative_path": str(group["relative_path"]),
            }
        )
    return rows, report


def json_list(values: Iterable[Any]) -> str:
    import json

    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def write_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_FIELDS), delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_manifest(path: Path) -> List[Dict[str, str]]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    try:
        indices = [int(row.get("index", "")) for row in rows]
    except ValueError as exc:
        raise ValueError(f"manifest index is not an integer: {path}") from exc
    if indices != list(range(len(rows))):
        raise ValueError("manifest indices must be contiguous from zero")

    seen = set()
    for row in rows:
        raw_system = str(row.get("system_path", "")).strip()
        if not raw_system:
            raise ValueError("manifest row is missing system_path")
        system = Path(raw_system)
        if not system.is_absolute():
            system = path.parent / system
        system = system.expanduser().resolve()
        row["system_path"] = str(system)
        if str(system) in seen:
            raise ValueError(f"manifest contains duplicate system_path: {system}")
        seen.add(str(system))
        row.setdefault("block", "unknown")
        row.setdefault("label", "")
    return rows


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def parse_json_list(value: str, field: str) -> List[Any]:
    import json

    if not str(value).strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {field} JSON in manifest: {value}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"manifest {field} must be a JSON list")
    return parsed
