"""DPA4C-compatible LMDB conversion, validation, and round-trip comparison."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..utils import write_json
from .manifest import (
    _read_type_metadata,
    _validate_set_arrays,
    inspect_group,
    manifest_sha256,
    read_manifest,
)


SCHEMA = "nnpgen.dpa4c-lmdb.v1"
LEGACY_SCHEMA = "legacy-dpa4c"
FRAME_FIELDS = ("atom_types", "coords", "cells", "energies", "forces")


def _dependencies() -> Tuple[Any, Any]:
    try:
        import lmdb  # type: ignore
        import msgpack  # type: ignore
    except ImportError as exc:
        raise RuntimeError("LMDB support requires `python -m pip install -e '.[lmdb]'`.") from exc
    return lmdb, msgpack


def _parse_type_map(value: str) -> List[str]:
    names = [item.strip() for item in str(value).split(",") if item.strip()]
    if not names:
        return []
    if len(names) != len(set(names)):
        raise ValueError("type-map names must be unique")
    return names


def _pack_array(value: np.ndarray) -> Dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {"type": array.dtype.str, "shape": list(array.shape), "data": array.tobytes()}


def _unpack_array(value: Mapping[str, Any]) -> np.ndarray:
    try:
        dtype = np.dtype(str(value["type"]))
        shape = tuple(int(item) for item in value["shape"])
        data = value["data"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid packed array") from exc
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(data) != expected:
        raise ValueError("packed array byte count does not match shape and dtype")
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def _frame_key(index: int) -> bytes:
    return f"{int(index):012d}".encode("ascii")


def _manifest_systems(manifest: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    rows = read_manifest(manifest)
    infos: List[Dict[str, Any]] = []
    for row in rows:
        group = Path(row["system_path"])
        info = inspect_group(group)
        if row.get("frames") and int(row["frames"]) != int(info["frames"]):
            raise ValueError(f"manifest frame count mismatch: {group}")
        if row.get("natoms") and int(row["natoms"]) != int(info["natoms"]):
            raise ValueError(f"manifest atom count mismatch: {group}")
        infos.append(info)
    return rows, infos


def _canonical_type_map(rows: Sequence[Mapping[str, str]], requested: str) -> List[str]:
    explicit = _parse_type_map(requested)
    if explicit:
        return explicit
    result: List[str] = []
    for row in rows:
        for name in _read_type_metadata(Path(row["system_path"]))["type_map"]:
            if name not in result:
                result.append(name)
    if not result:
        raise ValueError("could not infer a non-empty type map")
    return result


def _estimated_map_size(rows: Sequence[Mapping[str, str]]) -> int:
    total = 0
    for row in rows:
        group = Path(row["system_path"])
        for path in group.glob("set.*/*.npy"):
            if path.is_file():
                total += path.stat().st_size
    return max(64 * 1024 * 1024, total * 3 + 16 * 1024 * 1024)


def _atomic_staging_path(output: Path) -> Path:
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    if output.exists():
        raise FileExistsError(f"refusing existing LMDB output: {output}")
    if staging.exists():
        raise FileExistsError(f"refusing existing staging path: {staging}")
    return staging


def convert_npy_to_lmdb(
    manifest: Path,
    output: Path,
    report_path: Path,
    type_map: str = "",
    max_frames_per_system: int = 0,
    map_size: int = 0,
) -> Dict[str, Any]:
    """Convert a manifest of DPData groups into the DPA4C LMDB schema."""

    if int(max_frames_per_system) < 0:
        raise ValueError("max-frames-per-system must be non-negative")
    rows, infos = _manifest_systems(Path(manifest))
    global_type_map = _canonical_type_map(rows, type_map)
    global_index = {name: index for index, name in enumerate(global_type_map)}
    output = Path(output).expanduser().resolve()
    report_path = Path(report_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = _atomic_staging_path(output)
    staging.mkdir(parents=True)
    lmdb, msgpack = _dependencies()
    size = int(map_size) if int(map_size) > 0 else _estimated_map_size(rows)

    frame_nlocs: List[int] = []
    frame_system_ids: List[int] = []
    frame_set_ids: List[int] = []
    system_info: List[Dict[str, Any]] = []
    block_frames: Counter[str] = Counter()
    label_frames: Counter[str] = Counter()
    frame_index = 0
    environment = None
    transaction = None
    try:
        environment = lmdb.open(str(staging), map_size=size, subdir=True, meminit=False)
        transaction = environment.begin(write=True)
        for system_id, (row, info) in enumerate(zip(rows, infos)):
            group = Path(row["system_path"])
            local_info = _read_type_metadata(group)
            missing = [name for name in local_info["type_map"] if name not in global_index]
            if missing:
                raise ValueError(f"dataset elements missing from global type map at {group}: {missing}")
            remap = np.asarray([global_index[name] for name in local_info["type_map"]], dtype=np.int32)
            atom_types = remap[local_info["types"]].astype(np.int32, copy=False)
            packed_atom_types = _pack_array(atom_types)
            source_frames = int(info["frames"])
            remaining = int(max_frames_per_system) if int(max_frames_per_system) else None
            converted = 0
            set_id = 0
            for set_dir in sorted(path for path in group.glob("set.*") if path.is_dir()):
                _validate_set_arrays(set_dir, int(atom_types.size))
                arrays: Dict[str, Any] = {
                    name: np.load(str(set_dir / f"{name}.npy"), mmap_mode="r")
                    for name in ("coord", "box", "energy", "force")
                }
                virial_path = set_dir / "virial.npy"
                if virial_path.is_file():
                    arrays["virial"] = np.load(str(virial_path), mmap_mode="r")
                nframes = int(arrays["energy"].shape[0])
                if remaining is not None:
                    nframes = min(nframes, remaining)
                for offset in range(nframes):
                    frame: Dict[str, Any] = {
                        "atom_types": packed_atom_types,
                        "coords": _pack_array(np.asarray(arrays["coord"][offset]).reshape(int(atom_types.size), 3)),
                        "cells": _pack_array(np.asarray(arrays["box"][offset]).reshape(3, 3)),
                        "energies": _pack_array(np.asarray(arrays["energy"][offset]).reshape(1)),
                        "forces": _pack_array(np.asarray(arrays["force"][offset]).reshape(int(atom_types.size), 3)),
                    }
                    if "virial" in arrays:
                        frame["virials"] = _pack_array(np.asarray(arrays["virial"][offset]).reshape(9))
                    transaction.put(_frame_key(frame_index), msgpack.packb(frame, use_bin_type=True))
                    frame_nlocs.append(int(atom_types.size))
                    frame_system_ids.append(system_id)
                    frame_set_ids.append(set_id)
                    frame_index += 1
                    converted += 1
                    block_frames[str(row.get("block", "unknown"))] += 1
                    label = str(row.get("label", ""))
                    if label:
                        label_frames[label] += 1
                    if frame_index % 1024 == 0:
                        transaction.commit()
                        transaction = environment.begin(write=True)
                if remaining is not None:
                    remaining -= nframes
                    if remaining <= 0:
                        break
                set_id += 1
            if converted <= 0:
                raise ValueError(f"zero converted frames: {group}")
            global_type_counts = np.bincount(atom_types, minlength=len(global_type_map)).astype(int).tolist()
            system_info.append(
                {
                    "system_path": str(group),
                    "block": str(row.get("block", "unknown")),
                    "label": str(row.get("label", "")),
                    # DeepMD's LMDB reader uses the first system's ``natoms``
                    # list to initialize its type layout. Keep that field in
                    # the established per-type-count form and retain the
                    # total separately for human-readable provenance.
                    "natoms": global_type_counts,
                    "natoms_total": int(atom_types.size),
                    "type_map": list(local_info["type_map"]),
                    "type_counts": list(local_info["counts"]),
                    "frames_source": source_frames,
                    "frames_converted": converted,
                }
            )

        if transaction is not None:
            transaction.commit()
            transaction = None
        metadata = {
            "schema": SCHEMA,
            "nframes": frame_index,
            "frame_idx_fmt": "012d",
            "type_map": global_type_map,
            "system_info": system_info,
            "system_paths": [str(row["system_path"]) for row in rows],
            "system_blocks": [str(row.get("block", "unknown")) for row in rows],
            "system_labels": [str(row.get("label", "")) for row in rows],
            "frame_nlocs": frame_nlocs,
            "frame_system_ids": frame_system_ids,
            "frame_set_ids": frame_set_ids,
            "source_manifest": str(Path(manifest).expanduser().resolve()),
            "source_manifest_sha256": manifest_sha256(Path(manifest)),
        }
        with environment.begin(write=True) as transaction:
            transaction.put(b"__metadata__", msgpack.packb(metadata, use_bin_type=True))
        environment.sync()
        environment.close()
        environment = None
        os.replace(staging, output)
    except Exception as exc:
        if transaction is not None:
            transaction.abort()
        if environment is not None:
            environment.close()
        shutil.rmtree(staging, ignore_errors=True)
        if isinstance(exc, lmdb.MapFullError):
            raise ValueError(f"LMDB map size is too small ({size} bytes); pass a larger --map-size-gib") from exc
        raise

    report = {
        "status": "PASS",
        "schema": SCHEMA,
        "manifest": str(Path(manifest).expanduser().resolve()),
        "manifest_sha256": metadata["source_manifest_sha256"],
        "output": str(output),
        "systems": len(rows),
        "frames": frame_index,
        "blocks": dict(sorted(block_frames.items())),
        "labels": dict(sorted(label_frames.items())),
        "natoms_min": min(frame_nlocs),
        "natoms_max": max(frame_nlocs),
        "virial_frames": int(sum(item["virial_frames"] for info in infos for item in info["sets"])),
        "type_map": global_type_map,
        "map_size": size,
        "max_frames_per_system": int(max_frames_per_system),
    }
    write_json(report_path, report)
    return report


def _decode_frame(raw: bytes, msgpack: Any) -> Dict[str, np.ndarray]:
    decoded = msgpack.unpackb(raw, raw=False)
    if not isinstance(decoded, dict):
        raise ValueError("LMDB frame value is not a mapping")
    result: Dict[str, np.ndarray] = {}
    for name, value in decoded.items():
        result[str(name)] = _unpack_array(value)
    return result


def validate_lmdb(
    input_path: Path,
    full: bool = False,
    sample_frames: int = 3,
) -> Dict[str, Any]:
    """Validate metadata and representative/all DPA4C LMDB frame values."""

    if int(sample_frames) < 0:
        raise ValueError("sample-frames must be non-negative")
    lmdb, msgpack = _dependencies()
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.is_dir():
        raise ValueError(f"LMDB directory not found: {input_path}")
    errors: List[str] = []
    checked = 0
    metadata: Dict[str, Any] = {}
    environment = lmdb.open(str(input_path), readonly=True, lock=False, subdir=True)
    try:
        with environment.begin() as transaction:
            raw_metadata = transaction.get(b"__metadata__")
            if raw_metadata is None:
                errors.append("missing __metadata__")
            else:
                try:
                    metadata = msgpack.unpackb(raw_metadata, raw=False)
                except Exception as exc:
                    errors.append(f"invalid __metadata__: {exc}")
            if metadata:
                schema = metadata.get("schema")
                if schema not in (None, "", SCHEMA):
                    errors.append(f"unsupported schema: {schema}")
                nframes = int(metadata.get("nframes", -1))
                if nframes < 0:
                    errors.append("metadata nframes is invalid")
                    nframes = 0
                if environment.stat()["entries"] != nframes + 1:
                    errors.append("LMDB entry count does not match nframes plus metadata")
                nlocs = list(metadata.get("frame_nlocs", []))
                system_ids = list(metadata.get("frame_system_ids", []))
                if len(nlocs) != nframes or len(system_ids) != nframes:
                    errors.append("frame metadata arrays do not match nframes")
                type_map = list(metadata.get("type_map", []))
                if not type_map or len(type_map) != len(set(type_map)):
                    errors.append("metadata type_map is empty or duplicated")
                system_count = len(metadata.get("system_paths", []))
                if any(int(value) < 0 or int(value) >= system_count for value in system_ids):
                    errors.append("frame_system_ids contain an invalid system index")
                if full:
                    indices = range(nframes)
                else:
                    count = min(int(sample_frames), nframes)
                    indices = np.linspace(0, nframes - 1, count, dtype=int).tolist() if count else []
                for index in indices:
                    raw_frame = transaction.get(_frame_key(index))
                    if raw_frame is None:
                        errors.append(f"missing frame key {index}")
                        continue
                    try:
                        frame = _decode_frame(raw_frame, msgpack)
                        missing = [name for name in FRAME_FIELDS if name not in frame]
                        if missing:
                            errors.append(f"frame {index} missing fields: {missing}")
                            continue
                        natoms = int(nlocs[index])
                        expected_shapes = {
                            "atom_types": (natoms,),
                            "coords": (natoms, 3),
                            "cells": (3, 3),
                            "energies": (1,),
                            "forces": (natoms, 3),
                        }
                        for name, shape in expected_shapes.items():
                            if tuple(frame[name].shape) != shape:
                                errors.append(f"frame {index} {name} shape {frame[name].shape} != {shape}")
                            if not np.isfinite(frame[name]).all():
                                errors.append(f"frame {index} {name} contains NaN/Inf")
                        if frame["atom_types"].size and (
                            int(frame["atom_types"].min()) < 0
                            or int(frame["atom_types"].max()) >= len(type_map)
                        ):
                            errors.append(f"frame {index} atom_types contain an invalid type index")
                        if "virials" in frame and tuple(frame["virials"].shape) != (9,):
                            errors.append(f"frame {index} virials shape is invalid")
                        checked += 1
                    except (TypeError, ValueError, KeyError) as exc:
                        errors.append(f"frame {index} decode failed: {exc}")
    finally:
        environment.close()

    report = {
        "status": "PASS" if not errors else "FAIL",
        "valid": not errors,
        "schema": metadata.get("schema") or LEGACY_SCHEMA,
        "input": str(input_path),
        "systems": len(metadata.get("system_paths", [])),
        "frames": int(metadata.get("nframes", 0) or 0),
        "checked_frames": checked,
        "type_map": metadata.get("type_map", []),
        "errors": errors,
    }
    return report


def _max_abs_difference(expected: np.ndarray, actual: np.ndarray) -> float:
    if expected.size == 0:
        return 0.0
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def compare_npy_lmdb(
    manifest: Path,
    lmdb_path: Path,
    output: Optional[Path] = None,
    atol: float = 0.0,
    max_errors: int = 10,
) -> Dict[str, Any]:
    """Compare source NPY frames with the LMDB frame order and values."""

    if float(atol) < 0:
        raise ValueError("atol must be non-negative")
    if int(max_errors) < 1:
        raise ValueError("max-errors must be positive")
    rows, infos = _manifest_systems(Path(manifest))
    lmdb, msgpack = _dependencies()
    lmdb_path = Path(lmdb_path).expanduser().resolve()
    errors: List[str] = []
    compared = 0
    max_diffs: Counter[str] = Counter()
    environment = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=True)
    try:
        with environment.begin() as transaction:
            raw_metadata = transaction.get(b"__metadata__")
            if raw_metadata is None:
                raise ValueError("LMDB is missing __metadata__")
            metadata = msgpack.unpackb(raw_metadata, raw=False)
            expected_sha = manifest_sha256(Path(manifest))
            if metadata.get("source_manifest_sha256") != expected_sha:
                errors.append("source manifest SHA256 differs from LMDB metadata")
            global_type_map = list(metadata.get("type_map", []))
            global_index = {name: index for index, name in enumerate(global_type_map)}
            actual_frames = int(metadata.get("nframes", 0))
            frame_index = 0
            for system_id, (row, info) in enumerate(zip(rows, infos)):
                group = Path(row["system_path"])
                local_info = _read_type_metadata(group)
                remap = np.asarray([global_index[name] for name in local_info["type_map"]], dtype=np.int32)
                expected_types = remap[local_info["types"]].astype(np.int32, copy=False)
                for set_dir in sorted(path for path in group.glob("set.*") if path.is_dir()):
                    _validate_set_arrays(set_dir, int(expected_types.size))
                    arrays: Dict[str, Any] = {
                        name: np.load(str(set_dir / f"{name}.npy"), mmap_mode="r")
                        for name in ("coord", "box", "energy", "force")
                    }
                    virial_path = set_dir / "virial.npy"
                    if virial_path.is_file():
                        arrays["virial"] = np.load(str(virial_path), mmap_mode="r")
                    for offset in range(int(arrays["energy"].shape[0])):
                        if frame_index >= actual_frames:
                            errors.append("LMDB has fewer frames than the source manifest")
                            break
                        raw_frame = transaction.get(_frame_key(frame_index))
                        if raw_frame is None:
                            errors.append(f"missing LMDB frame {frame_index}")
                            frame_index += 1
                            continue
                        frame = _decode_frame(raw_frame, msgpack)
                        expected_values = {
                            "atom_types": expected_types,
                            "coords": np.asarray(arrays["coord"][offset]).reshape(int(expected_types.size), 3),
                            "cells": np.asarray(arrays["box"][offset]).reshape(3, 3),
                            "energies": np.asarray(arrays["energy"][offset]).reshape(1),
                            "forces": np.asarray(arrays["force"][offset]).reshape(int(expected_types.size), 3),
                        }
                        for name, expected in expected_values.items():
                            actual = frame.get(name)
                            if actual is None or actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0.0, atol=float(atol), equal_nan=False):
                                if len(errors) < int(max_errors):
                                    errors.append(f"frame {frame_index} {name} differs from NPY source")
                            max_diffs[name] = max(max_diffs[name], _max_abs_difference(expected, actual) if actual is not None and actual.shape == expected.shape else float("inf"))
                        if "virial" in arrays:
                            expected_virial = np.asarray(arrays["virial"][offset]).reshape(9)
                            actual_virial = frame.get("virials")
                            if actual_virial is None or actual_virial.shape != expected_virial.shape or not np.allclose(actual_virial, expected_virial, rtol=0.0, atol=float(atol), equal_nan=False):
                                if len(errors) < int(max_errors):
                                    errors.append(f"frame {frame_index} virials differs from NPY source")
                                max_diffs["virials"] = max(max_diffs["virials"], _max_abs_difference(expected_virial, actual_virial) if actual_virial is not None and actual_virial.shape == expected_virial.shape else float("inf"))
                        elif "virials" in frame:
                            if len(errors) < int(max_errors):
                                errors.append(f"frame {frame_index} contains unexpected virials")
                        compared += 1
                        frame_index += 1
            if frame_index != actual_frames:
                errors.append(f"LMDB frame count {actual_frames} != source frame count {frame_index}")
    finally:
        environment.close()

    report = {
        "status": "PASS" if not errors else "FAIL",
        "valid": not errors,
        "manifest": str(Path(manifest).expanduser().resolve()),
        "lmdb": str(lmdb_path),
        "systems": len(rows),
        "frames_compared": compared,
        "atol": float(atol),
        "max_abs_difference": dict(sorted(max_diffs.items())),
        "errors": errors[: int(max_errors)],
    }
    if output:
        write_json(Path(output), report)
    return report


def add_convert_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True, help="TSV manifest produced by dataset build-manifest.")
    parser.add_argument("--output", type=Path, required=True, help="New LMDB directory; existing targets are refused.")
    parser.add_argument("--report", type=Path, required=True, help="JSON conversion report.")
    parser.add_argument("--type-map", default="", help="Comma-separated canonical type map; omitted means first-seen union.")
    parser.add_argument("--max-frames-per-system", type=int, default=0, help="Keep only the first N frames per system; 0 means all.")
    parser.add_argument("--map-size-gib", type=float, default=0.0, help="LMDB map size in GiB; 0 means estimate from source NPY files.")


def run_convert(args: argparse.Namespace) -> Dict[str, Any]:
    if float(args.map_size_gib) < 0:
        raise ValueError("map-size-gib must be non-negative")
    size = int(float(args.map_size_gib) * 1024**3) if float(args.map_size_gib) else 0
    report = convert_npy_to_lmdb(args.manifest, args.output, args.report, args.type_map, args.max_frames_per_system, size)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def add_validate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", dest="input_path", type=Path, required=True, help="LMDB directory to validate.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON validation report.")
    parser.add_argument("--full", action="store_true", help="Decode every frame instead of representative frames.")
    parser.add_argument("--sample-frames", type=int, default=3, help="Representative frames to check when --full is absent.")


def run_validate(args: argparse.Namespace) -> Dict[str, Any]:
    report = validate_lmdb(args.input_path, args.full, args.sample_frames)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise ValueError("LMDB validation failed: " + "; ".join(report["errors"]))
    return report


def add_compare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True, help="Source TSV manifest.")
    parser.add_argument("--lmdb", dest="lmdb_path", type=Path, required=True, help="LMDB directory to compare.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON comparison report.")
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance for floating-point fields.")
    parser.add_argument("--max-errors", type=int, default=10, help="Maximum mismatch messages to retain.")


def run_compare(args: argparse.Namespace) -> Dict[str, Any]:
    report = compare_npy_lmdb(args.manifest, args.lmdb_path, args.output, args.atol, args.max_errors)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise ValueError("NPY/LMDB comparison failed: " + "; ".join(report["errors"]))
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert and validate DPA4C-compatible LMDB datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert_parser = subparsers.add_parser("convert")
    add_convert_arguments(convert_parser)
    validate_parser = subparsers.add_parser("validate")
    add_validate_arguments(validate_parser)
    compare_parser = subparsers.add_parser("compare")
    add_compare_arguments(compare_parser)
    args = parser.parse_args(argv)
    if args.command == "convert":
        run_convert(args)
    elif args.command == "validate":
        run_validate(args)
    else:
        run_compare(args)


if __name__ == "__main__":
    main()
