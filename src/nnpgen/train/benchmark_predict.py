#!/usr/bin/env python3
"""Detailed, streaming DeepMD evaluation for DPData/NPY systems."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..dataset.manifest import discover_groups, manifest_sha256, read_manifest
from ..utils import write_json


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SystemSpec:
    index: int
    path: Path
    block: str
    label: str


@dataclass
class ErrorAccumulator:
    systems: int = 0
    frames: int = 0
    energy_abs: float = 0.0
    energy_sq: float = 0.0
    energy_per_atom_abs: float = 0.0
    energy_per_atom_sq: float = 0.0
    force_abs: float = 0.0
    force_sq: float = 0.0
    force_count: int = 0
    virial_abs: float = 0.0
    virial_sq: float = 0.0
    virial_count: int = 0

    def update(
        self,
        energy_true: np.ndarray,
        energy_pred: np.ndarray,
        natoms: int,
        force_true: np.ndarray,
        force_pred: np.ndarray,
        virial_true: Optional[np.ndarray] = None,
        virial_pred: Optional[np.ndarray] = None,
    ) -> None:
        energy_error = np.asarray(energy_pred, dtype=float).reshape(-1) - np.asarray(energy_true, dtype=float).reshape(-1)
        force_error = np.asarray(force_pred, dtype=float).reshape(-1) - np.asarray(force_true, dtype=float).reshape(-1)
        energy_per_atom = energy_error / int(natoms)
        self.frames += int(energy_error.size)
        self.energy_abs += float(np.abs(energy_error).sum())
        self.energy_sq += float(np.square(energy_error).sum())
        self.energy_per_atom_abs += float(np.abs(energy_per_atom).sum())
        self.energy_per_atom_sq += float(np.square(energy_per_atom).sum())
        self.force_abs += float(np.abs(force_error).sum())
        self.force_sq += float(np.square(force_error).sum())
        self.force_count += int(force_error.size)
        if virial_true is not None and virial_pred is not None:
            virial_error = np.asarray(virial_pred, dtype=float).reshape(-1) - np.asarray(virial_true, dtype=float).reshape(-1)
            self.virial_abs += float(np.abs(virial_error).sum())
            self.virial_sq += float(np.square(virial_error).sum())
            self.virial_count += int(virial_error.size)

    def merge(self, other: "ErrorAccumulator") -> None:
        for field in (
            "systems",
            "frames",
            "energy_abs",
            "energy_sq",
            "energy_per_atom_abs",
            "energy_per_atom_sq",
            "force_abs",
            "force_sq",
            "force_count",
            "virial_abs",
            "virial_sq",
            "virial_count",
        ):
            setattr(self, field, getattr(self, field) + getattr(other, field))

    def metrics(self) -> Dict[str, float]:
        if self.frames <= 0 or self.force_count <= 0:
            raise ValueError("cannot calculate metrics from an empty accumulator")
        result = {
            "energy_mae_eV": self.energy_abs / self.frames,
            "energy_rmse_eV": math.sqrt(self.energy_sq / self.frames),
            "energy_mae_meV_per_atom": 1000.0 * self.energy_per_atom_abs / self.frames,
            "energy_rmse_meV_per_atom": 1000.0 * math.sqrt(self.energy_per_atom_sq / self.frames),
            "force_mae_eV_per_A": self.force_abs / self.force_count,
            "force_rmse_eV_per_A": math.sqrt(self.force_sq / self.force_count),
            "force_mae_meV_per_A": 1000.0 * self.force_abs / self.force_count,
            "force_rmse_meV_per_A": 1000.0 * math.sqrt(self.force_sq / self.force_count),
        }
        if self.virial_count:
            result.update(
                {
                    "virial_mae_eV": self.virial_abs / self.virial_count,
                    "virial_rmse_eV": math.sqrt(self.virial_sq / self.virial_count),
                }
            )
        return result


def frames_per_chunk(natoms: int, chunk_atoms: int, batch_size: int = 0) -> int:
    """Return a positive frame batch bounded by atom count and optional frame cap."""

    if int(natoms) <= 0:
        raise ValueError("natoms must be positive")
    if int(chunk_atoms) <= 0 and int(batch_size) <= 0:
        raise ValueError("chunk-atoms or batch-size must be positive")
    frames = max(1, int(chunk_atoms) // int(natoms)) if int(chunk_atoms) > 0 else int(batch_size)
    if int(batch_size) > 0:
        frames = min(frames, int(batch_size))
    return max(1, frames)


def _load_system_specs(dataset_root: Optional[Path], manifest: Optional[Path]) -> List[SystemSpec]:
    if manifest is not None:
        rows = read_manifest(manifest)
        return [
            SystemSpec(
                index=int(row["index"]),
                path=Path(row["system_path"]),
                block=str(row.get("block", "") or "unknown"),
                label=str(row.get("label", "")),
            )
            for row in rows
        ]
    if dataset_root is None:
        raise ValueError("dataset-root or manifest is required")
    root = dataset_root.expanduser().resolve()
    leaves = discover_groups([root])
    specs: List[SystemSpec] = []
    for index, leaf in enumerate(leaves):
        relative = leaf.relative_to(root)
        block = relative.parts[0] if relative.parts else root.name
        specs.append(SystemSpec(index=index, path=leaf, block=block or "unknown", label=""))
    return specs


def _load_type_info(dataset_dir: Path, model_type_map: Sequence[str]) -> Tuple[int, np.ndarray, List[str]]:
    type_map = [line.strip() for line in (dataset_dir / "type_map.raw").read_text(encoding="utf-8").splitlines() if line.strip()]
    raw = np.loadtxt(str(dataset_dir / "type.raw"), dtype=int, ndmin=1).reshape(-1)
    if not type_map or len(type_map) != len(set(type_map)):
        raise ValueError(f"invalid type_map.raw: {dataset_dir}")
    if raw.size == 0 or int(raw.min()) < 0 or int(raw.max()) >= len(type_map):
        raise ValueError(f"type.raw index out of range: {dataset_dir}")
    model_index = {name: index for index, name in enumerate(model_type_map)}
    species = [type_map[int(index)] for index in raw]
    missing = sorted({name for name in species if name not in model_index})
    if missing:
        raise ValueError(f"dataset species missing from model type map at {dataset_dir}: {missing}")
    return int(raw.size), np.asarray([model_index[name] for name in species], dtype=np.int32), type_map


def _load_set(set_dir: Path, natoms: int) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for name in ("coord", "box", "energy", "force"):
        path = set_dir / f"{name}.npy"
        if not path.is_file():
            raise ValueError(f"missing {path.name}: {set_dir}")
        arrays[name] = np.load(str(path), mmap_mode="r", allow_pickle=False)
    frames = int(arrays["energy"].shape[0])
    if frames <= 0 or any(int(array.shape[0]) != frames for array in arrays.values()):
        raise ValueError(f"inconsistent or empty arrays: {set_dir}")
    arrays["coord"] = arrays["coord"].reshape(frames, natoms * 3)
    arrays["box"] = arrays["box"].reshape(frames, 9)
    arrays["energy"] = arrays["energy"].reshape(frames)
    arrays["force"] = arrays["force"].reshape(frames, natoms * 3)
    virial_path = set_dir / "virial.npy"
    if virial_path.is_file():
        virial = np.load(str(virial_path), mmap_mode="r", allow_pickle=False)
        if int(virial.shape[0]) != frames:
            raise ValueError(f"virial frame mismatch: {set_dir}")
        arrays["virial"] = virial.reshape(frames, 9)
    return arrays


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-root", type=Path, help="Root containing DPData/NPY systems.")
    source.add_argument("--manifest", type=Path, help="TSV manifest produced by `dataset build-manifest`.")
    parser.add_argument("--model", type=Path, required=True, help="DeepMD model checkpoint path.")
    parser.add_argument("--output-root", type=Path, required=True, help="New or empty output directory.")
    parser.add_argument("--model-label", default="", help="Report label; defaults to the model filename.")
    parser.add_argument("--chunk-atoms", type=int, default=20000, help="Maximum atoms per inference call.")
    parser.add_argument("--batch-size", type=int, default=0, help="Optional frame cap per call; 0 uses only chunk-atoms.")
    parser.add_argument("--group-by", choices=("block", "label"), default="block", help="Manifest field used for per-family metrics.")
    parser.add_argument("--max-datasets", type=int, default=0, help="Limit system count; 0 means all.")
    parser.add_argument("--max-frames-per-dataset", type=int, default=0, help="Limit frames per system; 0 means all.")
    parser.add_argument("--model-type-map", default="", help="Optional comma-separated override; omitted reads the model type map.")
    parser.add_argument("--save-arrays", action="store_true", help="Store prediction arrays for parity plots; disabled by default.")
    parser.add_argument("--continue-on-error", action="store_true", help="Record invalid systems and continue instead of failing closed.")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else None
    manifest = Path(args.manifest).expanduser().resolve() if args.manifest else None
    model_path = Path(args.model).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if dataset_root is not None and not dataset_root.is_dir():
        raise ValueError(f"dataset-root not found: {dataset_root}")
    if manifest is not None and not manifest.is_file():
        raise ValueError(f"manifest not found: {manifest}")
    if not model_path.is_file():
        raise ValueError(f"model not found: {model_path}")
    if int(args.chunk_atoms) < 0 or int(args.batch_size) < 0:
        raise ValueError("chunk-atoms and batch-size must be non-negative")
    frames_per_chunk(1, int(args.chunk_atoms), int(args.batch_size))
    if int(args.max_datasets) < 0 or int(args.max_frames_per_dataset) < 0:
        raise ValueError("dataset and frame limits must be non-negative")
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = [output_root / name for name in ("summary.json", "per_system.csv", "per_family.csv", "benchmark_arrays.npz")]
    if any(path.exists() for path in output_paths):
        raise FileExistsError(f"refusing existing benchmark outputs in: {output_root}")

    import deepmd
    from deepmd.infer import DeepPot

    started_at = _now()
    wall_started = time.perf_counter()
    dp = DeepPot(str(model_path))
    if str(args.model_type_map).strip():
        model_type_map = [name.strip() for name in str(args.model_type_map).split(",") if name.strip()]
    else:
        try:
            model_type_map = [str(name) for name in dp.get_type_map()]
        except AttributeError as exc:
            raise ValueError("the model does not expose get_type_map(); pass --model-type-map") from exc
    if not model_type_map or len(model_type_map) != len(set(model_type_map)):
        raise ValueError("model type map must contain unique non-empty names")

    specs = _load_system_specs(dataset_root, manifest)
    if int(args.max_datasets) > 0:
        specs = specs[: int(args.max_datasets)]
    if not specs:
        raise ValueError("no DPData/NPY systems found")

    overall = ErrorAccumulator()
    family_accumulators: Dict[str, ErrorAccumulator] = {}
    system_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    array_chunks: Dict[str, List[np.ndarray]] = {
        key: []
        for key in ("energy_true", "energy_pred", "natoms", "force_true", "force_pred", "virial_true", "virial_pred")
    }

    for spec in specs:
        try:
            natoms, atom_types, dataset_type_map = _load_type_info(spec.path, model_type_map)
            set_dirs = sorted(path for path in spec.path.glob("set.*") if path.is_dir())
            if not set_dirs:
                raise ValueError(f"no set.* directories: {spec.path}")
            system_accumulator = ErrorAccumulator()
            inference_seconds = 0.0
            remaining = int(args.max_frames_per_dataset) or None
            batch_frames = frames_per_chunk(natoms, int(args.chunk_atoms), int(args.batch_size))
            for set_dir in set_dirs:
                arrays = _load_set(set_dir, natoms)
                frame_count = int(arrays["energy"].shape[0])
                if remaining is not None:
                    frame_count = min(frame_count, remaining)
                for start in range(0, frame_count, batch_frames):
                    stop = min(start + batch_frames, frame_count)
                    coord = np.asarray(arrays["coord"][start:stop], dtype=np.float64)
                    box = np.asarray(arrays["box"][start:stop], dtype=np.float64)
                    tick = time.perf_counter()
                    output = dp.eval(coord, box, atom_types)
                    inference_seconds += time.perf_counter() - tick
                    energy_pred = np.asarray(output[0], dtype=float).reshape(stop - start)
                    force_pred = np.asarray(output[1], dtype=float).reshape(stop - start, natoms * 3)
                    virial_pred = np.asarray(output[2], dtype=float).reshape(stop - start, 9)
                    energy_true = np.asarray(arrays["energy"][start:stop], dtype=float).reshape(stop - start)
                    force_true = np.asarray(arrays["force"][start:stop], dtype=float).reshape(stop - start, natoms * 3)
                    virial_true = np.asarray(arrays["virial"][start:stop], dtype=float) if "virial" in arrays else None
                    if not np.isfinite(coord).all() or not np.isfinite(box).all() or not np.isfinite(energy_true).all() or not np.isfinite(force_true).all():
                        raise ValueError(f"input contains NaN/Inf: {spec.path}")
                    if virial_true is not None and not np.isfinite(virial_true).all():
                        raise ValueError(f"virial contains NaN/Inf: {spec.path}")
                    if not np.isfinite(energy_pred).all() or not np.isfinite(force_pred).all() or not np.isfinite(virial_pred).all():
                        raise ValueError(f"prediction contains NaN/Inf: {spec.path}")
                    system_accumulator.update(
                        energy_true,
                        energy_pred,
                        natoms,
                        force_true,
                        force_pred,
                        virial_true,
                        virial_pred if virial_true is not None else None,
                    )
                    if args.save_arrays:
                        array_chunks["energy_true"].append(energy_true)
                        array_chunks["energy_pred"].append(energy_pred)
                        array_chunks["natoms"].append(np.full(stop - start, natoms, dtype=int))
                        array_chunks["force_true"].append(force_true.reshape(-1))
                        array_chunks["force_pred"].append(force_pred.reshape(-1))
                        if virial_true is not None:
                            array_chunks["virial_true"].append(virial_true.reshape(-1))
                            array_chunks["virial_pred"].append(virial_pred.reshape(-1))
                if remaining is not None:
                    remaining -= frame_count
                    if remaining <= 0:
                        break
            if system_accumulator.frames <= 0:
                raise ValueError(f"no frames evaluated: {spec.path}")
            system_accumulator.systems = 1
            family = spec.block if args.group_by == "block" else (spec.label or "unlabeled")
            family_accumulators.setdefault(family, ErrorAccumulator()).merge(system_accumulator)
            overall.merge(system_accumulator)
            system_rows.append(
                {
                    "index": spec.index,
                    "family": family,
                    "block": spec.block,
                    "label": spec.label,
                    "system_path": str(spec.path),
                    "natoms": natoms,
                    "sets": len(set_dirs),
                    "frames": system_accumulator.frames,
                    "force_components": system_accumulator.force_count,
                    "virial_components": system_accumulator.virial_count,
                    "batch_frames": batch_frames,
                    "inference_seconds": inference_seconds,
                    "frames_per_second": system_accumulator.frames / inference_seconds if inference_seconds > 0 else None,
                    "dataset_type_map": ",".join(dataset_type_map),
                    **system_accumulator.metrics(),
                }
            )
        except Exception as exc:
            errors.append({"system_path": str(spec.path), "error": str(exc)})
            if not args.continue_on_error:
                break

    if system_rows:
        max(system_rows, key=lambda row: float(row["energy_rmse_meV_per_atom"]))["max_energy_rmse"] = True
        max(system_rows, key=lambda row: float(row["force_rmse_meV_per_A"]))["max_force_rmse"] = True
        for row in system_rows:
            row.setdefault("max_energy_rmse", False)
            row.setdefault("max_force_rmse", False)
    family_rows = [
        {
            "family": family,
            "systems": accumulator.systems,
            "frames": accumulator.frames,
            "force_components": accumulator.force_count,
            "virial_components": accumulator.virial_count,
            **accumulator.metrics(),
        }
        for family, accumulator in sorted(family_accumulators.items())
    ]
    wall_time = time.perf_counter() - wall_started
    status = "FAIL" if not system_rows or (errors and not args.continue_on_error) else "PARTIAL" if errors else "PASS"
    arrays_path: Optional[Path] = None
    if args.save_arrays and overall.frames:
        arrays_path = output_root / "benchmark_arrays.npz"
        np.savez_compressed(
            str(arrays_path),
            **{
                key: np.concatenate(chunks) if chunks else np.asarray([], dtype=float)
                for key, chunks in array_chunks.items()
            },
        )
    summary: Dict[str, Any] = {
        "schema": "nnpgen.deepmd-benchmark.v1",
        "status": status,
        "created_at": _now(),
        "started_at": started_at,
        "host": socket.gethostname(),
        "dataset_root": str(dataset_root) if dataset_root else None,
        "manifest": str(manifest) if manifest else None,
        "manifest_sha256": manifest_sha256(manifest) if manifest else None,
        "model": str(model_path),
        "model_label": str(args.model_label).strip() or model_path.name,
        "model_sha256": _sha256(model_path),
        "deepmd_version": str(getattr(deepmd, "__version__", "unknown")),
        "model_type_map": model_type_map,
        "output_root": str(output_root),
        "arrays_path": str(arrays_path) if arrays_path else None,
        "systems_requested": len(specs),
        "systems_evaluated": len(system_rows),
        "systems_failed": len(errors),
        "frames_total": overall.frames,
        "force_components_total": overall.force_count,
        "virial_components_total": overall.virial_count,
        "chunk_atoms": int(args.chunk_atoms),
        "batch_size_cap": int(args.batch_size),
        "group_by": str(args.group_by),
        "wall_time_seconds": wall_time,
        "frames_per_second": overall.frames / wall_time if wall_time > 0 else None,
        "metrics": overall.metrics() if overall.frames else {},
        "definitions": {
            "energy_rmse_meV_per_atom": "1000*sqrt(mean_frames(((E_pred-E_ref)/N_atoms)^2))",
            "force_rmse_meV_per_A": "1000*sqrt(mean_all_force_components((F_pred-F_ref)^2))",
            "family_metrics": f"SSE/count weighted by manifest {args.group_by}",
        },
        "per_system_csv": str(output_root / "per_system.csv") if system_rows else None,
        "per_family_csv": str(output_root / "per_family.csv") if family_rows else None,
        "errors": errors,
    }
    _write_csv(output_root / "per_system.csv", system_rows)
    _write_csv(output_root / "per_family.csv", family_rows)
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if status == "FAIL":
        raise RuntimeError(f"benchmark failed closed: {errors[0]['error']}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Stream DeepMD predictions over DPData/NPY systems with grouped error reports.")
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
