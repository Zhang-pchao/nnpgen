#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _discover_dp_leaves(dataset_root: Path) -> List[Path]:
    leaves: List[Path] = []
    for type_raw in dataset_root.rglob("type.raw"):
        d = type_raw.parent
        if not (d / "type_map.raw").is_file():
            continue
        set_dirs = [p for p in sorted(d.glob("set.*")) if p.is_dir() and (p / "coord.npy").is_file() and (p / "box.npy").is_file()]
        if set_dirs:
            leaves.append(d)
    uniq = sorted({str(p.resolve()): p for p in leaves}.values(), key=lambda x: str(x))
    return uniq


def _load_type_info(dataset_dir: Path, model_type_map: List[str]) -> Tuple[int, np.ndarray, List[str], List[str]]:
    tmap = [ln.strip() for ln in (dataset_dir / "type_map.raw").read_text().splitlines() if ln.strip()]
    raw = np.loadtxt(str(dataset_dir / "type.raw"), dtype=int, ndmin=1)
    if raw.ndim != 1:
        raw = raw.reshape(-1)

    species: List[str] = []
    for x in raw:
        i = int(x)
        if i < 0 or i >= len(tmap):
            raise ValueError("type.raw index out of range in {0}".format(dataset_dir))
        species.append(tmap[i])

    model_index = {e: i for i, e in enumerate(model_type_map)}
    missing = sorted({s for s in species if s not in model_index})
    if missing:
        raise ValueError("Dataset contains species not in model type map: {0}".format(missing))

    atype_model = np.array([model_index[s] for s in species], dtype=np.int32)
    return int(raw.size), atype_model, species, tmap


def _reshape_coord(arr: np.ndarray, natoms: int) -> np.ndarray:
    arr = np.array(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.reshape(arr, (-1, natoms * 3))


def _reshape_box(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.reshape(arr, (-1, 9))


def _reshape_energy(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr)
    return np.reshape(arr, (-1,))


def _reshape_force(arr: np.ndarray, natoms: int) -> np.ndarray:
    arr = np.array(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.reshape(arr, (-1, natoms * 3))


def _reshape_virial(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.reshape(arr, (-1, 9))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", required=True, help="Root directory containing DP test datasets")
    parser.add_argument("--model", required=True, help="DeepMD model checkpoint path")
    parser.add_argument("--output-root", required=True, help="Output directory for benchmark arrays and summary")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--max-datasets", type=int, default=0, help="Limit dataset leaf count (0 means all)")
    parser.add_argument("--max-frames-per-dataset", type=int, default=0, help="Limit frames per dataset leaf (0 means all)")
    parser.add_argument(
        "--model-type-map",
        default="H,O,N,Na,Cl,Ti,C,Si",
        help="Comma-separated model type map",
    )


def run(args: argparse.Namespace) -> Dict[str, object]:
    dataset_root = Path(args.dataset_root)
    model_path = Path(args.model)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not dataset_root.is_dir():
        raise ValueError("dataset-root not found: {0}".format(dataset_root))
    if not model_path.is_file():
        raise ValueError("model not found: {0}".format(model_path))
    if int(args.batch_size) <= 0:
        raise ValueError("batch-size must be > 0")

    model_type_map = [x.strip() for x in str(args.model_type_map).split(",") if x.strip()]
    if not model_type_map:
        raise ValueError("model-type-map is empty")

    leaves = _discover_dp_leaves(dataset_root)
    if not leaves:
        raise ValueError("No DP datasets found under: {0}".format(dataset_root))
    if int(args.max_datasets) > 0:
        leaves = leaves[: int(args.max_datasets)]

    from deepmd.infer import DeepPot

    dp = DeepPot(str(model_path))

    true_e_chunks: List[np.ndarray] = []
    pred_e_chunks: List[np.ndarray] = []
    natom_chunks: List[np.ndarray] = []
    true_f_chunks: List[np.ndarray] = []
    pred_f_chunks: List[np.ndarray] = []
    true_v_chunks: List[np.ndarray] = []
    pred_v_chunks: List[np.ndarray] = []

    per_dataset: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []

    for leaf in leaves:
        try:
            natoms, atype_model, _, data_type_map = _load_type_info(leaf, model_type_map)
            set_dirs = [p for p in sorted(leaf.glob("set.*")) if p.is_dir() and (p / "coord.npy").is_file() and (p / "box.npy").is_file()]

            ds_frames_total = 0
            ds_virial_frames = 0

            for set_dir in set_dirs:
                coord = _reshape_coord(np.load(str(set_dir / "coord.npy")), natoms)
                box = _reshape_box(np.load(str(set_dir / "box.npy")))

                if not (set_dir / "energy.npy").is_file() or not (set_dir / "force.npy").is_file():
                    continue
                energy = _reshape_energy(np.load(str(set_dir / "energy.npy")))
                force = _reshape_force(np.load(str(set_dir / "force.npy"), allow_pickle=False), natoms)

                n = coord.shape[0]
                if box.shape[0] != n or energy.shape[0] != n or force.shape[0] != n:
                    raise ValueError("shape mismatch in {0}".format(set_dir))

                virial = None
                if (set_dir / "virial.npy").is_file():
                    virial = _reshape_virial(np.load(str(set_dir / "virial.npy"), allow_pickle=False))
                    if virial.shape[0] != n:
                        raise ValueError("virial shape mismatch in {0}".format(set_dir))

                if int(args.max_frames_per_dataset) > 0:
                    n = min(n, int(args.max_frames_per_dataset) - ds_frames_total)
                    if n <= 0:
                        break
                    coord = coord[:n]
                    box = box[:n]
                    energy = energy[:n]
                    force = force[:n]
                    if virial is not None:
                        virial = virial[:n]

                bsize = int(args.batch_size)
                for i in range(0, n, bsize):
                    j = min(i + bsize, n)
                    c = coord[i:j]
                    b = box[i:j]
                    e_t = energy[i:j]
                    f_t = force[i:j]
                    v_t = virial[i:j] if virial is not None else None

                    out = dp.eval(c.astype(np.float64), b.astype(np.float64), atype_model)
                    e_p = np.array(out[0], dtype=float).reshape(-1)
                    f_p = np.array(out[1], dtype=float).reshape(j - i, natoms * 3)
                    v_p = np.array(out[2], dtype=float).reshape(j - i, 9)

                    true_e_chunks.append(e_t.reshape(-1))
                    pred_e_chunks.append(e_p.reshape(-1))
                    natom_chunks.append(np.full((j - i,), natoms, dtype=int))

                    true_f_chunks.append(f_t.reshape(-1))
                    pred_f_chunks.append(f_p.reshape(-1))

                    if v_t is not None:
                        true_v_chunks.append(v_t.reshape(-1))
                        pred_v_chunks.append(v_p.reshape(-1))
                        ds_virial_frames += (j - i)

                ds_frames_total += n

            per_dataset.append(
                {
                    "dataset_dir": str(leaf),
                    "natoms": natoms,
                    "sets": len(set_dirs),
                    "frames_used": ds_frames_total,
                    "virial_frames_used": ds_virial_frames,
                    "dataset_type_map": data_type_map,
                }
            )

        except Exception as exc:
            errors.append({"dataset_dir": str(leaf), "error": str(exc)})

    if not true_e_chunks:
        raise ValueError("No valid frames were benchmarked; check dataset format and model type map")

    energy_true = np.concatenate(true_e_chunks)
    energy_pred = np.concatenate(pred_e_chunks)
    natoms = np.concatenate(natom_chunks)
    force_true = np.concatenate(true_f_chunks)
    force_pred = np.concatenate(pred_f_chunks)

    if true_v_chunks and pred_v_chunks:
        virial_true = np.concatenate(true_v_chunks)
        virial_pred = np.concatenate(pred_v_chunks)
    else:
        virial_true = np.array([], dtype=float)
        virial_pred = np.array([], dtype=float)

    arrays_path = output_root / "benchmark_arrays.npz"
    np.savez_compressed(
        str(arrays_path),
        energy_true=energy_true,
        energy_pred=energy_pred,
        natoms=natoms,
        force_true=force_true,
        force_pred=force_pred,
        virial_true=virial_true,
        virial_pred=virial_pred,
    )

    metrics = {
        "energy_rmse_eV": float(np.sqrt(np.mean((energy_pred - energy_true) ** 2))),
        "energy_mae_eV": float(np.mean(np.abs(energy_pred - energy_true))),
        "energy_rmse_meV_per_atom": float(np.sqrt(np.mean((((energy_pred - energy_true) / natoms) * 1000.0) ** 2))),
        "energy_mae_meV_per_atom": float(np.mean(np.abs(((energy_pred - energy_true) / natoms) * 1000.0))),
        "force_rmse_eVA": float(np.sqrt(np.mean((force_pred - force_true) ** 2))),
        "force_mae_eVA": float(np.mean(np.abs(force_pred - force_true))),
    }

    if virial_true.size > 0 and virial_pred.size > 0:
        metrics["virial_rmse"] = float(np.sqrt(np.mean((virial_pred - virial_true) ** 2)))
        metrics["virial_mae"] = float(np.mean(np.abs(virial_pred - virial_true)))

    summary = {
        "created_at": _now(),
        "dataset_root": str(dataset_root),
        "model": str(model_path),
        "output_root": str(output_root),
        "arrays_path": str(arrays_path),
        "dataset_leaf_count": len(leaves),
        "dataset_success_count": len(per_dataset),
        "dataset_error_count": len(errors),
        "frames_total": int(energy_true.size),
        "force_components_total": int(force_true.size),
        "virial_components_total": int(virial_true.size),
        "metrics": metrics,
        "per_dataset": per_dataset,
        "errors": errors,
    }

    summary_path = output_root / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict DP test datasets with DeepMD model and store parity arrays")
    add_arguments(parser)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
