from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path
from typing import Dict

from ..utils import read_json, str_to_bool, write_json


def _validate_wrapped(atoms: Atoms, tol: float = 1e-6) -> None:
    scaled = atoms.get_scaled_positions(wrap=False)
    mn = float(scaled.min())
    mx = float(scaled.max())
    if mn < -tol or mx > 1.0 + tol:
        raise ValueError(f"Coordinates out of periodic bounds: min={mn:.6e}, max={mx:.6e}")


def _wrap_and_validate(atoms: Atoms, tol: float = 1e-6) -> None:
    import numpy as np

    atoms.wrap()
    _validate_wrapped(atoms, tol=tol)

    # Normalize tiny floating-point drift to [0, 1) while preserving strict failure on real outliers.
    scaled = atoms.get_scaled_positions(wrap=False)
    if float(scaled.min()) < 0.0 or float(scaled.max()) >= 1.0:
        atoms.set_scaled_positions(np.mod(scaled, 1.0))

    _validate_wrapped(atoms, tol=tol)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CPU post-processing for one frame directory")
    p.add_argument("--frame-dir", required=True, help="Absolute frame directory path")
    p.add_argument("--frame-info", default="", help="Optional frame_info.json path")
    p.add_argument("--frame-record", default="", help="Optional output record json path")
    p.add_argument("--pooled-poscar-path", default="", help="Optional pooled POSCAR destination path")
    p.add_argument("--lightweight-poscar-only", choices=["true", "false"], default="false")
    p.add_argument("--cleanup-frame-dir", choices=["true", "false"], default="false")
    p.add_argument("--delete-initial", choices=["true", "false"], default="true")
    p.add_argument("--delete-slurm-logs", choices=["true", "false"], default="false")
    return p


def run_post(args: argparse.Namespace) -> None:
    from ase.io import iread, read, write

    from ..poscar import atoms_to_poscar

    frame_dir = Path(args.frame_dir).resolve()
    frame_info_path = Path(args.frame_info).resolve() if args.frame_info else None
    frame_record_path = Path(args.frame_record).resolve() if args.frame_record else None
    pooled_poscar_path = Path(args.pooled_poscar_path).resolve() if args.pooled_poscar_path else None

    lightweight = str_to_bool(args.lightweight_poscar_only)
    cleanup_frame_dir = str_to_bool(args.cleanup_frame_dir)
    delete_initial = str_to_bool(args.delete_initial)
    delete_slurm_logs = str_to_bool(args.delete_slurm_logs)

    if cleanup_frame_dir and frame_record_path is None:
        raise ValueError("--frame-record is required when --cleanup-frame-dir true")

    final_xyz = frame_dir / "final.xyz"
    poscar = frame_dir / "POSCAR"
    traj_xyz = frame_dir / "traj.xyz"
    initial_xyz = frame_dir / "initial.xyz"

    if not final_xyz.exists():
        raise FileNotFoundError(f"Missing final.xyz: {final_xyz}")

    final_atoms = read(str(final_xyz), index=-1)
    _wrap_and_validate(final_atoms)
    write(str(final_xyz), final_atoms, format="extxyz")
    atoms_to_poscar(final_atoms, poscar)

    traj_frames = 0
    traj_wrapped = True
    if traj_xyz.exists():
        for atoms in iread(str(traj_xyz), index=":"):
            traj_frames += 1
            try:
                _validate_wrapped(atoms)
            except Exception:
                traj_wrapped = False
                break

    pooled_written = False
    if pooled_poscar_path is not None:
        pooled_poscar_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(poscar), str(pooled_poscar_path))
        pooled_written = True

    info: Dict = {}
    if frame_info_path is not None and frame_info_path.exists():
        info = read_json(frame_info_path)

    paths = dict(info.get("paths", {}))
    paths["frame_dir"] = str(frame_dir)
    paths["final_xyz"] = str(final_xyz)
    paths["final_poscar"] = str(poscar)
    paths["traj_xyz"] = str(traj_xyz)
    if pooled_poscar_path is not None:
        paths["pooled_poscar"] = str(pooled_poscar_path)

    info["paths"] = paths
    info["traj_frames"] = traj_frames
    info["traj_wrapped"] = traj_wrapped
    info["pooled_written"] = pooled_written
    info["lightweight_poscar_only"] = lightweight
    info["post_status"] = "finished"
    info["status"] = "finished"

    removed = []
    if lightweight:
        if traj_xyz.exists():
            traj_xyz.unlink()
            removed.append(str(traj_xyz))
        if delete_initial and initial_xyz.exists():
            initial_xyz.unlink()
            removed.append(str(initial_xyz))
        if delete_slurm_logs:
            for log_path in glob.glob(str(frame_dir / "slurm-*.out")):
                p = Path(log_path)
                if p.exists():
                    p.unlink()
                    removed.append(str(p))

    info["removed_paths"] = removed

    if cleanup_frame_dir:
        info["record_type"] = "frame_record"
        if frame_record_path.parent != frame_dir and not frame_record_path.parent.exists():
            frame_record_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(frame_record_path, info)
        shutil.rmtree(str(frame_dir))
    else:
        if frame_info_path is not None:
            write_json(frame_info_path, info)

    print(f"Post done: frame_dir={frame_dir} pooled_poscar={pooled_poscar_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_post(args)


if __name__ == "__main__":
    main()
