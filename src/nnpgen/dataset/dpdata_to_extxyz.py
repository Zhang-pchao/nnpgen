#!/usr/bin/env python3
import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _load_atom_types(dataset_dir: Path) -> Tuple[List[str], np.ndarray, List[str]]:
    type_map_path = dataset_dir / "type_map.raw"
    type_raw_path = dataset_dir / "type.raw"
    if not type_map_path.is_file():
        raise ValueError("Missing type_map.raw: {0}".format(type_map_path))
    if not type_raw_path.is_file():
        raise ValueError("Missing type.raw: {0}".format(type_raw_path))

    type_map = [ln.strip() for ln in type_map_path.read_text().splitlines() if ln.strip()]
    atom_types = np.loadtxt(str(type_raw_path), dtype=int, ndmin=1)
    if atom_types.ndim != 1:
        atom_types = atom_types.reshape(-1)

    if len(type_map) == 0:
        raise ValueError("type_map.raw is empty: {0}".format(type_map_path))

    atom_species: List[str] = []
    for t in atom_types:
        ti = int(t)
        if ti < 0 or ti >= len(type_map):
            raise ValueError("type.raw index out of range: {0} (len type_map={1})".format(ti, len(type_map)))
        atom_species.append(type_map[ti])

    return type_map, atom_types, atom_species


def _iter_set_dirs(dataset_dir: Path):
    # support both single-dataset layout (<root>/set.000) and grouped layout
    # (<root>/group_xxx_*/set.000)
    set_dirs = [Path(p) for p in sorted(glob.glob(str(dataset_dir / "**" / "set.*"), recursive=True))]
    if not set_dirs:
        raise ValueError("No set.* directories found under: {0}".format(dataset_dir))

    for sdir in set_dirs:
        box_path = sdir / "box.npy"
        coord_path = sdir / "coord.npy"
        force_path = sdir / "force.npy"
        energy_path = sdir / "energy.npy"
        virial_path = sdir / "virial.npy"

        if not box_path.is_file() or not coord_path.is_file() or not force_path.is_file():
            continue

        box = np.load(str(box_path))
        coord = np.load(str(coord_path))
        force = np.load(str(force_path))
        energy = np.load(str(energy_path)) if energy_path.is_file() else None
        virial = np.load(str(virial_path)) if virial_path.is_file() else None

        yield sdir.parent, sdir, box, coord, force, energy, virial


def add_dpdata_to_extxyz_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-dir", required=True, help="DP data directory containing type.raw and set.*")
    parser.add_argument("--output-xyz", default="", help="Output EXTXYZ file path (default: <dataset-dir>/DEEP2XYZ.xyz)")


def run_dpdata_to_extxyz(args: argparse.Namespace) -> Dict[str, object]:
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        raise ValueError("dataset-dir not found: {0}".format(dataset_dir))

    out_xyz = Path(args.output_xyz) if str(args.output_xyz).strip() else (dataset_dir / "DEEP2XYZ.xyz")
    out_xyz.parent.mkdir(parents=True, exist_ok=True)

    frames_written = 0
    set_dirs_used = 0
    dataset_roots_used: List[str] = []
    type_cache: Dict[str, Tuple[List[str], np.ndarray, List[str]]] = {}

    with out_xyz.open("w") as fw:
        for data_root, sdir, box, coord, force, energy, virial in _iter_set_dirs(dataset_dir):
            key = str(data_root)
            if key not in type_cache:
                type_cache[key] = _load_atom_types(data_root)
                dataset_roots_used.append(key)

            _, atom_types, atom_species = type_cache[key]
            natoms = int(atom_types.size)

            nframe_box = np.reshape(box, (-1, 9)).shape[0]
            box_r = np.reshape(box, (nframe_box, 9))
            coord_r = np.reshape(coord, (nframe_box, natoms, 3))
            force_r = np.reshape(force, (nframe_box, natoms, 3))

            energy_r = None
            if energy is not None and energy.size > 0:
                energy_r = np.reshape(energy, (nframe_box,))

            virial_r = None
            if virial is not None and virial.size > 0:
                virial_r = np.reshape(virial, (nframe_box, 9))

            for i in range(nframe_box):
                e = float(energy_r[i]) if energy_r is not None else 0.0
                lattice = " ".join(map(str, box_r[i].tolist()))

                hdr = []
                hdr.append("energy={0}".format(e))
                hdr.append('config_type=dpdata_to_extxyz')
                hdr.append('pbc="T T T"')
                if virial_r is not None:
                    hdr.append('virial="{0}"'.format(" ".join(map(str, virial_r[i].tolist()))))
                hdr.append('Lattice="{0}"'.format(lattice))
                hdr.append("Properties=species:S:1:pos:R:3:force:R:3")

                fw.write(str(natoms) + "\n")
                fw.write(" ".join(hdr) + "\n")

                for j in range(natoms):
                    pos = coord_r[i, j]
                    frc = force_r[i, j]
                    fw.write(
                        "{0} {1} {2} {3} {4} {5} {6}\n".format(
                            atom_species[j],
                            pos[0],
                            pos[1],
                            pos[2],
                            frc[0],
                            frc[1],
                            frc[2],
                        )
                    )

                frames_written += 1

            set_dirs_used += 1

    summary = {
        "dataset_dir": str(dataset_dir),
        "dataset_roots_used": dataset_roots_used,
        "set_dirs_used": set_dirs_used,
        "frames_written": frames_written,
        "output_xyz": str(out_xyz),
    }
    (dataset_dir / "dp2xyz_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DP data (set.*) to Extended XYZ")
    add_dpdata_to_extxyz_arguments(parser)
    args = parser.parse_args()
    run_dpdata_to_extxyz(args)


if __name__ == "__main__":
    main()
