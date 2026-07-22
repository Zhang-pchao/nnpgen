"""Build alpha-SiO2 slab systems with hemispherical or spherical N2 nanobubbles.

The builder is intentionally ASE-native because Packmol box constraints do not
represent the tilted alpha-SiO2 surface cell cleanly. Candidate molecule centers
are sampled in fractional x/y coordinates, then mapped through the full cell.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.geometry import get_distances
from ase.io import read, write


AVOGADRO = 6.02214076e23
H2O_MOLAR_MASS_G_MOL = 18.01528
N2_MOLAR_MASS_G_MOL = 28.0134
SPECIES_TEMPLATE_FILES = {
    "H2O": "H2O.xyz",
    "N2": "N2.xyz",
    "Na": "Na.xyz",
    "Cl": "Cl.xyz",
    "H3O": "H3O+.xyz",
    "OH": "OH-.xyz",
}
PLACEMENT_ORDER = ("N2", "Na", "Cl", "H3O", "OH", "H2O")
DEFAULT_ELEMENT_ORDER = ("H", "O", "N", "Na", "Cl", "Ti", "C", "Si")
CELL_ZERO_TOLERANCE = 1.0e-5


@dataclass
class SpeciesTemplate:
    name: str
    path: Path
    requested: int
    atoms_centered: Atoms
    diameter: float


@dataclass
class PlacementRecord:
    species_name: str
    atoms: Atoms


@dataclass
class BuildGeometry:
    area_xy: float
    slab_bottom_z: float
    slab_top_z: float
    solution_z_min: float
    solution_z_max: float
    bubble_base_z: float
    bubble_top_z: float
    bubble_radius: float
    bubble_center_z: float
    bubble_center_u: float
    bubble_center_v: float
    bubble_volume_a3: float
    solution_volume_a3: float
    water_volume_a3: float


def molecules_per_a3(density_kg_m3: float, molar_mass_g_mol: float) -> float:
    return (density_kg_m3 * 1.0e-27) / (molar_mass_g_mol / AVOGADRO)


def spherical_cap_volume(radius: float, height: float) -> float:
    h = max(0.0, min(float(height), float(radius)))
    return math.pi * h * h * (float(radius) - h / 3.0)


def sphere_volume(radius: float) -> float:
    return 4.0 * math.pi * float(radius) ** 3 / 3.0


def template_diameter(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    max_d = 0.0
    for i in range(len(positions) - 1):
        diffs = positions[i + 1 :] - positions[i]
        if diffs.size:
            max_d = max(max_d, float(np.sqrt(np.sum(diffs * diffs, axis=1)).max()))
    return max_d


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    q1 = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    q2 = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    return np.array(
        [
            [1 - 2 * (q3 * q3 + q4 * q4), 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)],
            [2 * (q2 * q3 + q1 * q4), 1 - 2 * (q2 * q2 + q4 * q4), 2 * (q3 * q4 - q1 * q2)],
            [2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3)],
        ],
        dtype=float,
    )


def add_sio2_nanobubble_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-slab", type=Path, required=True, help="Input alpha-SiO2 slab extxyz.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated outputs.")
    parser.add_argument("--output-prefix", default="sio2_nanobubble", help="Output filename prefix.")
    parser.add_argument("--template-dir", type=Path, required=True, help="Directory containing H2O/N2/ion xyz templates.")

    parser.add_argument("--repeat-a", type=int, default=1, help="ASE repeat count along cell vector a.")
    parser.add_argument("--repeat-b", type=int, default=1, help="ASE repeat count along cell vector b.")
    parser.add_argument(
        "--orthorhombic-hex120-supercell",
        action="store_true",
        help="Use new vectors repeat_a*a and repeat_b*(a+2b), useful for 120-degree alpha-SiO2 cells.",
    )
    parser.add_argument(
        "--control-no-slab",
        action="store_true",
        help="Use the source only for the repeated cell and build a no-interface control system.",
    )
    parser.add_argument(
        "--control-bottom-buffer",
        type=float,
        default=0.0,
        help="Bottom z buffer for --control-no-slab before the fill region starts.",
    )
    parser.add_argument("--target-bottom-z", type=float, default=5.0, help="Shift slab so min z equals this value.")
    parser.add_argument("--surface-gap", type=float, default=2.2, help="Gap between slab top and the solution fill-box bottom.")
    parser.add_argument(
        "--bubble-bottom-gap",
        type=float,
        default=None,
        help="Optional gap between slab top and the N2 bubble lower tangent point; defaults to --surface-gap.",
    )
    parser.add_argument("--solution-height", type=float, default=20.0, help="Solution fill-box height above its z-min.")
    parser.add_argument("--top-buffer", type=float, default=4.0, help="Empty z buffer above the solution region.")
    parser.add_argument("--bubble-radius", type=float, default=8.0, help="Bubble radius in Angstrom.")
    parser.add_argument("--bubble-shape", choices=["hemisphere", "sphere"], default="hemisphere", help="N2 bubble geometry.")
    parser.add_argument("--bubble-center-u", type=float, default=0.5, help="Bubble center fractional u coordinate.")
    parser.add_argument("--bubble-center-v", type=float, default=0.5, help="Bubble center fractional v coordinate.")
    parser.add_argument("--bubble-clearance", type=float, default=1.0, help="Extra water/ion exclusion shell around bubble.")

    parser.add_argument("--h2o-count", type=int, default=None, help="Explicit H2O molecule count.")
    parser.add_argument("--n2-count", type=int, default=None, help="Explicit N2 molecule count.")
    parser.add_argument("--salt-pairs", type=int, default=0, help="Convenience count for equal Na and Cl ions.")
    parser.add_argument("--na-count", type=int, default=None, help="Explicit Na atom count.")
    parser.add_argument("--cl-count", type=int, default=None, help="Explicit Cl atom count.")
    parser.add_argument("--h3o-count", type=int, default=0, help="H3O molecule count.")
    parser.add_argument("--oh-count", type=int, default=0, help="OH molecule count.")
    parser.add_argument(
        "--target-ph-naoh",
        type=float,
        default=None,
        help="Add Na/OH pairs from pH using [OH-] = 10^(pH - 14) and the water-region volume.",
    )
    parser.add_argument("--h2o-density-kg-m3", type=float, default=950.0, help="Density used for automatic H2O count.")
    parser.add_argument("--n2-density-kg-m3", type=float, default=350.0, help="Density used for automatic N2 count.")

    parser.add_argument("--buffer-xy", type=float, default=1.2, help="Fractional-cell edge buffer converted from Angstrom.")
    parser.add_argument("--min-slab-distance", type=float, default=1.8, help="Minimum inserted atom to slab atom distance.")
    parser.add_argument("--min-inserted-distance", type=float, default=1.4, help="Minimum atom distance among inserted species.")
    parser.add_argument("--placement-backend", choices=["ase", "packmol"], default="ase")
    parser.add_argument("--packmol-bin", default="packmol", help="Packmol executable used by --placement-backend packmol.")
    parser.add_argument("--packmol-tolerance", type=float, default=None, help="Defaults to --min-inserted-distance.")
    parser.add_argument("--max-attempts-per-object", type=int, default=25000)
    parser.add_argument("--atom-z-slack", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--estimate-only", action="store_true", help="Write count/geometry summary without inserting species.")
    parser.add_argument("--skip-lammps-data", action="store_true", help="Do not write ASE LAMMPS atomic data.")
    parser.add_argument("--write-run-scripts", action="store_true", help="Write reusable shell and SLURM scripts.")
    parser.add_argument("--run-script-dir", type=Path, default=None, help="Directory for reusable run scripts.")
    parser.add_argument("--slurm-script-dir", type=Path, default=None, help="Directory for generated SLURM scripts.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory for build and SLURM logs.")
    parser.add_argument("--conda-env", default=os.environ.get("NNPGEN_CONDA_ENV", "mace_ase_2"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root to use in generated run scripts.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SiO2 slab plus N2 nanobubble solution model.")
    add_sio2_nanobubble_arguments(parser)
    return parser.parse_args(argv)


def read_first_frame(path: Path) -> Atoms:
    atoms = read(str(path), index=0)
    if not isinstance(atoms, Atoms):
        raise TypeError(f"Expected one Atoms frame from {path}")
    return atoms


def bubble_bottom_gap(args: argparse.Namespace) -> float:
    value = getattr(args, "bubble_bottom_gap", None)
    return float(args.surface_gap if value is None else value)


def bubble_z_extent(solution_z_min: float, solution_z_max: float, args: argparse.Namespace) -> Tuple[float, float, float, float]:
    radius = float(args.bubble_radius)
    base_z = float(solution_z_min) - float(args.surface_gap) + bubble_bottom_gap(args)
    if args.bubble_shape == "sphere":
        height = 2.0 * radius
        center_z = base_z + radius
        volume = sphere_volume(radius)
    else:
        available_height = max(0.0, float(solution_z_max) - base_z)
        height = min(radius, available_height)
        center_z = base_z
        volume = spherical_cap_volume(radius, height)
    return base_z, base_z + height, center_z, volume


def prepare_slab(args: argparse.Namespace) -> Atoms:
    if args.repeat_a < 1 or args.repeat_b < 1:
        raise ValueError("--repeat-a and --repeat-b must be positive integers")
    slab = read_first_frame(args.source_slab).copy()
    if args.orthorhombic_hex120_supercell:
        matrix = np.array(
            [
                [int(args.repeat_a), 0, 0],
                [int(args.repeat_b), 2 * int(args.repeat_b), 0],
                [0, 0, 1],
            ],
            dtype=int,
        )
        slab = make_supercell(slab, matrix, wrap=True)
    else:
        slab = slab.repeat((args.repeat_a, args.repeat_b, 1))

    cell = np.array(slab.cell.array, dtype=float)
    cell[np.abs(cell) < CELL_ZERO_TOLERANCE] = 0.0
    slab.set_cell(cell, scale_atoms=False)
    slab.set_pbc([True, True, True])

    if args.control_no_slab:
        cell = np.array(slab.cell.array, dtype=float)
        solution_z_min = float(args.control_bottom_buffer)
        solution_z_max = solution_z_min + float(args.solution_height)
        _, bubble_top_z, _, _ = bubble_z_extent(solution_z_min, solution_z_max, args)
        target_cz = max(solution_z_max, bubble_top_z) + float(args.top_buffer)
        if target_cz <= 0.0:
            raise ValueError("Computed non-positive no-slab control c-axis length")
        if abs(cell[2, 2]) < 1e-12:
            raise ValueError("The source cell must have a non-zero Cartesian z component in c")
        cell[2, :] *= target_cz / float(cell[2, 2])
        empty = Atoms(symbols=[], positions=np.empty((0, 3)), cell=cell, pbc=[True, True, True])
        empty.info["control_no_slab"] = True
        return empty

    positions = slab.get_positions()
    dz = float(args.target_bottom_z) - float(np.min(positions[:, 2]))
    positions[:, 2] += dz
    slab.set_positions(positions)

    cell = np.array(slab.cell.array, dtype=float)
    top_z = float(np.max(slab.get_positions()[:, 2]))
    solution_z_min = top_z + float(args.surface_gap)
    solution_z_max = solution_z_min + float(args.solution_height)
    _, bubble_top_z, _, _ = bubble_z_extent(solution_z_min, solution_z_max, args)
    target_cz = max(solution_z_max, bubble_top_z) + float(args.top_buffer)
    if target_cz <= 0.0:
        raise ValueError("Computed non-positive c-axis length")
    if abs(cell[2, 2]) < 1e-12:
        raise ValueError("The slab cell must have a non-zero Cartesian z component in c")
    scale = target_cz / float(cell[2, 2])
    cell[2, :] *= scale
    slab.set_cell(cell, scale_atoms=False)
    return slab


def build_geometry(slab: Atoms, args: argparse.Namespace) -> BuildGeometry:
    cell = np.array(slab.cell.array, dtype=float)
    area_xy = float(np.linalg.norm(np.cross(cell[0], cell[1])))
    if area_xy <= 0.0:
        raise ValueError("Invalid xy cell area")
    if len(slab) == 0:
        slab_bottom_z = float("nan")
        slab_top_z = float(args.control_bottom_buffer) - float(args.surface_gap)
        solution_z_min = float(args.control_bottom_buffer)
    else:
        positions = slab.get_positions()
        slab_bottom_z = float(np.min(positions[:, 2]))
        slab_top_z = float(np.max(positions[:, 2]))
        solution_z_min = slab_top_z + float(args.surface_gap)
    solution_z_max = solution_z_min + float(args.solution_height)
    bubble_radius = float(args.bubble_radius)
    solution_height = max(0.0, solution_z_max - solution_z_min)
    bubble_base_z, bubble_top_z, bubble_center_z, bubble_volume = bubble_z_extent(solution_z_min, solution_z_max, args)
    if bubble_base_z < solution_z_min - 1.0e-8:
        raise ValueError("--bubble-bottom-gap must keep the bubble bottom inside or above the solution fill box")
    if bubble_top_z > solution_z_max + 1.0e-8:
        raise ValueError("--solution-height and --bubble-bottom-gap must keep the bubble inside the solution fill box")
    solution_volume = area_xy * solution_height
    water_volume = max(0.0, solution_volume - bubble_volume)
    return BuildGeometry(
        area_xy=area_xy,
        slab_bottom_z=slab_bottom_z,
        slab_top_z=slab_top_z,
        solution_z_min=solution_z_min,
        solution_z_max=solution_z_max,
        bubble_base_z=bubble_base_z,
        bubble_top_z=bubble_top_z,
        bubble_radius=bubble_radius,
        bubble_center_z=bubble_center_z,
        bubble_center_u=float(args.bubble_center_u) % 1.0,
        bubble_center_v=float(args.bubble_center_v) % 1.0,
        bubble_volume_a3=bubble_volume,
        solution_volume_a3=solution_volume,
        water_volume_a3=water_volume,
    )


def resolve_counts(args: argparse.Namespace, geometry: BuildGeometry) -> Dict[str, int]:
    h2o_count = args.h2o_count
    if h2o_count is None:
        h2o_count = int(geometry.water_volume_a3 * molecules_per_a3(args.h2o_density_kg_m3, H2O_MOLAR_MASS_G_MOL))
    n2_count = args.n2_count
    if n2_count is None:
        n2_count = int(geometry.bubble_volume_a3 * molecules_per_a3(args.n2_density_kg_m3, N2_MOLAR_MASS_G_MOL))

    naoh_count = 0
    if args.target_ph_naoh is not None:
        oh_concentration_mol_l = 10.0 ** (float(args.target_ph_naoh) - 14.0)
        water_volume_l = geometry.water_volume_a3 * 1.0e-27
        naoh_count = int(math.ceil(oh_concentration_mol_l * water_volume_l * AVOGADRO))

    na_count = args.salt_pairs if args.na_count is None else args.na_count
    cl_count = args.salt_pairs if args.cl_count is None else args.cl_count
    counts = {
        "H2O": max(0, int(h2o_count)),
        "N2": max(0, int(n2_count)),
        "Na": max(0, int(na_count) + naoh_count),
        "Cl": max(0, int(cl_count)),
        "H3O": max(0, int(args.h3o_count)),
        "OH": max(0, int(args.oh_count) + naoh_count),
    }
    return counts


def load_templates(template_dir: Path, counts: Dict[str, int]) -> Dict[str, SpeciesTemplate]:
    templates: Dict[str, SpeciesTemplate] = {}
    for name, count in counts.items():
        if count <= 0:
            continue
        path = template_dir / SPECIES_TEMPLATE_FILES[name]
        if not path.exists():
            raise FileNotFoundError(f"Missing template for {name}: {path}")
        atoms = read_first_frame(path).copy()
        atoms.set_positions(atoms.get_positions() - atoms.get_center_of_mass())
        templates[name] = SpeciesTemplate(
            name=name,
            path=path,
            requested=count,
            atoms_centered=atoms,
            diameter=template_diameter(atoms.get_positions()),
        )
    return templates


def fractional_xy_buffer(cell: np.ndarray, buffer_xy: float) -> Tuple[float, float]:
    a_len = float(np.linalg.norm(cell[0]))
    b_len = float(np.linalg.norm(cell[1]))
    if a_len <= 2.0 * buffer_xy or b_len <= 2.0 * buffer_xy:
        raise ValueError("--buffer-xy is too large for the repeated slab cell")
    return float(buffer_xy) / a_len, float(buffer_xy) / b_len


def atoms_inside_xy_buffer(trial_pos: np.ndarray, cell: np.ndarray, buffer_xy: float) -> bool:
    u_buffer, v_buffer = fractional_xy_buffer(cell, buffer_xy)
    scaled = np.asarray(trial_pos, dtype=float) @ np.linalg.inv(cell)
    u = scaled[:, 0]
    v = scaled[:, 1]
    return bool(
        np.all(u >= u_buffer)
        and np.all(u <= 1.0 - u_buffer)
        and np.all(v >= v_buffer)
        and np.all(v <= 1.0 - v_buffer)
    )


def point_from_uvz(cell: np.ndarray, u: float, v: float, z: float) -> np.ndarray:
    base = u * cell[0] + v * cell[1]
    c_z = float(cell[2, 2])
    if abs(c_z) < 1e-12:
        raise ValueError("The c vector must have a non-zero z component")
    w = (float(z) - float(base[2])) / c_z
    return base + w * cell[2]


def periodic_uv_delta(u: float, v: float, center_u: float, center_v: float) -> Tuple[float, float]:
    du = (float(u) - float(center_u) + 0.5) % 1.0 - 0.5
    dv = (float(v) - float(center_v) + 0.5) % 1.0 - 0.5
    return du, dv


def bubble_r2_from_uvz(cell: np.ndarray, u: float, v: float, z: float, geometry: BuildGeometry) -> float:
    du, dv = periodic_uv_delta(u, v, geometry.bubble_center_u, geometry.bubble_center_v)
    dxy = du * cell[0] + dv * cell[1]
    rxy2 = float(np.dot(dxy[:2], dxy[:2]))
    dz = float(z) - float(geometry.bubble_center_z)
    return rxy2 + dz * dz


def sample_candidate_center(
    rng: np.random.Generator,
    cell: np.ndarray,
    geometry: BuildGeometry,
    args: argparse.Namespace,
    region: str,
) -> Tuple[float, float, float, np.ndarray]:
    u_buffer, v_buffer = fractional_xy_buffer(cell, args.buffer_xy)
    radius = geometry.bubble_radius
    if region == "bubble" and geometry.bubble_top_z <= geometry.bubble_base_z:
        raise ValueError("Bubble region has no positive height")

    for _ in range(10000):
        u = float(rng.uniform(u_buffer, 1.0 - u_buffer))
        v = float(rng.uniform(v_buffer, 1.0 - v_buffer))
        if region == "bubble":
            z = float(rng.uniform(geometry.bubble_base_z, geometry.bubble_top_z))
            inside = bubble_r2_from_uvz(cell, u, v, z, geometry) <= radius * radius
            if not inside:
                continue
        else:
            z = float(rng.uniform(geometry.solution_z_min, geometry.solution_z_max))
            clearance_radius = radius + float(args.bubble_clearance)
            inside_clearance = bubble_r2_from_uvz(cell, u, v, z, geometry) <= clearance_radius * clearance_radius
            if inside_clearance and z >= geometry.bubble_base_z - args.atom_z_slack:
                continue
        return u, v, z, point_from_uvz(cell, u, v, z)
    raise RuntimeError(f"Failed to sample a candidate center for {region}")


def min_distance(points_a: np.ndarray, points_b: np.ndarray, cell: np.ndarray, pbc: Sequence[bool]) -> float:
    if points_a.size == 0 or points_b.size == 0:
        return math.inf
    _, d = get_distances(points_a, points_b, cell=cell, pbc=np.array(pbc, dtype=bool))
    return float(np.min(d))


class SpatialDistanceGrid:
    """Small cell-list index for exact local distance checks with xy PBC."""

    def __init__(self, cell: np.ndarray, cutoff: float) -> None:
        self.cell = np.array(cell, dtype=float)
        self.inv_cell = np.linalg.inv(self.cell)
        self.cutoff = max(float(cutoff), 1.0e-6)
        lengths = [float(np.linalg.norm(self.cell[i])) for i in range(3)]
        lengths[2] = abs(float(self.cell[2, 2])) if abs(float(self.cell[2, 2])) > 1.0e-12 else lengths[2]
        self.dims = tuple(max(1, int(length / self.cutoff)) for length in lengths)
        self.neighbor_range = (2, 2, 2)
        self.bins: Dict[Tuple[int, int, int], List[np.ndarray]] = {}

    def _scaled(self, positions: np.ndarray) -> np.ndarray:
        scaled = np.asarray(positions, dtype=float) @ self.inv_cell
        scaled[:, 0] = np.mod(scaled[:, 0], 1.0)
        scaled[:, 1] = np.mod(scaled[:, 1], 1.0)
        return scaled

    def _key_from_scaled_one(self, scaled: np.ndarray) -> Tuple[int, int, int]:
        nu, nv, nw = self.dims
        iu = int(math.floor(float(scaled[0]) * nu)) % nu
        iv = int(math.floor(float(scaled[1]) * nv)) % nv
        iz = int(math.floor(float(scaled[2]) * nw))
        iz = min(max(iz, 0), nw - 1)
        return iu, iv, iz

    def add_positions(self, positions: np.ndarray) -> None:
        if positions.size == 0:
            return
        scaled = self._scaled(np.asarray(positions, dtype=float))
        for pos, sc in zip(np.asarray(positions, dtype=float), scaled):
            key = self._key_from_scaled_one(sc)
            self.bins.setdefault(key, []).append(np.array(pos, dtype=float))

    def _candidate_positions(self, points: np.ndarray) -> np.ndarray:
        if not self.bins:
            return np.empty((0, 3), dtype=float)
        nu, nv, nw = self.dims
        ru, rv, rw = self.neighbor_range
        scaled = self._scaled(np.asarray(points, dtype=float))
        candidates: List[np.ndarray] = []
        seen: set[int] = set()
        for sc in scaled:
            iu, iv, iz = self._key_from_scaled_one(sc)
            for du in range(-ru, ru + 1):
                ku = (iu + du) % nu
                for dv in range(-rv, rv + 1):
                    kv = (iv + dv) % nv
                    for dz in range(-rw, rw + 1):
                        kz = iz + dz
                        if kz < 0 or kz >= nw:
                            continue
                        bucket = self.bins.get((ku, kv, kz), [])
                        for arr in bucket:
                            ident = id(arr)
                            if ident not in seen:
                                seen.add(ident)
                                candidates.append(arr)
        if not candidates:
            return np.empty((0, 3), dtype=float)
        return np.vstack(candidates)

    def min_distance(self, points: np.ndarray) -> float:
        pts = np.asarray(points, dtype=float)
        candidates = self._candidate_positions(pts)
        if candidates.size == 0:
            return math.inf
        min_d2 = math.inf
        a_vec = self.cell[0]
        b_vec = self.cell[1]
        for ia in (-1, 0, 1):
            for ib in (-1, 0, 1):
                shifted = candidates + ia * a_vec + ib * b_vec
                diff = pts[:, None, :] - shifted[None, :, :]
                local = float(np.min(np.sum(diff * diff, axis=2)))
                if local < min_d2:
                    min_d2 = local
        return math.sqrt(min_d2)


def region_ok_for_trial(
    trial_pos: np.ndarray,
    u: float,
    v: float,
    z: float,
    cell: np.ndarray,
    geometry: BuildGeometry,
    args: argparse.Namespace,
    region: str,
) -> bool:
    if not atoms_inside_xy_buffer(trial_pos, cell, args.buffer_xy):
        return False

    z_min = float(np.min(trial_pos[:, 2]))
    z_max = float(np.max(trial_pos[:, 2]))
    slack = float(args.atom_z_slack)
    if region == "bubble":
        if z_min < geometry.bubble_base_z - slack or z_max > geometry.bubble_top_z + slack:
            return False
        return bubble_r2_from_uvz(cell, u, v, z, geometry) <= geometry.bubble_radius * geometry.bubble_radius

    if z_min < geometry.solution_z_min - slack or z_max > geometry.solution_z_max + slack:
        return False
    clearance_radius = geometry.bubble_radius + float(args.bubble_clearance)
    return bubble_r2_from_uvz(cell, u, v, z, geometry) > clearance_radius * clearance_radius


def place_species(
    slab: Atoms,
    templates: Dict[str, SpeciesTemplate],
    geometry: BuildGeometry,
    args: argparse.Namespace,
) -> Tuple[List[PlacementRecord], Dict[str, int], Dict[str, int], int, int]:
    rng = np.random.default_rng(int(args.seed))
    cell = np.array(slab.cell.array, dtype=float)
    slab_positions = slab.get_positions()
    slab_grid = SpatialDistanceGrid(cell, args.min_slab_distance)
    if len(slab) > 0:
        slab_grid.add_positions(slab_positions)
    inserted_grid = SpatialDistanceGrid(cell, args.min_inserted_distance)

    queue: List[SpeciesTemplate] = []
    for name in PLACEMENT_ORDER:
        template = templates.get(name)
        if template is not None:
            queue.extend([template] * template.requested)

    placed: List[PlacementRecord] = []
    inserted_counts: Dict[str, int] = Counter()
    failed_counts: Dict[str, int] = Counter()
    attempts = 0
    retries = 0

    for template in queue:
        region = "bubble" if template.name == "N2" else "solution"
        placed_this = False
        centered = template.atoms_centered

        for _ in range(int(args.max_attempts_per_object)):
            attempts += 1
            try:
                u, v, z, center = sample_candidate_center(rng, cell, geometry, args, region)
            except RuntimeError:
                retries += 1
                continue

            positions = centered.get_positions()
            if len(centered) > 1:
                trial_pos = positions @ random_rotation_matrix(rng).T + center
            else:
                trial_pos = positions + center

            if not region_ok_for_trial(trial_pos, u, v, z, cell, geometry, args, region):
                retries += 1
                continue

            d_slab = slab_grid.min_distance(trial_pos) if len(slab) > 0 else math.inf
            if d_slab < float(args.min_slab_distance):
                retries += 1
                continue

            d_inserted = inserted_grid.min_distance(trial_pos)
            if d_inserted < float(args.min_inserted_distance):
                retries += 1
                continue

            atoms = centered.copy()
            atoms.set_positions(trial_pos)
            atoms.set_cell(slab.cell)
            atoms.set_pbc(slab.get_pbc())
            placed.append(PlacementRecord(species_name=template.name, atoms=atoms))
            inserted_counts[template.name] += 1
            inserted_grid.add_positions(trial_pos)
            placed_this = True
            break

        if not placed_this:
            failed_counts[template.name] += 1

    return placed, dict(inserted_counts), dict(failed_counts), attempts, retries


def combine_system(slab: Atoms, placed: Iterable[PlacementRecord], counts: Dict[str, int]) -> Atoms:
    combined = Atoms(
        symbols=slab.get_chemical_symbols(),
        positions=slab.get_positions().copy(),
        cell=slab.cell,
        pbc=slab.get_pbc(),
    )
    for record in placed:
        combined += record.atoms
    combined.info["builder"] = "nnpgen.geo.sio2_nanobubble"
    combined.info["requested_counts"] = json.dumps(counts, sort_keys=True)
    return combined


def packmol_xyz_path(path: Path) -> str:
    return str(path.resolve())


def write_plain_xyz(path: Path, atoms: Atoms) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write(str(path), atoms, format="xyz")


def packmol_box_bounds(slab: Atoms, geometry: BuildGeometry, args: argparse.Namespace) -> Tuple[float, float, float, float, float, float]:
    cell = np.array(slab.cell.array, dtype=float)
    xlo = float(args.buffer_xy)
    ylo = float(args.buffer_xy)
    xhi = float(np.linalg.norm(cell[0])) - float(args.buffer_xy)
    yhi = float(np.linalg.norm(cell[1])) - float(args.buffer_xy)
    return xlo, ylo, float(geometry.solution_z_min), xhi, yhi, float(geometry.solution_z_max)


def run_packmol_placement(
    slab: Atoms,
    templates: Dict[str, SpeciesTemplate],
    geometry: BuildGeometry,
    counts: Dict[str, int],
    args: argparse.Namespace,
) -> Tuple[Atoms, Dict[str, int], Dict[str, int], int, int]:
    cell = np.array(slab.cell.array, dtype=float)
    if abs(float(cell[0, 1])) > CELL_ZERO_TOLERANCE or abs(float(cell[1, 0])) > CELL_ZERO_TOLERANCE:
        raise ValueError("--placement-backend packmol requires an orthogonal xy cell; use --orthorhombic-hex120-supercell")
    if abs(float(cell[0, 2])) > CELL_ZERO_TOLERANCE or abs(float(cell[1, 2])) > CELL_ZERO_TOLERANCE or abs(float(cell[2, 0])) > CELL_ZERO_TOLERANCE or abs(float(cell[2, 1])) > CELL_ZERO_TOLERANCE:
        raise ValueError("--placement-backend packmol requires an axis-aligned cell")

    out_dir = args.output_dir.resolve()
    scratch = out_dir / f"{args.output_prefix}.packmol_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    output_xyz = scratch / f"{args.output_prefix}.packmol.xyz"
    input_path = scratch / "packmol.in"
    stdout_path = scratch / "packmol.out"
    slab_path = scratch / "slab.xyz"

    if len(slab) > 0:
        write_plain_xyz(slab_path, slab)

    xlo, ylo, zlo, xhi, yhi, zhi = packmol_box_bounds(slab, geometry, args)
    bubble_center = point_from_uvz(cell, geometry.bubble_center_u, geometry.bubble_center_v, geometry.bubble_center_z)
    tolerance = float(args.packmol_tolerance if args.packmol_tolerance is not None else args.min_inserted_distance)
    clearance_radius = float(geometry.bubble_radius) + float(args.bubble_clearance)

    lines = [
        f"tolerance {tolerance:.6f}",
        "filetype xyz",
        f"output {packmol_xyz_path(output_xyz)}",
        f"seed {int(args.seed)}",
        "",
    ]
    if len(slab) > 0:
        lines.extend(
            [
                "structure " + packmol_xyz_path(slab_path),
                "  number 1",
                "  fixed 0. 0. 0. 0. 0. 0.",
                "end structure",
                "",
            ]
        )

    for species_name in PLACEMENT_ORDER:
        count = int(counts.get(species_name, 0))
        if count <= 0:
            continue
        template = templates[species_name]
        lines.append("structure " + packmol_xyz_path(template.path))
        lines.append(f"  number {count}")
        if species_name == "N2":
            lines.append(
                "  inside sphere "
                + f"{bubble_center[0]:.6f} {bubble_center[1]:.6f} {bubble_center[2]:.6f} {geometry.bubble_radius:.6f}"
            )
            lines.append(
                "  inside box "
                + f"{xlo:.6f} {ylo:.6f} {geometry.bubble_base_z:.6f} {xhi:.6f} {yhi:.6f} {geometry.bubble_top_z:.6f}"
            )
        else:
            lines.append(f"  inside box {xlo:.6f} {ylo:.6f} {zlo:.6f} {xhi:.6f} {yhi:.6f} {zhi:.6f}")
            lines.append(
                "  outside sphere "
                + f"{bubble_center[0]:.6f} {bubble_center[1]:.6f} {bubble_center[2]:.6f} {clearance_radius:.6f}"
            )
        lines.append("end structure")
        lines.append("")

    input_path.write_text("\n".join(lines), encoding="utf-8")
    with input_path.open("r", encoding="utf-8") as stdin, stdout_path.open("w", encoding="utf-8") as stdout:
        result = subprocess.run([str(args.packmol_bin)], stdin=stdin, stdout=stdout, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Packmol failed with return code {result.returncode}; see {stdout_path}")
    if not output_xyz.exists():
        raise RuntimeError(f"Packmol did not write expected output: {output_xyz}")

    combined = read_first_frame(output_xyz)
    combined.set_cell(slab.cell)
    combined.set_pbc(slab.get_pbc())
    combined.info["builder"] = "nnpgen.geo.sio2_nanobubble"
    combined.info["placement_backend"] = "packmol"
    combined.info["requested_counts"] = json.dumps(counts, sort_keys=True)
    inserted_counts = {k: int(v) for k, v in counts.items() if int(v) > 0}
    return combined, inserted_counts, {}, 0, 0


def count_elements(atoms: Atoms) -> Dict[str, int]:
    return dict(Counter(atoms.get_chemical_symbols()))


def json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def lammps_specorder(symbols: Iterable[str]) -> List[str]:
    present = set(symbols)
    order = list(DEFAULT_ELEMENT_ORDER)
    order.extend(sorted(present.difference(order)))
    return order


def write_outputs(
    combined: Optional[Atoms],
    slab: Atoms,
    geometry: BuildGeometry,
    counts: Dict[str, int],
    inserted_counts: Dict[str, int],
    failed_counts: Dict[str, int],
    attempts: int,
    retries: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix

    extxyz_path = out_dir / f"{prefix}.xyz"
    poscar_path = out_dir / f"{prefix}.POSCAR"
    data_path = out_dir / f"{prefix}.atomic.data"
    summary_path = out_dir / f"{prefix}.summary.json"
    validation_path = out_dir / f"{prefix}.validation.csv"
    specorder = lammps_specorder(combined.get_chemical_symbols() if combined is not None else slab.get_chemical_symbols())
    atom_type_order = {element: i + 1 for i, element in enumerate(specorder)}

    lammps_status = "skipped"
    if combined is not None:
        write(str(extxyz_path), combined, format="extxyz")
        write(str(poscar_path), combined, format="vasp", vasp5=True, sort=False, direct=False)
        if not args.skip_lammps_data:
            try:
                write(
                    str(data_path),
                    combined,
                    format="lammps-data",
                    atom_style="atomic",
                    masses=True,
                    specorder=specorder,
                )
                lammps_status = "written"
            except Exception as exc:
                lammps_status = f"failed: {exc!r}"

    requested_total_atoms = len(slab)
    for name, count in counts.items():
        if name in {"H2O", "H3O"}:
            requested_total_atoms += 3 * count if name == "H2O" else 4 * count
        elif name in {"N2", "OH"}:
            requested_total_atoms += 2 * count
        else:
            requested_total_atoms += count

    estimate_only = bool(args.estimate_only)
    success = True if estimate_only else (
        combined is not None and not failed_counts and inserted_counts == {k: v for k, v in counts.items() if v > 0}
    )
    summary: Dict[str, object] = {
        "source_slab": str(args.source_slab.resolve()),
        "output_dir": str(out_dir),
        "estimate_only": estimate_only,
        "repeat": {"a": int(args.repeat_a), "b": int(args.repeat_b), "c": 1},
        "slab_atoms": len(slab),
        "requested_counts": counts,
        "inserted_counts": inserted_counts,
        "failed_counts": failed_counts,
        "requested_total_atoms": requested_total_atoms,
        "final_total_atoms": len(combined) if combined is not None else None,
        "cell_lengths": [float(x) for x in slab.cell.lengths()],
        "cell_angles": [float(x) for x in slab.cell.angles()],
        "cell": [[float(v) for v in row] for row in slab.cell.array.tolist()],
        "geometry": geometry.__dict__,
        "placement_backend": args.placement_backend,
        "packmol_tolerance": args.packmol_tolerance if args.packmol_tolerance is not None else args.min_inserted_distance,
        "lammps_atom_type_order": atom_type_order,
        "attempts": int(attempts),
        "retries": int(retries),
        "success": success,
        "outputs": {
            "extxyz": str(extxyz_path) if combined is not None else "",
            "poscar": str(poscar_path) if combined is not None else "",
            "lammps_data": str(data_path) if combined is not None and lammps_status == "written" else "",
            "summary": str(summary_path),
            "validation_csv": str(validation_path),
        },
        "lammps_data_status": lammps_status,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")

    with validation_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "output_prefix",
            "success",
            "slab_atoms",
            "requested_total_atoms",
            "final_total_atoms",
            "requested_counts",
            "inserted_counts",
            "failed_counts",
            "attempts",
            "retries",
            "lammps_data_status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "output_prefix": prefix,
                "success": summary["success"],
                "slab_atoms": len(slab),
                "requested_total_atoms": requested_total_atoms,
                "final_total_atoms": summary["final_total_atoms"],
                "requested_counts": json.dumps(counts, sort_keys=True),
                "inserted_counts": json.dumps(inserted_counts, sort_keys=True),
                "failed_counts": json.dumps(failed_counts, sort_keys=True),
                "attempts": int(attempts),
                "retries": int(retries),
                "lammps_data_status": lammps_status,
            }
        )
    return summary


def shell_join(items: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(x)) for x in items)


def write_run_scripts(args: argparse.Namespace, argv: Sequence[str]) -> None:
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.run_script_dir.resolve() if args.run_script_dir is not None else out_dir
    slurm_dir = args.slurm_script_dir.resolve() if args.slurm_script_dir is not None else out_dir
    log_dir = args.log_dir.resolve() if args.log_dir is not None else out_dir
    for directory in (run_dir, slurm_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    run_path = run_dir / f"run_{args.output_prefix}.sh"
    sbatch_path = slurm_dir / f"submit_{args.output_prefix}.sbatch"
    repo_root = args.repo_root.resolve()
    command = shell_join(["python", "-m", "nnpgen", "geo", "build-sio2-nanobubble", *argv])
    log_path = log_dir / f"{args.output_prefix}.build.log"
    slurm_log_path = log_dir / f"{args.output_prefix}.slurm-%j.out"
    run_text = f"""#!/usr/bin/env bash
set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
  fi
fi

module load conda
conda activate {shlex.quote(str(args.conda_env))}
export PYTHONPATH={shlex.quote(str(repo_root / "src"))}:${{PYTHONPATH:-}}
cd {shlex.quote(str(out_dir))}

{command} 2>&1 | tee {shlex.quote(str(log_path))}
"""
    sbatch_text = f"""#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --job-name={args.output_prefix[:32]}
#SBATCH --output={slurm_log_path}

cd {shlex.quote(str(out_dir))}
bash {shlex.quote(str(run_path))}
"""
    run_path.write_text(run_text, encoding="utf-8")
    sbatch_path.write_text(sbatch_text, encoding="utf-8")
    os.chmod(run_path, 0o755)
    os.chmod(sbatch_path, 0o755)


def run_sio2_nanobubble(args: argparse.Namespace, argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    slab = prepare_slab(args)
    geometry = build_geometry(slab, args)
    counts = resolve_counts(args, geometry)

    if args.estimate_only:
        summary = write_outputs(
            combined=None,
            slab=slab,
            geometry=geometry,
            counts=counts,
            inserted_counts={},
            failed_counts={},
            attempts=0,
            retries=0,
            args=args,
        )
    else:
        templates = load_templates(args.template_dir.resolve(), counts)
        if args.placement_backend == "packmol":
            combined, inserted_counts, failed_counts, attempts, retries = run_packmol_placement(
                slab=slab,
                templates=templates,
                geometry=geometry,
                counts=counts,
                args=args,
            )
        else:
            placed, inserted_counts, failed_counts, attempts, retries = place_species(slab, templates, geometry, args)
            combined = combine_system(slab, placed, counts)
        summary = write_outputs(
            combined=combined,
            slab=slab,
            geometry=geometry,
            counts=counts,
            inserted_counts=inserted_counts,
            failed_counts=failed_counts,
            attempts=attempts,
            retries=retries,
            args=args,
        )

    if args.write_run_scripts:
        write_run_scripts(args, list(argv or []))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    run_sio2_nanobubble(args, raw_argv)


if __name__ == "__main__":
    main()
