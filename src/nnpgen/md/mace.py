from __future__ import annotations

import argparse
from pathlib import Path

from ..config import (
    DEFAULT_DEVICE,
    DEFAULT_FRICTION_PER_FS,
    DEFAULT_INTERVAL,
    DEFAULT_MODEL_PATH,
    DEFAULT_STEPS,
    DEFAULT_TEMPERATURE_K,
    DEFAULT_TIMESTEP_FS,
)


def add_md_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Absolute path to input structure (XYZ)")
    parser.add_argument("--output-xyz", required=True, help="Absolute path to sampled trajectory XYZ")
    parser.add_argument("--final-xyz", required=True, help="Absolute path to final frame XYZ")
    parser.add_argument("--final-poscar", required=True, help="Absolute path to final POSCAR")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument("--timestep-fs", type=float, default=DEFAULT_TIMESTEP_FS)
    parser.add_argument("--friction", type=float, default=DEFAULT_FRICTION_PER_FS, help="Langevin friction in 1/fs")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    parser.add_argument("--device", choices=["gpu", "cpu"], default=DEFAULT_DEVICE)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--seed", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run ASE+MACE NVT MD for one frame")
    add_md_arguments(p)
    return p


def _resolve_mace_device(device: str) -> str:
    if device == "gpu":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("--device gpu requested but CUDA is not available")
        return "cuda"
    return "cpu"


def _validate_wrapped(atoms: Atoms, tol: float = 1e-6) -> None:
    scaled = atoms.get_scaled_positions(wrap=False)
    minv = float(scaled.min())
    maxv = float(scaled.max())
    if minv < -tol or maxv > 1.0 + tol:
        raise ValueError(
            f"Wrapped coordinates out of bounds: scaled min={minv:.6e}, max={maxv:.6e}"
        )


def _wrap_for_output(atoms: Atoms, tol: float = 1e-6) -> Atoms:
    import numpy as np

    atoms.wrap()
    _validate_wrapped(atoms, tol=tol)

    # Normalize tiny floating-point drift (e.g. -1e-8) to [0, 1) without masking real failures.
    scaled = atoms.get_scaled_positions(wrap=False)
    if float(scaled.min()) < 0.0 or float(scaled.max()) >= 1.0:
        atoms.set_scaled_positions(np.mod(scaled, 1.0))

    _validate_wrapped(atoms, tol=tol)
    return atoms


def run_md(args: argparse.Namespace) -> None:
    import numpy as np
    from ase import units
    from ase.io import read, write
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    from ..poscar import atoms_to_poscar

    if args.steps < 0:
        raise ValueError("--steps must be >= 0")
    if args.interval <= 0:
        raise ValueError("--interval must be > 0")

    input_path = Path(args.input).resolve()
    output_xyz = Path(args.output_xyz).resolve()
    final_xyz = Path(args.final_xyz).resolve()
    final_poscar = Path(args.final_poscar).resolve()
    model_path = Path(args.model_path).resolve()

    output_xyz.parent.mkdir(parents=True, exist_ok=True)
    final_xyz.parent.mkdir(parents=True, exist_ok=True)
    final_poscar.parent.mkdir(parents=True, exist_ok=True)

    atoms = read(str(input_path), index=0)
    if atoms.cell is None or atoms.cell.rank < 3:
        raise ValueError("Input structure must include a 3D periodic cell")
    atoms.pbc = [True, True, True]

    mace_device = _resolve_mace_device(args.device)
    from mace.calculators import MACECalculator

    calc = MACECalculator(str(model_path), device=mace_device)
    atoms.calc = calc

    rng = np.random.default_rng(args.seed) if args.seed is not None else None
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=rng)

    dyn = Langevin(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=args.friction / units.fs,
    )

    if output_xyz.exists():
        output_xyz.unlink()

    wrote_any = {"value": False}

    def write_snapshot() -> None:
        step = dyn.get_number_of_steps()
        if step == 0:
            return
        _wrap_for_output(dyn.atoms)
        write(str(output_xyz), dyn.atoms, format="extxyz", append=wrote_any["value"])
        wrote_any["value"] = True

    dyn.attach(write_snapshot, interval=args.interval)

    print(
        f"MD start: steps={args.steps} interval={args.interval} temp={args.temperature}K "
        f"dt={args.timestep_fs}fs friction={args.friction}/fs device={args.device}"
    )
    dyn.run(args.steps)

    if not wrote_any["value"]:
        _wrap_for_output(dyn.atoms)
        write(str(output_xyz), dyn.atoms, format="extxyz")

    _wrap_for_output(dyn.atoms)
    write(str(final_xyz), dyn.atoms, format="extxyz")

    _wrap_for_output(dyn.atoms)
    atoms_to_poscar(dyn.atoms, final_poscar)

    print("MD done")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_md(args)


if __name__ == "__main__":
    main()
