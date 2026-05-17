from pathlib import Path
from typing import List

from ase import Atoms
from ase.io import iread, read, write


def read_frames(xyz_path: Path) -> List[Atoms]:
    frames = list(iread(str(xyz_path), index=":"))
    if frames:
        return frames

    single = read(str(xyz_path), index=0)
    return [single]


def write_frame_xyz(atoms: Atoms, out_path: Path) -> None:
    write(str(out_path), atoms, format="extxyz")


def write_frames_xyz(frames: List[Atoms], out_path: Path) -> None:
    if out_path.exists():
        out_path.unlink()
    for i, atoms in enumerate(frames):
        write(str(out_path), atoms, format="extxyz", append=(i > 0))
