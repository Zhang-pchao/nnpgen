import argparse

import numpy as np

from nnpgen.dataset.dpdata_to_extxyz import run_dpdata_to_extxyz
from nnpgen.dft.plan import parse_poscar_element_order


def test_parse_poscar_element_order(tmp_path):
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        """SiO2
1.0
1 0 0
0 1 0
0 0 1
Si O
1 2
Cartesian
0 0 0
0.25 0.25 0.25
0.5 0.5 0.5
"""
    )

    assert parse_poscar_element_order(poscar) == ["Si", "O"]


def test_dpdata_to_extxyz_tiny_dataset(tmp_path):
    root = tmp_path / "dp"
    set_dir = root / "set.000"
    set_dir.mkdir(parents=True)
    (root / "type_map.raw").write_text("H\nO\n")
    np.savetxt(root / "type.raw", np.array([0, 1]), fmt="%d")
    np.save(set_dir / "box.npy", np.eye(3).reshape(1, 9))
    np.save(set_dir / "coord.npy", np.array([[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]]))
    np.save(set_dir / "force.npy", np.zeros((1, 6)))
    np.save(set_dir / "energy.npy", np.array([[-1.0]]))
    np.save(set_dir / "virial.npy", np.zeros((1, 9)))

    out = tmp_path / "out.xyz"
    summary = run_dpdata_to_extxyz(argparse.Namespace(dataset_dir=str(root), output_xyz=str(out)))

    assert out.exists()
    assert summary["frames_written"] == 1
    assert "Lattice=" in out.read_text()
