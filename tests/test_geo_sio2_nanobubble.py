import json
from argparse import Namespace

from ase import Atoms
from ase.io import read, write

from nnpgen.geo.sio2_nanobubble import build_geometry, parse_args, prepare_slab, run_sio2_nanobubble


def write_template(path, atoms):
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, atoms, format="extxyz")


def test_sio2_nanobubble_small_build(tmp_path):
    slab_path = tmp_path / "slab.xyz"
    slab = Atoms(
        symbols=["Si", "O", "O", "H"],
        positions=[
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.8],
            [1.0, 2.0, 2.1],
            [2.0, 2.0, 2.8],
        ],
        cell=[[10.0, 0.0, 0.0], [-3.0, 9.0, 0.0], [0.0, 0.0, 22.0]],
        pbc=[True, True, True],
    )
    write(slab_path, slab, format="extxyz")

    template_dir = tmp_path / "templates"
    write_template(
        template_dir / "H2O.xyz",
        Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.76, 0.58, 0.0], [-0.76, 0.58, 0.0]]),
    )
    write_template(template_dir / "N2.xyz", Atoms("N2", positions=[[-0.55, 0.0, 0.0], [0.55, 0.0, 0.0]]))
    write_template(template_dir / "Na.xyz", Atoms("Na", positions=[[0.0, 0.0, 0.0]]))
    write_template(template_dir / "Cl.xyz", Atoms("Cl", positions=[[0.0, 0.0, 0.0]]))

    out_dir = tmp_path / "out"
    argv = [
        "--source-slab",
        str(slab_path),
        "--output-dir",
        str(out_dir),
        "--output-prefix",
        "small",
        "--template-dir",
        str(template_dir),
        "--repeat-a",
        "1",
        "--repeat-b",
        "1",
        "--surface-gap",
        "2.0",
        "--solution-height",
        "9.0",
        "--bubble-radius",
        "2.4",
        "--h2o-count",
        "3",
        "--n2-count",
        "1",
        "--salt-pairs",
        "1",
        "--min-slab-distance",
        "0.8",
        "--min-inserted-distance",
        "0.8",
        "--max-attempts-per-object",
        "4000",
    ]
    summary = run_sio2_nanobubble(parse_args(argv), argv)

    assert summary["success"] is True
    assert summary["final_total_atoms"] == 4 + 3 * 3 + 2 + 2
    assert summary["lammps_atom_type_order"] == {
        "H": 1,
        "O": 2,
        "N": 3,
        "Na": 4,
        "Cl": 5,
        "Ti": 6,
        "C": 7,
        "Si": 8,
    }
    assert (out_dir / "small.xyz").exists()
    assert (out_dir / "small.POSCAR").exists()
    assert (out_dir / "small.atomic.data").exists()
    assert (out_dir / "small.summary.json").exists()
    assert len(read(out_dir / "small.xyz")) == summary["final_total_atoms"]

    loaded = json.loads((out_dir / "small.summary.json").read_text())
    assert loaded["requested_counts"]["H2O"] == 3
    lammps_text = (out_dir / "small.atomic.data").read_text()
    assert "8 atom types" in lammps_text
    assert "# Ti" in lammps_text
    assert "# C" in lammps_text
    assert "# Si" in lammps_text


def test_orthorhombic_hex120_snaps_roundoff(tmp_path):
    slab_path = tmp_path / "mixed291_like.xyz"
    slab = Atoms(
        symbols=["Si"],
        positions=[[0.0, 0.0, 0.0]],
        cell=[[14.823377, 0.0, 0.0], [-7.411688, 12.837421, 0.0], [0.0, 0.0, 30.886063]],
        pbc=[True, True, True],
    )
    write(slab_path, slab, format="extxyz")

    args = Namespace(
        source_slab=slab_path,
        repeat_a=5,
        repeat_b=3,
        orthorhombic_hex120_supercell=True,
        control_no_slab=False,
        control_bottom_buffer=2.3,
        target_bottom_z=5.0,
        surface_gap=2.0,
        bubble_bottom_gap=None,
        solution_height=88.0,
        top_buffer=50.0,
        bubble_radius=21.0,
        bubble_shape="sphere",
    )
    repeated = prepare_slab(args)

    assert repeated.cell[0, 1] == 0.0
    assert repeated.cell[1, 0] == 0.0


def test_sphere_bubble_geometry_uses_surface_gap_as_bottom_clearance():
    slab = Atoms(
        symbols=["Si", "O"],
        positions=[[0.0, 0.0, 5.0], [1.0, 1.0, 10.0]],
        cell=[[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 80.0]],
        pbc=[True, True, True],
    )
    args = Namespace(
        control_bottom_buffer=0.0,
        surface_gap=2.0,
        bubble_bottom_gap=None,
        solution_height=50.0,
        bubble_radius=6.0,
        bubble_shape="sphere",
        bubble_center_u=0.5,
        bubble_center_v=0.5,
    )

    geometry = build_geometry(slab, args)

    assert geometry.solution_z_min == 12.0
    assert geometry.bubble_base_z == 12.0
    assert geometry.bubble_center_z == 18.0
    assert geometry.bubble_top_z == 24.0


def test_sphere_bubble_bottom_gap_can_be_decoupled_from_solution_gap():
    slab = Atoms(
        symbols=["Si", "O"],
        positions=[[0.0, 0.0, 5.0], [1.0, 1.0, 10.0]],
        cell=[[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 120.0]],
        pbc=[True, True, True],
    )
    args = Namespace(
        control_bottom_buffer=0.0,
        surface_gap=2.0,
        bubble_bottom_gap=18.0,
        solution_height=80.0,
        bubble_radius=6.0,
        bubble_shape="sphere",
        bubble_center_u=0.5,
        bubble_center_v=0.5,
    )

    geometry = build_geometry(slab, args)

    assert geometry.solution_z_min == 12.0
    assert geometry.solution_z_max == 92.0
    assert geometry.bubble_base_z == 28.0
    assert geometry.bubble_center_z == 34.0
    assert geometry.bubble_top_z == 40.0
