import json

import numpy as np

from nnpgen.dataset.inspect import run_build_manifest
from nnpgen.dataset.manifest import inspect_group, inspect_roots, read_manifest


def _write_group(root, type_map=("O", "H"), types=(0, 1, 1), frames=2, with_virial=True):
    root.mkdir(parents=True)
    (root / "type_map.raw").write_text("\n".join(type_map) + "\n")
    (root / "type.raw").write_text("\n".join(str(value) for value in types) + "\n")
    set_dir = root / "set.000"
    set_dir.mkdir()
    natoms = len(types)
    np.save(set_dir / "coord.npy", np.arange(frames * natoms * 3, dtype=np.float32).reshape(frames, natoms, 3))
    np.save(set_dir / "box.npy", np.tile(np.eye(3, dtype=np.float32), (frames, 1, 1)))
    np.save(set_dir / "energy.npy", np.arange(frames, dtype=np.float64))
    np.save(set_dir / "force.npy", np.zeros((frames, natoms, 3), dtype=np.float32))
    if with_virial:
        np.save(set_dir / "virial.npy", np.zeros((frames, 9), dtype=np.float32))


def test_inspect_reports_frames_composition_and_nonfinite_values(tmp_path):
    group = tmp_path / "block" / "system_0"
    _write_group(group)

    report = inspect_group(group)
    assert report["frames"] == 2
    assert report["natoms"] == 3
    assert report["elements"] == ["O", "H"]
    assert report["counts"] == [1, 2]
    assert report["virial_frames"] == 2
    assert report["nonfinite"]["coord"] == 0


def test_build_manifest_is_deterministic_and_readable(tmp_path):
    root = tmp_path / "dataset"
    _write_group(root / "b")
    _write_group(root / "a", type_map=("H", "O"), types=(1, 0, 0))
    manifest = tmp_path / "manifests" / "systems.tsv"
    report_path = tmp_path / "reports" / "inspect.json"

    args = type("Args", (), {"root": [str(root)], "output": manifest, "report": report_path, "block": "training", "label": "dft"})()
    report = run_build_manifest(args)
    rows = read_manifest(manifest)
    assert report["valid"] is True
    assert [row["index"] for row in rows] == ["0", "1"]
    assert all(row["block"] == "training" for row in rows)
    assert all(json.loads(row["elements"]) for row in rows)
    assert json.loads(report_path.read_text())["systems"] == 2


def test_inspect_marks_invalid_group_without_overwriting_source(tmp_path):
    root = tmp_path / "dataset" / "system"
    _write_group(root)
    np.save(root / "set.000" / "force.npy", np.zeros((1, 3, 3), dtype=np.float32))

    report = inspect_roots([tmp_path / "dataset"])
    assert report["valid"] is False
    assert report["systems"] == 0
    assert report["errors"]
