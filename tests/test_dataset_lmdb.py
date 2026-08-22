import json

import numpy as np
import pytest

pytest.importorskip("lmdb")
pytest.importorskip("msgpack")

from nnpgen.dataset.lmdb import compare_npy_lmdb, convert_npy_to_lmdb, validate_lmdb
from nnpgen.dataset.manifest import write_manifest


def _write_group(root, type_map, types, offset):
    root.mkdir(parents=True)
    (root / "type_map.raw").write_text("\n".join(type_map) + "\n")
    (root / "type.raw").write_text("\n".join(str(value) for value in types) + "\n")
    set_dir = root / "set.000"
    set_dir.mkdir()
    natoms = len(types)
    frames = 2
    coord = (np.arange(frames * natoms * 3, dtype=np.float32).reshape(frames, natoms, 3) + offset)
    box = np.tile(np.eye(3, dtype=np.float32), (frames, 1, 1))
    energy = np.asarray([offset, offset + 1], dtype=np.float64)
    force = np.full((frames, natoms, 3), offset, dtype=np.float32)
    np.save(set_dir / "coord.npy", coord)
    np.save(set_dir / "box.npy", box)
    np.save(set_dir / "energy.npy", energy)
    np.save(set_dir / "force.npy", force)
    return frames


def test_npy_lmdb_round_trip_preserves_remapped_frames(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_group(first, ("O", "H"), (0, 1), 0)
    _write_group(second, ("H", "O"), (1, 0), 10)
    manifest = tmp_path / "systems.tsv"
    write_manifest(
        manifest,
        [
            {"index": "0", "system_path": str(first), "block": "one", "label": "dft"},
            {"index": "1", "system_path": str(second), "block": "two", "label": "distilled"},
        ],
    )
    output = tmp_path / "training.lmdb"
    report_path = tmp_path / "training.json"

    report = convert_npy_to_lmdb(manifest, output, report_path, type_map="H,O")
    assert report["status"] == "PASS"
    assert report["systems"] == 2
    assert report["frames"] == 4
    assert report["type_map"] == ["H", "O"]

    validation = validate_lmdb(output, full=True)
    assert validation["valid"] is True
    comparison = compare_npy_lmdb(manifest, output, atol=0.0)
    assert comparison["valid"] is True
    assert comparison["frames_compared"] == 4
    assert json.loads(report_path.read_text())["schema"].startswith("nnpgen.dpa4c-lmdb")


def test_conversion_refuses_existing_output(tmp_path):
    group = tmp_path / "group"
    _write_group(group, ("H",), (0,), 0)
    manifest = tmp_path / "systems.tsv"
    write_manifest(manifest, [{"index": "0", "system_path": str(group), "block": "x"}])
    output = tmp_path / "existing.lmdb"
    output.mkdir()
    with pytest.raises(FileExistsError):
        convert_npy_to_lmdb(manifest, output, tmp_path / "report.json")
