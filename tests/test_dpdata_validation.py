import json

import numpy as np

from nnpgen.dataset.validation import composition_key, validate_dp_dataset


def _write_dataset(root, frames=2):
    set_dir = root / "set.000"
    set_dir.mkdir(parents=True)
    arrays = {
        "energy": np.zeros(frames),
        "box": np.zeros((frames, 9)),
        "coord": np.zeros((frames, 6)),
        "force": np.zeros((frames, 6)),
        "virial": np.zeros((frames, 9)),
    }
    for name, array in arrays.items():
        np.save(set_dir / (name + ".npy"), array)
        np.savetxt(root / (name + ".raw"), array.reshape(frames, -1))
    (root / "type.raw").write_text("0\n1\n")
    (root / "type_map.raw").write_text("H\nO\n")
    summary = {
        "output_dir": str(root),
        "groups": [{
            "output_dir": str(root),
            "elements": ["H", "O"],
            "counts": [1, 1],
            "frames_selected": frames,
            "frames_converted": frames,
            "frames_skipped": 0,
        }],
        "group_count": 1,
        "frames_converted": frames,
        "frames_skipped": 0,
        "frames_skipped_invalid_poscar": 0,
    }
    (root / "convert_summary.json").write_text(json.dumps(summary))


def test_dpdata_validation_checks_all_raw_and_npy_frame_counts(tmp_path):
    root = tmp_path / "dpdata"
    _write_dataset(root)

    report = validate_dp_dataset(root)
    assert report["valid"] is True
    assert report["frames"] == 2
    assert composition_key({"elements": ["H", "O"], "counts": [1, 1]}) == (("H", "O"), (1, 1))

    np.save(root / "set.000" / "force.npy", np.zeros((1, 6)))
    report = validate_dp_dataset(root)
    assert report["valid"] is False
    assert any("force.npy frame count mismatch" in error for error in report["errors"])
