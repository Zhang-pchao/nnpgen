import argparse
import sys
import types

import numpy as np

from nnpgen.train.predict_poscar import run_predict_poscar


def test_prediction_uses_model_types_but_writes_requested_output_types(monkeypatch, tmp_path):
    frame = tmp_path / "input" / "system_001" / "frame_000001"
    frame.mkdir(parents=True)
    (frame / "POSCAR").write_text(
        """H O
1.0
1 0 0
0 1 0
0 0 1
H O
1 1
Cartesian
0 0 0
0.5 0.5 0.5
"""
    )
    model = tmp_path / "model.pb"
    model.touch()

    class FakeDeepPot:
        def __init__(self, _):
            pass

        def get_type_map(self):
            return ["O", "H"]

        def eval(self, coord, box, atom_types):
            assert atom_types.tolist() == [1, 0]
            return np.array([[-1.0]]), np.zeros((1, 2, 3)), np.zeros((1, 9))

    infer = types.ModuleType("deepmd.infer")
    infer.DeepPot = FakeDeepPot
    deepmd = types.ModuleType("deepmd")
    deepmd.infer = infer
    monkeypatch.setitem(sys.modules, "deepmd", deepmd)
    monkeypatch.setitem(sys.modules, "deepmd.infer", infer)

    output = tmp_path / "output"
    summary = run_predict_poscar(
        argparse.Namespace(
            input_root=str(tmp_path / "input"),
            output_dir=str(output),
            model=str(model),
            type_map="H,O",
            max_frames=0,
            batch_size=8,
            skip_dft_selected=False,
            dft_manifest_glob="",
            dft_skip_statuses="",
            remote_host="",
            dft_run_roots="",
            fallback_remote_scan=False,
            exclude_list="",
        )
    )

    assert summary["model_type_map"] == ["O", "H"]
    assert summary["output_type_map"] == ["H", "O"]
    assert (output / "type.raw").read_text().split() == ["0", "1"]
