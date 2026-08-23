import math
import sys
import types
from argparse import Namespace

import numpy as np

from nnpgen.dataset.manifest import write_manifest
from nnpgen.train.benchmark_predict import ErrorAccumulator, frames_per_chunk, run
from nnpgen.train.deepmd_test import parse_dp_test_log, run as run_dp_test


def test_dp_test_parser_uses_all_mixed_nloc_frames_and_weighted_metrics():
    log = """
[time] DEEPMD INFO # mixed-nloc LMDB: testing 2 groups: {80: 2, 100: 3}
[time] DEEPMD INFO # testing sub-group : data [nloc=80]
[time] DEEPMD INFO # number of test data : 2
[time] DEEPMD INFO Energy RMSE/Natoms : 9.0e-03 eV
[time] DEEPMD INFO # testing sub-group : data [nloc=100]
[time] DEEPMD INFO # number of test data : 3
[time] DEEPMD INFO Energy RMSE/Natoms : 8.0e-03 eV
[time] DEEPMD INFO # ----------weighted average of errors-----------
[time] DEEPMD INFO # number of systems : 1
[time] DEEPMD INFO Energy RMSE/Natoms : 2.500000e-03 eV
[time] DEEPMD INFO Force  RMSE        : 9.400000e-02 eV/A
[time] DEEPMD INFO # -----------------------------------------------
"""
    parsed = parse_dp_test_log(log)
    assert parsed["frames_tested"] == 5
    assert parsed["mixed_nloc_group_count"] == 2
    assert parsed["system_count"] == 1
    assert parsed["energy_rmse_meV_per_atom"] == 2.5
    assert parsed["force_rmse_meV_per_A"] == 94.0


def test_dp_test_wrapper_runs_native_command_and_writes_audit(tmp_path):
    executable = tmp_path / "fake-dp"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if '--version' in sys.argv:\n"
        "    print('fake-deepmd 1.0')\n"
        "else:\n"
        "    print('# number of test data : 2')\n"
        "    print('# ----------weighted average of errors-----------')\n"
        "    print('# number of systems : 1')\n"
        "    print('Energy RMSE/Natoms : 1.5e-03 eV')\n"
        "    print('Force RMSE : 8.0e-02 eV/A')\n"
    )
    executable.chmod(0o755)
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    system = tmp_path / "validation.lmdb"
    system.mkdir()
    (system / "data.mdb").write_bytes(b"data")
    output = tmp_path / "native-report"
    summary = run_dp_test(
        Namespace(
            model=model,
            system=system,
            output_dir=output,
            numb_test=0,
            chunk_atoms=1234,
            dp_command=str(executable),
            pt_expt=True,
            detail_file=None,
            audit_input=[],
        )
    )
    assert summary["status"] == "PASS"
    assert summary["frames_tested"] == 2
    assert summary["all_frames_requested"] is True
    assert summary["environment"]["DP_TEST_CHUNK_ATOMS"] == "1234"
    assert summary["input_sha256"][str(system / "data.mdb")]
    assert (output / "dp-test.log").is_file()
    assert (output / "summary.json").is_file()


def test_error_accumulator_merges_sse_and_counts_instead_of_averaging_rmse():
    first = ErrorAccumulator(systems=1)
    first.update(
        energy_true=np.zeros(1),
        energy_pred=np.asarray([2.0]),
        natoms=2,
        force_true=np.zeros(6),
        force_pred=np.ones(6),
    )
    second = ErrorAccumulator(systems=1)
    second.update(
        energy_true=np.zeros(3),
        energy_pred=np.full(3, 4.0),
        natoms=4,
        force_true=np.zeros(36),
        force_pred=np.full(36, 2.0),
    )
    combined = ErrorAccumulator()
    combined.merge(first)
    combined.merge(second)
    metrics = combined.metrics()
    assert combined.systems == 2
    assert combined.frames == 4
    assert metrics["energy_rmse_meV_per_atom"] == 1000.0
    assert math.isclose(metrics["force_rmse_eV_per_A"], math.sqrt((6 + 36 * 4) / 42))


def test_atom_budget_chunking_adapts_to_system_size_and_frame_cap():
    assert frames_per_chunk(100, 20000) == 200
    assert frames_per_chunk(600, 20000) == 33
    assert frames_per_chunk(25000, 20000) == 1
    assert frames_per_chunk(100, 20000, batch_size=8) == 8


def _write_system(path, natoms, frames):
    path.mkdir(parents=True)
    (path / "type_map.raw").write_text("H\nO\n")
    (path / "type.raw").write_text("\n".join("0" if index % 2 == 0 else "1" for index in range(natoms)) + "\n")
    set_dir = path / "set.000"
    set_dir.mkdir()
    np.save(set_dir / "coord.npy", np.zeros((frames, natoms, 3)))
    np.save(set_dir / "box.npy", np.tile(np.eye(3), (frames, 1, 1)))
    np.save(set_dir / "energy.npy", np.zeros(frames))
    np.save(set_dir / "force.npy", np.zeros((frames, natoms, 3)))


def test_benchmark_writes_streaming_system_and_family_reports(tmp_path, monkeypatch):
    first = tmp_path / "data" / "first"
    second = tmp_path / "data" / "second"
    _write_system(first, natoms=2, frames=3)
    _write_system(second, natoms=4, frames=2)
    manifest = tmp_path / "systems.tsv"
    rows = [
        {"index": "0", "system_path": str(first), "block": "water", "label": "reference"},
        {"index": "1", "system_path": str(second), "block": "oxide", "label": "reference"},
    ]
    write_manifest(manifest, rows)
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")

    class FakeDeepPot:
        def __init__(self, path):
            assert path == str(model)

        def get_type_map(self):
            return ["H", "O"]

        def eval(self, coord, box, atom_types):
            frames = len(coord)
            natoms = len(atom_types)
            return np.ones((frames, 1)), np.ones((frames, natoms, 3)), np.zeros((frames, 9))

    deepmd = types.ModuleType("deepmd")
    deepmd.__version__ = "test"
    infer = types.ModuleType("deepmd.infer")
    infer.DeepPot = FakeDeepPot
    monkeypatch.setitem(sys.modules, "deepmd", deepmd)
    monkeypatch.setitem(sys.modules, "deepmd.infer", infer)
    output = tmp_path / "report"
    summary = run(
        Namespace(
            dataset_root=None,
            manifest=manifest,
            model=model,
            output_root=output,
            model_label="example",
            chunk_atoms=8,
            batch_size=0,
            group_by="block",
            max_datasets=0,
            max_frames_per_dataset=0,
            model_type_map="",
            save_arrays=False,
            continue_on_error=False,
        )
    )
    assert summary["status"] == "PASS"
    assert summary["systems_evaluated"] == 2
    assert summary["frames_total"] == 5
    assert summary["metrics"]["energy_rmse_meV_per_atom"] == math.sqrt((3 * 0.5**2 + 2 * 0.25**2) / 5) * 1000
    assert (output / "per_system.csv").read_text().count("\n") == 3
    assert (output / "per_family.csv").read_text().count("\n") == 3
    assert not (output / "benchmark_arrays.npz").exists()
