import json
import subprocess
from types import SimpleNamespace

import pytest

from nnpgen.dft import archive


def test_parse_remote_spec_requires_generic_named_endpoint():
    target = archive.parse_remote_spec("primary=login.example:/work/dft")
    assert target.name == "primary"
    assert target.host == "login.example"
    assert target.root == "/work/dft"


def test_parse_scan_rows_preserves_terminal_markers():
    raw = "system_001\tframe_000001\tsuccess\tfinished\t1\t0\t1\t1\t1\t1\t1\t0\tsystem_001/frame_000001\n"
    rows = archive.parse_scan_rows(raw, "primary")
    row = rows[("system_001", "frame_000001")]
    assert row["result"] == "success"
    assert row["tag_finished"] is True
    assert row["poscar"] is True
    assert row["ediff_reached"] is True
    assert row["normal_footer"] is True
    assert row["task_info"] is True
    assert row["target"] == "primary"


def test_remote_scan_requires_finish_tag_ediff_and_normal_footer(tmp_path):
    frame = tmp_path / "system_001" / "frame_000001"
    frame.mkdir(parents=True)
    (frame / "POSCAR").write_text("placeholder\n")
    (frame / "tag_finished").touch()
    (frame / "OUTCAR").write_text(
        "aborting loop because EDIFF is reached\n"
        "General timing and accounting informations for this job:\n"
    )
    result = subprocess.run(
        ["bash", "-s", "--", str(tmp_path)],
        input=archive.REMOTE_SCAN_SCRIPT,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    assert archive.parse_scan_rows(result.stdout)[("system_001", "frame_000001")]["result"] == "success"

    (frame / "OUTCAR").write_text("aborting loop because EDIFF is reached\n")
    result = subprocess.run(
        ["bash", "-s", "--", str(tmp_path)],
        input=archive.REMOTE_SCAN_SCRIPT,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    assert archive.parse_scan_rows(result.stdout)[("system_001", "frame_000001")]["result"] == "pending"


def test_archive_manifest_stages_only_complete_system(monkeypatch, tmp_path):
    target = archive.RemoteTarget("primary", "login.example", "/work/dft")
    rows = {
        ("system_001", "frame_000001"): {
            "system": "system_001", "frame": "frame_000001", "result": "success",
            "status": "finished", "tag_finished": True, "tag_failed": False,
            "poscar": True, "outcar": True, "ediff_reached": True, "normal_footer": True,
            "task_info": True, "frame_info": False,
            "relative": "system_001/frame_000001", "target": "primary",
        },
        ("system_001", "frame_000002"): {
            "system": "system_001", "frame": "frame_000002", "result": "pending",
            "status": "running", "tag_finished": False, "tag_failed": False,
            "poscar": True, "outcar": False, "task_info": True, "frame_info": False,
            "relative": "system_001/frame_000002", "target": "primary",
        },
    }
    monkeypatch.setattr(archive, "scan_target", lambda _: rows)
    manifest = {"kind": "dft", "entries": [
        {"system_name": "system_001", "frame_name": "frame_000001", "remote_target": "primary"},
        {"system_name": "system_001", "frame_name": "frame_000002", "remote_target": "primary"},
    ]}
    report = archive.archive_manifest(manifest, [target], tmp_path / "archive", dry_run=True)
    assert report["systems"]["system_001"]["status"] == "partial"
    assert report["summary"]["pending_frames"] == 1


def test_archive_rejects_converged_duplicates_across_targets(monkeypatch, tmp_path):
    targets = [
        archive.RemoteTarget("a", "a.example", "/work/a"),
        archive.RemoteTarget("b", "b.example", "/work/b"),
    ]
    row = {
        "system": "system_001", "frame": "frame_000001", "result": "success",
        "status": "finished", "tag_finished": True, "tag_failed": False,
        "poscar": True, "outcar": True, "ediff_reached": True, "normal_footer": True,
        "task_info": True, "frame_info": False, "relative": "system_001/frame_000001",
    }
    monkeypatch.setattr(
        archive,
        "scan_target",
        lambda target: {("system_001", "frame_000001"): {**row, "target": target.name}},
    )
    manifest = {"entries": [{"system": "system_001", "frame": "frame_000001", "remote_target": "a"}]}

    try:
        archive.archive_manifest(manifest, targets, tmp_path, dry_run=True)
    except ValueError as exc:
        assert "multiple targets" in str(exc)
    else:
        raise AssertionError("duplicate converged task was accepted")


def test_dataset_install_creates_backup_and_rolls_back_on_post_install_failure(monkeypatch, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "old.txt").write_text("old")

    def fake_converter(command, **_):
        output = command[command.index("--output-dir") + 1]
        output_path = archive.Path(output)
        (output_path / "new.txt").write_text("new")
        return SimpleNamespace(stdout="converted")

    monkeypatch.setattr(archive.subprocess, "run", fake_converter)
    calls = []

    def validate(path):
        calls.append(archive.Path(path))
        if len(calls) == 2:
            raise ValueError("post-install check failed")
        return {"valid": True}

    monkeypatch.setattr(archive, "require_valid_dp_dataset", validate)
    with pytest.raises(ValueError, match="post-install"):
        archive.rebuild_dp_dataset(work, dataset)

    assert (dataset / "old.txt").read_text() == "old"
    assert not (dataset / "new.txt").exists()
