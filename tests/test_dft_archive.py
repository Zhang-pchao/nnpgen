import json

from nnpgen.dft import archive


def test_parse_remote_spec_requires_generic_named_endpoint():
    target = archive.parse_remote_spec("primary=login.example:/work/dft")
    assert target.name == "primary"
    assert target.host == "login.example"
    assert target.root == "/work/dft"


def test_parse_scan_rows_preserves_terminal_markers():
    raw = "system_001\tframe_000001\tsuccess\tfinished\t1\t0\t1\t1\t1\t0\tsystem_001/frame_000001\n"
    rows = archive.parse_scan_rows(raw, "primary")
    row = rows[("system_001", "frame_000001")]
    assert row["result"] == "success"
    assert row["tag_finished"] is True
    assert row["poscar"] is True
    assert row["task_info"] is True
    assert row["target"] == "primary"


def test_archive_manifest_stages_only_complete_system(monkeypatch, tmp_path):
    target = archive.RemoteTarget("primary", "login.example", "/work/dft")
    rows = {
        ("system_001", "frame_000001"): {
            "system": "system_001", "frame": "frame_000001", "result": "success",
            "status": "finished", "tag_finished": True, "tag_failed": False,
            "poscar": True, "outcar": True, "task_info": True, "frame_info": False,
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
    assert report["systems"]["system_001"]["status"] == "skipped_incomplete"
    assert report["summary"]["pending_frames"] == 1
