import json

from nnpgen.monitor.summary import count_xyz_frames, summarize_manifest, summarize_run_root


def test_monitor_summary_counts_manifest_and_run_status(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"kind": "dft", "entries": [
        {"status": "finished"}, {"status": "running"}, {"status": "failed"}
    ]}))
    run_root = tmp_path / "run"
    frame = run_root / "system_001" / "frame_000001"
    frame.mkdir(parents=True)
    (frame / "frame_info.json").write_text('{"status": "finished"}')
    (frame / "frame_records.json").write_text('{}')
    assert summarize_manifest(manifest_path)["status_counts"] == {"failed": 1, "finished": 1, "running": 1}
    assert summarize_run_root(run_root)["status_counts"] == {"finished": 1}


def test_count_xyz_frames_ignores_incomplete_tail(tmp_path):
    path = tmp_path / "frames.xyz"
    path.write_text("1\nframe=1\nH 0 0 0\n1\nframe=2\nH 1 0 0\n2\npartial\nH 0 0 0\n")
    assert count_xyz_frames(path) == 2
