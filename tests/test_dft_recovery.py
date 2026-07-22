from nnpgen.dft.recovery import parse_active_jobs, recover_manifest_entries


def test_parse_active_jobs_supports_pbs_and_slurm():
    pbs = parse_active_jobs("pbs", "123.server job_a user 00:00:00 R queue\n")
    slurm = parse_active_jobs("slurm", "123|job_b|RUNNING\n")
    assert "123.server" in pbs["ids"]
    assert "job_a" in pbs["names"]
    assert "123" in slurm["ids"]
    assert "job_b" in slurm["names"]


def test_recovery_preserves_active_and_terminal_entries():
    manifest = {"entries": [
        {"system_name": "s1", "frame_name": "f1", "remote_target": "main", "status": "running", "job_id": "101"},
        {"system_name": "s1", "frame_name": "f2", "remote_target": "main", "status": "submitted", "job_id": "102"},
        {"system_name": "s1", "frame_name": "f3", "remote_target": "main", "status": "running", "job_id": "103"},
    ]}
    states = {
        "main": {
            ("s1", "f1"): {"result": "success"},
            ("s1", "f3"): {"result": "failed"},
        }
    }
    active = {"main": {"ids": {"102"}, "names": set()}}
    recovered = recover_manifest_entries(manifest, states, active, timestamp="2026-01-01T00:00:00Z")
    statuses = [entry["status"] for entry in recovered["entries"]]
    assert statuses == ["finished", "submitted", "failed"]
    assert recovered["recovery_summary"]["preserved_active"] == 1
    assert recovered["recovery_summary"]["normalized_terminal"] == 2


def test_recovery_requires_explicit_permission_for_unavailable_target():
    manifest = {"entries": [{"system_name": "s1", "frame_name": "f1", "remote_target": "lost", "status": "running", "job_id": "7"}]}
    states = {"lost": {}}
    active = {"lost": {"ids": set(), "names": set()}}
    planned = recover_manifest_entries(manifest, states, active, unavailable_targets={"lost"})
    assert planned["entries"][0]["status"] == "running"
    assert planned["recovery_summary"]["unresolved"] == 1
    applied = recover_manifest_entries(manifest, states, active, unavailable_targets={"lost"}, allow_lost_targets={"lost"})
    assert applied["entries"][0]["status"] == "selected"
