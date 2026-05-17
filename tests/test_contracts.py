from nnpgen.contracts import MANIFEST_SCHEMA_VERSION, normalize_manifest_record, new_manifest


def test_normalize_manifest_record_accepts_legacy_keys():
    record = normalize_manifest_record(
        {
            "poscar_path_11": "/example/POSCAR",
            "task_dir_15": "/remote/task",
        }
    )

    assert record["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert record["source_structure_path"] == "/example/POSCAR"
    assert record["remote_task_dir"] == "/remote/task"
    assert record["status"] == "planned"


def test_new_manifest_has_schema_version():
    manifest = new_manifest("dft")

    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["kind"] == "dft"
    assert manifest["records"] == []
