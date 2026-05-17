"""Versioned manifest helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping


MANIFEST_SCHEMA_VERSION = "nnpgen.manifest.v1"


LEGACY_KEY_MAP = {
    "poscar_path_11": "source_structure_path",
    "target_work_dir_15": "remote_task_dir",
    "task_dir_15": "remote_task_dir",
    "host_15": "remote_host",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_manifest_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(record)
    for old_key, new_key in LEGACY_KEY_MAP.items():
        if new_key not in out and old_key in out:
            out[new_key] = out[old_key]
    out.setdefault("schema_version", MANIFEST_SCHEMA_VERSION)
    out.setdefault("status", "planned")
    return out


def new_manifest(kind: str, **extra: Any) -> Dict[str, Any]:
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": kind,
        "created_at": utc_now(),
        "records": [],
    }
    manifest.update(extra)
    return manifest
