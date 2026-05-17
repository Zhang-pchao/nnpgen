import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from ..config import ELEMENT_ORDER, PROJECT_ROOT, RUN_ROOT
from ..utils import abs_path, ensure_dir, read_json, write_json


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _assert_under_project(path: Path) -> None:
    p = path.resolve()
    root = PROJECT_ROOT.resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"Path outside project root: {p}")


def _parse_poscar_summary(poscar_path: str) -> Dict:
    if not poscar_path:
        return {"elements_order": [], "element_counts": {}, "atom_count": 0, "formula": ""}

    p = Path(poscar_path).resolve()
    if not p.exists():
        return {"elements_order": [], "element_counts": {}, "atom_count": 0, "formula": ""}

    lines = p.read_text().splitlines()
    if len(lines) < 7:
        return {"elements_order": [], "element_counts": {}, "atom_count": 0, "formula": ""}

    elems = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    if len(elems) != len(counts):
        return {"elements_order": [], "element_counts": {}, "atom_count": 0, "formula": ""}

    element_counts = {e: c for e, c in zip(elems, counts)}
    formula = "".join([f"{e}{c}" for e, c in zip(elems, counts)])
    return {
        "elements_order": elems,
        "element_counts": element_counts,
        "atom_count": sum(counts),
        "formula": formula,
    }


def _summary_from_symbols(symbols: List[str]) -> Dict:
    if not symbols:
        return {"elements_order": [], "element_counts": {}, "atom_count": 0, "formula": ""}
    present = set(symbols)
    ordered = [e for e in ELEMENT_ORDER if e in present]
    extras = sorted([e for e in present if e not in ordered])
    elems = ordered + extras
    counts = {e: symbols.count(e) for e in elems}
    formula = "".join([f"{e}{counts[e]}" for e in elems])
    return {
        "elements_order": elems,
        "element_counts": counts,
        "atom_count": len(symbols),
        "formula": formula,
    }


def _parse_xyz_summary(xyz_path: str) -> Dict:
    if not xyz_path:
        return {"elements_order": [], "element_counts": {}, "atom_count": 0, "formula": ""}
    p = Path(xyz_path).resolve()
    if not p.exists():
        return {"elements_order": [], "element_counts": {}, "atom_count": 0, "formula": ""}
    lines = p.read_text().splitlines()
    if len(lines) < 3:
        return {"elements_order": [], "element_counts": {}, "atom_count": 0, "formula": ""}
    symbols = []
    try:
        natoms = int(lines[0].strip())
        atom_lines = lines[2 : 2 + natoms]
    except Exception:
        atom_lines = lines[2:]
    for line in atom_lines:
        parts = line.split()
        if parts:
            symbols.append(parts[0])
    return _summary_from_symbols(symbols)


def _record_kind(path: Path) -> str:
    return "frame_info" if path.name == "frame_info.json" else "frame_record"


def _infer_frame_name(path: Path, info: Dict) -> str:
    frame_dir = str(info.get("paths", {}).get("frame_dir", "")).strip()
    if frame_dir:
        return Path(frame_dir).name
    if path.name == "frame_info.json":
        return path.parent.name
    return path.stem


def _infer_frame_dir(path: Path, info: Dict) -> str:
    frame_dir = str(info.get("paths", {}).get("frame_dir", "")).strip()
    if frame_dir:
        return str(Path(frame_dir).resolve())
    if path.name == "frame_info.json":
        return str(path.parent.resolve())
    guess = path.parent.parent / path.stem
    return str(guess.resolve())


def _collect_stage1_records(run_root: Path) -> List[Path]:
    paths = sorted(run_root.glob("system_*/frame_*/frame_info.json"))
    paths += sorted(run_root.glob("system_*/frame_records/frame_*.json"))
    return paths


def add_stage1_manifest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", required=True, help="11-side run/<job_name> root")
    parser.add_argument("--source-root", required=True, help="Input xyz discovery root for this structure type")
    parser.add_argument("--structure-type", required=True, help="Structure type label used in Stage-1 manifest")
    parser.add_argument("--output", default="", help="Manifest output path under project root")
    parser.add_argument("--finished-only", action="store_true", help="Only include finished entries")


def _map_stage1_status(status: str) -> str:
    s = str(status or "").strip().lower()
    if s in {"prepared", "planned"}:
        return "planned"
    if s in {"submitted", "running", "finished", "failed"}:
        return s
    return "planned"


def run_build_stage1_manifest(args: argparse.Namespace) -> Path:
    run_root = abs_path(args.run_root)
    _assert_under_project(run_root)
    if not run_root.exists():
        raise FileNotFoundError(f"run-root does not exist: {run_root}")

    source_root = Path(args.source_root).expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"source-root does not exist: {source_root}")

    run_manifest_path = run_root / "run_manifest.json"
    run_manifest = read_json(run_manifest_path) if run_manifest_path.exists() else {}
    job_name = str(run_manifest.get("job_name", run_root.name))

    discovered_xyz_count = int(run_manifest.get("discovered_xyz_count", 0))
    input_xyz_used_count = int(run_manifest.get("input_xyz_used_count", 0))

    source_xyz_all = sorted([str(p.resolve()) for p in source_root.rglob("*.xyz") if p.is_file()])

    record_paths = _collect_stage1_records(run_root)
    entries = []

    skipped_nonfinished = 0

    for rp in record_paths:
        info = read_json(rp)
        raw_status = str(info.get("status", "prepared"))
        status = _map_stage1_status(raw_status)
        if args.finished_only and status != "finished":
            skipped_nonfinished += 1
            continue

        paths = dict(info.get("paths", {}))
        pooled_poscar = str(paths.get("pooled_poscar", "")).strip()
        pooled_exists = bool(pooled_poscar and Path(pooled_poscar).exists())

        frame_dir = _infer_frame_dir(rp, info)
        frame_name = _infer_frame_name(rp, info)
        system_name = str(info.get("system_name", "system_unknown"))
        source_xyz_path = str(info.get("source_input", info.get("source_xyz_path", "")))

        sampling_rule = dict(info.get("sampling", {}))
        if not sampling_rule:
            sampling_rule = {
                "frame_start": None,
                "frame_stride": None,
                "frame_max": None,
            }

        final_xyz_path = str(paths.get("final_xyz", "")).strip()
        final_xyz_retained = bool(final_xyz_path and Path(final_xyz_path).exists())

        run_sbatch = str(paths.get("run_sbatch", "")).strip()
        post_sbatch = str(paths.get("post_sbatch", "")).strip()
        frame_info_path = str((Path(frame_dir) / "frame_info.json").resolve())
        frame_record_path = str(rp.resolve()) if _record_kind(rp) == "frame_record" else str(
            (Path(frame_dir).parent / "frame_records" / f"{frame_name}.json").resolve()
        )

        summary = _parse_poscar_summary(pooled_poscar)
        if summary.get("atom_count", 0) <= 0:
            summary = _parse_poscar_summary(str(paths.get("final_poscar", "")))
        if summary.get("atom_count", 0) <= 0:
            summary = _parse_xyz_summary(str(paths.get("initial_xyz", "")))

        removed_paths = [str(x) for x in info.get("removed_paths", []) if str(x).strip()]
        retained_paths = []
        for p in [pooled_poscar, final_xyz_path, run_sbatch, post_sbatch, frame_info_path, frame_record_path]:
            if p and Path(p).exists():
                retained_paths.append(str(Path(p).resolve()))

        entry = {
            "structure_type": args.structure_type,
            "source_root": str(source_root),
            "source_xyz_path": source_xyz_path,
            "source_frame_index_0based": info.get("source_frame_index_0based"),
            "sampling_rule": sampling_rule,
            "task_name": f"{system_name}__{frame_name}",
            "system_name": system_name,
            "frame_name": frame_name,
            "run_dir_11": frame_dir,
            "pooled_poscar_path": str(Path(pooled_poscar).resolve()),
            "final_xyz_path": str(Path(final_xyz_path).resolve()) if final_xyz_path else "",
            "composition_summary": {
                "formula": summary.get("formula", ""),
                "elements_order": summary.get("elements_order", []),
                "element_counts": summary.get("element_counts", {}),
            },
            "atom_count": summary.get("atom_count", 0),
            "status": status,
            "raw_status": raw_status,
            "scheduler": {
                "md_job_id": info.get("job_id"),
                "post_job_id": info.get("post_job_id"),
                "post_status": info.get("post_status"),
            },
            "notes": {
                "record_kind": _record_kind(rp),
                "pooled_poscar_exists": pooled_exists,
                "retained_paths": retained_paths,
                "removed_paths": removed_paths,
                "cleanup_options": info.get("cleanup_options", {}),
                "final_xyz_retained": final_xyz_retained,
            },
        }
        entries.append(entry)

    output_path = abs_path(args.output) if args.output else (PROJECT_ROOT / "plan" / f"{job_name}_stage1_manifest.json")
    _assert_under_project(output_path)
    ensure_dir(output_path.parent)

    manifest = {
        "schema_version": "nnpgen.stage1_manifest.v1",
        "created_at": _now(),
        "job_name": job_name,
        "structure_type": args.structure_type,
        "source_root": str(source_root),
        "run_root_11": str(run_root),
        "source_xyz_count_under_root": len(source_xyz_all),
        "run_manifest_discovered_xyz_count": discovered_xyz_count,
        "run_manifest_input_xyz_used_count": input_xyz_used_count,
        "records_scanned": len(record_paths),
        "entries_count": len(entries),
        "skipped_nonfinished": skipped_nonfinished,
        "entries": entries,
    }
    write_json(output_path, manifest)
    print(f"Stage1 manifest written: {output_path}")
    print(
        f"records_scanned={len(record_paths)} entries_count={len(entries)} "
        f"skipped_nonfinished={skipped_nonfinished}"
    )
    return output_path
