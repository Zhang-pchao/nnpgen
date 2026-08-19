#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import re
import shlex
import subprocess
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
except ModuleNotFoundError:  # Keep CLI help available without dataset extras.
    np = None

from ..config import DEFAULT_PLAN_ROOT, DEFAULT_REMOTE_HOST, ELEMENT_ORDER, VASP15_RUN_ROOT


DEFAULT_DFT_MANIFEST_GLOB = str(DEFAULT_PLAN_ROOT / "dft_*_manifest.json")
DEFAULT_DFT_RUN_ROOTS = [str(VASP15_RUN_ROOT / "dft")]
TASK_KEY_PATTERN = re.compile(r"^system_[^/]+__frame_\d+$")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_idx(name: str, prefix: str) -> int:
    m = re.match(r"^{0}(\d+)".format(re.escape(prefix)), name)
    if m:
        return int(m.group(1))
    return 10**12


def _frame_sort_key(frame_dir: Path) -> Tuple[int, int, str]:
    if frame_dir.name.startswith("frame_") and frame_dir.parent.name.startswith("system_"):
        return (
            _extract_idx(frame_dir.parent.name, "system_"),
            _extract_idx(frame_dir.name, "frame_"),
            str(frame_dir),
        )
    return (10**12, 10**12, str(frame_dir))


def _find_frame_dirs(input_root: Path) -> List[Path]:
    frame_dirs: List[Path] = []

    if (input_root / "POSCAR").is_file():
        frame_dirs.append(input_root)

    frame_dirs.extend([p for p in input_root.glob("frame_*") if (p / "POSCAR").is_file()])
    frame_dirs.extend([p for p in input_root.glob("system_*/frame_*") if (p / "POSCAR").is_file()])

    unique = OrderedDict((str(p.resolve()), p) for p in frame_dirs)
    return sorted(unique.values(), key=_frame_sort_key)


def _frame_task_key(frame_dir: Path) -> str:
    if frame_dir.name.startswith("frame_") and frame_dir.parent.name.startswith("system_"):
        return "{0}__{1}".format(frame_dir.parent.name, frame_dir.name)
    return ""


def _task_key_from_text(s: str) -> str:
    txt = str(s).strip().replace("\\", "/")
    if TASK_KEY_PATTERN.match(txt):
        return txt
    m = re.search(r"(system_[^/]+)/(frame_\d+)", txt)
    if m:
        return "{0}__{1}".format(m.group(1), m.group(2))
    return ""


def _load_exclude_list(exclude_list_path: Path, input_root: Path) -> Dict[str, object]:
    if not exclude_list_path.is_file():
        raise ValueError("exclude-list file not found: {0}".format(exclude_list_path))

    lines = exclude_list_path.read_text().splitlines()
    task_keys: Set[str] = set()
    poscar_paths: Set[str] = set()
    invalid_entries: List[Dict[str, object]] = []
    entry_count = 0

    for i, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entry_count += 1

        value = line
        low = line.lower()
        if low.startswith("task_key:"):
            value = line.split(":", 1)[1].strip()
        elif low.startswith("path:"):
            value = line.split(":", 1)[1].strip()

        if not value:
            invalid_entries.append({"line_no": i, "line": raw_line, "reason": "empty_after_prefix"})
            continue

        matched = False

        tk_direct = _task_key_from_text(value)
        if tk_direct:
            task_keys.add(tk_direct)
            matched = True

        p = Path(value).expanduser()
        candidates: List[Path] = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append((input_root / p))
            candidates.append((Path.cwd() / p))

        seen = set()
        uniq_candidates: List[Path] = []
        for c in candidates:
            s = str(c)
            if s not in seen:
                seen.add(s)
                uniq_candidates.append(c)

        for cand in uniq_candidates:
            if cand.is_dir():
                tk = _frame_task_key(cand)
                if tk:
                    task_keys.add(tk)
                    matched = True
                pos = cand / "POSCAR"
                if pos.is_file():
                    poscar_paths.add(str(pos.resolve()))
                    matched = True
            elif cand.is_file():
                if cand.name == "POSCAR":
                    poscar_paths.add(str(cand.resolve()))
                    tk = _frame_task_key(cand.parent)
                    if tk:
                        task_keys.add(tk)
                    matched = True
                else:
                    tk = _task_key_from_text(str(cand))
                    if tk:
                        task_keys.add(tk)
                        matched = True

        if not matched:
            invalid_entries.append({"line_no": i, "line": raw_line, "reason": "unresolved_or_unrecognized"})

    return {
        "exclude_list_path": str(exclude_list_path),
        "entry_count": entry_count,
        "task_keys": task_keys,
        "poscar_paths": poscar_paths,
        "invalid_entries": invalid_entries,
    }


def _load_dft_skip_keys(
    manifest_glob: str,
    statuses: Optional[Set[str]] = None,
) -> Dict[str, object]:
    manifest_paths = sorted(glob.glob(str(manifest_glob)))
    task_keys: Set[str] = set()
    source_poscars: Set[str] = set()
    entry_count = 0
    malformed_manifests: List[str] = []

    for mp in manifest_paths:
        mpath = Path(mp)
        try:
            js = json.loads(mpath.read_text())
        except Exception:
            malformed_manifests.append(str(mpath))
            continue

        if not isinstance(js, dict):
            continue
        entries = js.get("entries")
        if not isinstance(entries, list):
            continue

        for e in entries:
            if not isinstance(e, dict):
                continue
            entry_count += 1

            status = str(e.get("status", "")).strip().lower()
            if statuses and status and status not in statuses:
                continue

            task_key = str(e.get("task_key") or e.get("source_task_key") or e.get("stage1_task_name") or "").strip()
            if not task_key and e.get("system_name") and e.get("frame_name"):
                task_key = "{0}__{1}".format(e["system_name"], e["frame_name"])
            if task_key:
                task_keys.add(task_key)

            src_poscar = str(e.get("source_structure_path") or e.get("source_poscar_path_11") or "").strip()
            if src_poscar:
                try:
                    source_poscars.add(str(Path(src_poscar).resolve()))
                except Exception:
                    source_poscars.add(src_poscar)

    return {
        "manifest_paths": manifest_paths,
        "manifest_count": len(manifest_paths),
        "malformed_manifests": malformed_manifests,
        "entry_count": entry_count,
        "task_keys": task_keys,
        "source_poscar_paths": source_poscars,
    }


def _load_remote_skip_keys(remote_host: str, run_roots: Sequence[str]) -> Dict[str, object]:
    roots = [str(x).strip() for x in run_roots if str(x).strip()]
    if not roots:
        return {"task_keys": set(), "frame_dir_count": 0, "roots": [], "errors": ["no run roots provided"]}

    find_cmds = []
    for root in roots:
        q = shlex.quote(root)
        find_cmds.append(
            "if [ -d {r} ]; then find {r} -mindepth 2 -maxdepth 2 -type d -path '*/system_*/frame_*'; fi".format(r=q)
        )

    remote_cmd = "set -euo pipefail; " + " ; ".join(find_cmds)
    proc = subprocess.run(
        ["ssh", remote_host, "bash -lc {0}".format(shlex.quote(remote_cmd))],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    if proc.returncode != 0:
        return {
            "task_keys": set(),
            "frame_dir_count": 0,
            "roots": roots,
            "errors": [proc.stderr.strip()[-2000:]],
        }

    task_keys: Set[str] = set()
    frame_dir_count = 0
    for line in proc.stdout.splitlines():
        p = line.strip()
        if not p:
            continue
        frame_dir_count += 1

        m = re.search(r"/(system_[^/]+)/(frame_\d+)$", p)
        if not m:
            continue
        task_keys.add("{0}__{1}".format(m.group(1), m.group(2)))

    return {
        "task_keys": task_keys,
        "frame_dir_count": frame_dir_count,
        "roots": roots,
        "errors": [],
    }


def _filter_dft_selected_frames(
    frame_dirs: List[Path],
    selected_task_keys: Set[str],
    source_poscar_paths: Set[str],
) -> Tuple[List[Path], List[Dict[str, str]]]:
    kept: List[Path] = []
    skipped: List[Dict[str, str]] = []

    for frame_dir in frame_dirs:
        task_key = _frame_task_key(frame_dir)
        poscar_path = str((frame_dir / "POSCAR").resolve())

        reason = ""
        if task_key and task_key in selected_task_keys:
            reason = "matched_task_key"
        elif poscar_path in source_poscar_paths:
            reason = "matched_source_structure_path"

        if reason:
            skipped.append(
                {
                    "frame": str(frame_dir),
                    "task_key": task_key,
                    "reason": reason,
                }
            )
        else:
            kept.append(frame_dir)

    return kept, skipped


def _filter_exclude_list_frames(
    frame_dirs: List[Path],
    exclude_task_keys: Set[str],
    exclude_poscar_paths: Set[str],
) -> Tuple[List[Path], List[Dict[str, str]]]:
    kept: List[Path] = []
    skipped: List[Dict[str, str]] = []

    for frame_dir in frame_dirs:
        task_key = _frame_task_key(frame_dir)
        poscar_path = str((frame_dir / "POSCAR").resolve())

        reason = ""
        if task_key and task_key in exclude_task_keys:
            reason = "matched_exclude_task_key"
        elif poscar_path in exclude_poscar_paths:
            reason = "matched_exclude_poscar_path"

        if reason:
            skipped.append(
                {
                    "frame": str(frame_dir),
                    "task_key": task_key,
                    "reason": reason,
                }
            )
        else:
            kept.append(frame_dir)

    return kept, skipped


def _parse_poscar(poscar_path: Path, type_map: List[str]) -> Dict[str, object]:
    lines = [ln.rstrip() for ln in poscar_path.read_text().splitlines() if ln.strip()]
    if len(lines) < 8:
        raise ValueError("POSCAR too short: {0}".format(poscar_path))

    try:
        scale = float(lines[1].split()[0])
    except Exception as exc:
        raise ValueError("Invalid scale line in POSCAR: {0}".format(poscar_path)) from exc

    try:
        lattice = np.array(
            [
                [float(x) for x in lines[2].split()[:3]],
                [float(x) for x in lines[3].split()[:3]],
                [float(x) for x in lines[4].split()[:3]],
            ],
            dtype=float,
        )
    except Exception as exc:
        raise ValueError("Invalid lattice vectors in POSCAR: {0}".format(poscar_path)) from exc

    if scale <= 0:
        raise ValueError("Unsupported POSCAR scale <= 0 in {0}".format(poscar_path))
    lattice = lattice * scale

    elements = lines[5].split()
    if not elements:
        raise ValueError("Missing element line in POSCAR: {0}".format(poscar_path))

    try:
        counts = [int(x) for x in lines[6].split()]
    except Exception as exc:
        raise ValueError("Invalid atom counts line in POSCAR: {0}".format(poscar_path)) from exc

    if len(elements) != len(counts):
        raise ValueError("Element/count length mismatch in POSCAR: {0}".format(poscar_path))

    natoms = int(sum(counts))
    if natoms <= 0:
        raise ValueError("natoms <= 0 in POSCAR: {0}".format(poscar_path))

    idx = 7
    if idx >= len(lines):
        raise ValueError("Missing coordinate mode in POSCAR: {0}".format(poscar_path))

    if lines[idx].strip().lower().startswith("s"):
        idx += 1

    if idx >= len(lines):
        raise ValueError("Missing coordinate mode after selective dynamics: {0}".format(poscar_path))

    mode = lines[idx].strip().lower()
    idx += 1

    if len(lines) < idx + natoms:
        raise ValueError("Not enough coordinate lines in POSCAR: {0}".format(poscar_path))

    coord_in = []
    for i in range(natoms):
        toks = lines[idx + i].split()
        if len(toks) < 3:
            raise ValueError("Invalid coordinate line in POSCAR: {0}".format(poscar_path))
        coord_in.append([float(toks[0]), float(toks[1]), float(toks[2])])

    coord_arr = np.array(coord_in, dtype=float)

    if mode.startswith("d"):
        coord_cart = np.matmul(coord_arr, lattice)
    elif mode.startswith("c") or mode.startswith("k"):
        coord_cart = coord_arr * scale
    else:
        raise ValueError("Unsupported coordinate mode {0} in POSCAR: {1}".format(mode, poscar_path))

    map_index = {e: i for i, e in enumerate(type_map)}
    atom_types: List[int] = []
    atom_symbols: List[str] = []
    for elem, cnt in zip(elements, counts):
        if elem not in map_index:
            raise ValueError("Element {0} not in type map for POSCAR: {1}".format(elem, poscar_path))
        atom_types.extend([map_index[elem]] * cnt)
        atom_symbols.extend([elem] * cnt)

    return {
        "elements": elements,
        "counts": counts,
        "natoms": natoms,
        "coord_cart": coord_cart,
        "box": lattice,
        "atom_types": np.array(atom_types, dtype=np.int32),
        "atom_symbols": atom_symbols,
    }


def _write_raw(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), rows)


def _write_type_files(output_dir: Path, elements: List[str], counts: List[int], type_map: List[str]) -> None:
    map_index: Dict[str, int] = {e: i for i, e in enumerate(type_map)}

    with (output_dir / "type_map.raw").open("w") as f:
        for e in type_map:
            f.write(e + "\n")

    with (output_dir / "type.raw").open("w") as f:
        for elem, cnt in zip(elements, counts):
            ti = map_index[elem]
            for _ in range(int(cnt)):
                f.write(str(ti) + "\n")


def _group_dir_name(group_idx_1based: int, elements: List[str], counts: List[int]) -> str:
    natoms = int(sum(counts))
    elems = "_".join(e.lower() for e in elements)
    return "group_{0:03d}_n{1}_{2}".format(group_idx_1based, natoms, elems)


def _collect_by_group(frame_dirs: List[Path], type_map: List[str]) -> Tuple["OrderedDict[Tuple[Tuple[str, ...], Tuple[int, ...]], List[Path]]", List[Dict[str, str]]]:
    grouped: "OrderedDict[Tuple[Tuple[str, ...], Tuple[int, ...]], List[Path]]" = OrderedDict()
    skipped: List[Dict[str, str]] = []

    for frame_dir in frame_dirs:
        poscar = frame_dir / "POSCAR"
        try:
            info = _parse_poscar(poscar, type_map)
            key = (tuple(info["elements"]), tuple(info["counts"]))
            grouped.setdefault(key, []).append(frame_dir)
        except Exception as exc:
            skipped.append({"frame": str(frame_dir), "error": str(exc)})

    return grouped, skipped


def add_predict_poscar_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-root", required=True, help="Path containing system_*/frame_*/POSCAR or frame_* directories")
    parser.add_argument("--output-dir", required=True, help="Output DP-data directory")
    parser.add_argument("--model", required=True, help="DeepMD model checkpoint (.pt/.pb)")
    parser.add_argument(
        "--type-map",
        default=",".join(ELEMENT_ORDER),
        help="Comma-separated output type map; model atom types are read from the checkpoint.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size per composition group")
    parser.add_argument(
        "--skip-dft-selected",
        dest="skip_dft_selected",
        action="store_true",
        default=True,
        help="Skip frames that already appear in DFT manifests (default: enabled).",
    )
    parser.add_argument(
        "--no-skip-dft-selected",
        dest="skip_dft_selected",
        action="store_false",
        help="Disable DFT selection skipping.",
    )
    parser.add_argument(
        "--dft-manifest-glob",
        default=DEFAULT_DFT_MANIFEST_GLOB,
        help="Glob for DFT manifest JSON files.",
    )
    parser.add_argument(
        "--dft-skip-statuses",
        default="",
        help="Optional comma-separated status filter for skip lookup; empty means all statuses",
    )
    parser.add_argument(
        "--remote-host",
        default=DEFAULT_REMOTE_HOST,
        help="SSH host for the optional fallback frame scan.",
    )
    parser.add_argument(
        "--dft-run-roots",
        default=",".join(DEFAULT_DFT_RUN_ROOTS),
        help="Comma-separated remote DFT run roots for fallback scanning.",
    )
    parser.add_argument(
        "--fallback-remote-scan",
        action="store_true",
        help="Also scan remote DFT run directories for selected frame keys.",
    )
    parser.add_argument(
        "--exclude-list",
        default="",
        help="Optional text file with explicit exclusions (task keys and/or frame/POSCAR paths)",
    )


def run_predict_poscar(args: argparse.Namespace) -> Dict[str, object]:
    if np is None:
        raise RuntimeError('NumPy is required; install nnpgen with the "dataset" extra')
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model)
    output_type_map = [x.strip() for x in str(args.type_map).split(",") if x.strip()]

    if not input_root.exists():
        raise ValueError("input-root not found: {0}".format(input_root))
    if not model_path.is_file():
        raise ValueError("model not found: {0}".format(model_path))
    if not output_type_map:
        raise ValueError("type-map is empty")
    if int(args.batch_size) <= 0:
        raise ValueError("batch-size must be > 0")

    frame_dirs_all = _find_frame_dirs(input_root)
    if not frame_dirs_all:
        raise ValueError("No frame directories with POSCAR found under: {0}".format(input_root))

    frame_dirs = frame_dirs_all
    dft_skip_meta: Dict[str, object] = {
        "enabled": bool(args.skip_dft_selected),
        "manifest_glob": str(args.dft_manifest_glob),
        "manifest_count": 0,
        "entry_count": 0,
        "skip_task_key_count": 0,
        "skip_source_poscar_count": 0,
        "frames_skipped_count": 0,
        "frames_skipped_examples": [],
        "statuses_filter": [],
        "malformed_manifests": [],
        "fallback_remote_scan": {
            "enabled": False,
            "remote_host": str(args.remote_host),
            "run_roots": [],
            "frame_dir_count": 0,
            "task_key_count": 0,
            "errors": [],
        },
    }
    exclude_list_meta: Dict[str, object] = {
        "enabled": False,
        "exclude_list_path": "",
        "entry_count": 0,
        "task_key_count": 0,
        "poscar_path_count": 0,
        "frames_skipped_count": 0,
        "frames_skipped_examples": [],
        "invalid_entries": [],
    }

    if bool(args.skip_dft_selected):
        statuses_filter = {x.strip().lower() for x in str(args.dft_skip_statuses).split(",") if x.strip()}
        skip_info = _load_dft_skip_keys(str(args.dft_manifest_glob), statuses_filter or None)
        dft_skip_meta["manifest_count"] = int(skip_info.get("manifest_count", 0))
        dft_skip_meta["entry_count"] = int(skip_info.get("entry_count", 0))
        dft_skip_meta["skip_task_key_count"] = len(skip_info.get("task_keys", set()))
        dft_skip_meta["skip_source_poscar_count"] = len(skip_info.get("source_poscar_paths", set()))
        dft_skip_meta["statuses_filter"] = sorted(list(statuses_filter))
        dft_skip_meta["malformed_manifests"] = list(skip_info.get("malformed_manifests", []))

        skip_task_keys: Set[str] = set(skip_info.get("task_keys", set()))
        skip_poscar_paths: Set[str] = set(skip_info.get("source_poscar_paths", set()))

        do_fallback = bool(args.fallback_remote_scan)
        if do_fallback:
            run_roots = [x.strip() for x in str(args.dft_run_roots).split(",") if x.strip()]
            fb = _load_remote_skip_keys(str(args.remote_host), run_roots)
            fb_keys = set(fb.get("task_keys", set()))
            skip_task_keys.update(fb_keys)

            dft_skip_meta["fallback_remote_scan"] = {
                "enabled": True,
                "remote_host": str(args.remote_host),
                "run_roots": run_roots,
                "frame_dir_count": int(fb.get("frame_dir_count", 0)),
                "task_key_count": len(fb_keys),
                "errors": list(fb.get("errors", [])),
            }
            dft_skip_meta["skip_task_key_count"] = len(skip_task_keys)

        frame_dirs, skipped_dft = _filter_dft_selected_frames(
            frame_dirs=frame_dirs,
            selected_task_keys=skip_task_keys,
            source_poscar_paths=skip_poscar_paths,
        )
        dft_skip_meta["frames_skipped_count"] = len(skipped_dft)
        dft_skip_meta["frames_skipped_examples"] = skipped_dft[:50]

    frames_after_dft_skip = len(frame_dirs)

    if str(args.exclude_list).strip():
        excl = _load_exclude_list(Path(str(args.exclude_list).strip()), input_root)
        exclude_list_meta["enabled"] = True
        exclude_list_meta["exclude_list_path"] = str(excl.get("exclude_list_path", ""))
        exclude_list_meta["entry_count"] = int(excl.get("entry_count", 0))
        exclude_list_meta["task_key_count"] = len(excl.get("task_keys", set()))
        exclude_list_meta["poscar_path_count"] = len(excl.get("poscar_paths", set()))
        exclude_list_meta["invalid_entries"] = list(excl.get("invalid_entries", []))[:50]

        frame_dirs, skipped_manual = _filter_exclude_list_frames(
            frame_dirs=frame_dirs,
            exclude_task_keys=set(excl.get("task_keys", set())),
            exclude_poscar_paths=set(excl.get("poscar_paths", set())),
        )
        exclude_list_meta["frames_skipped_count"] = len(skipped_manual)
        exclude_list_meta["frames_skipped_examples"] = skipped_manual[:50]

    if not frame_dirs:
        raise ValueError("No frames left after DFT/exclude filters under: {0}".format(input_root))

    frames_after_all_filters = len(frame_dirs)
    if int(args.max_frames) > 0:
        frame_dirs = frame_dirs[: int(args.max_frames)]
    frames_after_max_frames = len(frame_dirs)

    from deepmd.infer import DeepPot

    dp = DeepPot(str(model_path))
    model_type_map = [str(value) for value in dp.get_type_map()]
    if not model_type_map:
        raise ValueError("model type map is empty")
    grouped, skipped_group = _collect_by_group(frame_dirs, model_type_map)
    if not grouped:
        raise ValueError("No valid frames after POSCAR parsing")

    group_items = sorted(grouped.items(), key=lambda kv: (sum(kv[0][1]), list(kv[0][0])))
    single_group = len(group_items) == 1

    summaries: List[Dict[str, object]] = []

    for idx, (key, g_frames) in enumerate(group_items, start=1):
        elements = list(key[0])
        counts = list(key[1])
        natoms = int(sum(counts))
        missing_output_types = sorted(set(elements) - set(output_type_map))
        if missing_output_types:
            raise ValueError("Elements missing from output type map: {0}".format(missing_output_types))

        if single_group:
            out_group = output_dir
        else:
            out_group = output_dir / _group_dir_name(idx, elements, counts)

        all_energy: List[float] = []
        all_force: List[np.ndarray] = []
        all_coord: List[np.ndarray] = []
        all_box: List[np.ndarray] = []
        all_virial: List[np.ndarray] = []
        converted_frames: List[str] = []
        skipped_frames: List[Dict[str, str]] = []

        bsize = int(args.batch_size)
        for start in range(0, len(g_frames), bsize):
            batch_frames = g_frames[start : start + bsize]
            batch_coord: List[np.ndarray] = []
            batch_box: List[np.ndarray] = []
            batch_frame_names: List[str] = []
            atype_ref: Optional[np.ndarray] = None

            for frame_dir in batch_frames:
                try:
                    parsed = _parse_poscar(frame_dir / "POSCAR", model_type_map)
                    atype = parsed["atom_types"]
                    if int(parsed["natoms"]) != natoms:
                        raise ValueError("natoms changed inside group")

                    if atype_ref is None:
                        atype_ref = atype
                    elif not np.array_equal(atype_ref, atype):
                        raise ValueError("atom order/type mismatch within same composition group")

                    batch_coord.append(parsed["coord_cart"].reshape(-1))
                    batch_box.append(parsed["box"].reshape(-1))
                    batch_frame_names.append(str(frame_dir))
                except Exception as exc:
                    skipped_frames.append({"frame": str(frame_dir), "error": str(exc)})

            if not batch_coord:
                continue

            coord_arr = np.array(batch_coord, dtype=np.float64)
            box_arr = np.array(batch_box, dtype=np.float64)

            try:
                pred = dp.eval(coord_arr, box_arr, atype_ref)
                energy = np.array(pred[0], dtype=float).reshape(-1)
                force = np.array(pred[1], dtype=float).reshape(len(batch_frame_names), natoms, 3)
                virial = np.array(pred[2], dtype=float).reshape(len(batch_frame_names), 9)
            except Exception as exc:
                for frame_name in batch_frame_names:
                    skipped_frames.append({"frame": frame_name, "error": "DeepPot eval failed: {0}".format(exc)})
                continue

            for i, frame_name in enumerate(batch_frame_names):
                all_energy.append(float(energy[i]))
                all_coord.append(coord_arr[i].reshape(natoms * 3))
                all_force.append(force[i].reshape(natoms * 3))
                all_box.append(box_arr[i].reshape(9))
                all_virial.append(virial[i].reshape(9))
                converted_frames.append(frame_name)

        out_group.mkdir(parents=True, exist_ok=True)

        if all_energy:
            set_dir = out_group / "set.000"
            set_dir.mkdir(parents=True, exist_ok=True)

            energy_arr = np.array(all_energy, dtype=float).reshape(-1, 1)
            coord_arr = np.array(all_coord, dtype=float)
            force_arr = np.array(all_force, dtype=float)
            box_arr = np.array(all_box, dtype=float)
            virial_arr = np.array(all_virial, dtype=float)

            _write_raw(out_group / "energy.raw", energy_arr)
            _write_raw(out_group / "coord.raw", coord_arr)
            _write_raw(out_group / "force.raw", force_arr)
            _write_raw(out_group / "box.raw", box_arr)
            _write_raw(out_group / "virial.raw", virial_arr)

            np.save(str(set_dir / "energy.npy"), energy_arr.reshape(-1))
            np.save(str(set_dir / "coord.npy"), coord_arr)
            np.save(str(set_dir / "force.npy"), force_arr)
            np.save(str(set_dir / "box.npy"), box_arr)
            np.save(str(set_dir / "virial.npy"), virial_arr)

            _write_type_files(out_group, elements, counts, output_type_map)

            with (out_group / "source_frames.txt").open("w") as fw:
                for fr in converted_frames:
                    fw.write(fr + "\n")

        group_summary = {
            "output_dir": str(out_group),
            "elements": elements,
            "counts": counts,
            "natoms": natoms,
            "frames_selected": len(g_frames),
            "frames_converted": len(all_energy),
            "frames_skipped": len(skipped_frames),
            "skipped_examples": skipped_frames[:20],
        }
        (out_group / "predict_summary.json").write_text(json.dumps(group_summary, indent=2, sort_keys=True) + "\n")
        summaries.append(group_summary)

    summary = {
        "created_at": _now(),
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "model": str(model_path),
        "output_type_map": output_type_map,
        "model_type_map": model_type_map,
        "frames_discovered": len(frame_dirs_all),
        "frames_after_dft_skip": frames_after_dft_skip,
        "frames_after_all_filters": frames_after_all_filters,
        "frames_after_max_frames": frames_after_max_frames,
        "dft_skip": dft_skip_meta,
        "exclude_list": exclude_list_meta,
        "groups": summaries,
        "group_count": len(summaries),
        "frames_selected": int(sum(int(x.get("frames_selected", 0)) for x in summaries)),
        "frames_converted": int(sum(int(x.get("frames_converted", 0)) for x in summaries)),
        "frames_skipped": int(sum(int(x.get("frames_skipped", 0)) for x in summaries)),
        "frames_skipped_invalid_poscar": len(skipped_group),
        "invalid_poscar_examples": skipped_group[:20],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predict_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict POSCAR structures using a DeepMD model and export DP data")
    add_predict_poscar_arguments(parser)
    args = parser.parse_args()
    run_predict_poscar(args)


if __name__ == "__main__":
    main()
