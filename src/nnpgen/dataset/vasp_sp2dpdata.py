#!/usr/bin/env python3
import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


DEFAULT_TYPE_ORDER = ["H", "O", "N", "Na", "Cl", "Ti", "C", "Si"]


def _extract_idx(name: str, prefix: str) -> int:
    m = re.match(r"^{0}(\d+)".format(re.escape(prefix)), name)
    if m:
        return int(m.group(1))
    return 10**12


def _frame_sort_key(frame_dir: Path) -> Tuple[int, int, str]:
    sys_name = frame_dir.parent.name
    frame_name = frame_dir.name
    return (_extract_idx(sys_name, "system_"), _extract_idx(frame_name, "frame_"), str(frame_dir))


def _parse_poscar(poscar_path: Path) -> Tuple[List[str], List[int]]:
    lines = poscar_path.read_text().splitlines()
    if len(lines) < 7:
        raise ValueError("Invalid POSCAR (too short): {0}".format(poscar_path))
    elements = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    if len(elements) != len(counts):
        raise ValueError("POSCAR element/count mismatch: {0}".format(poscar_path))
    return elements, counts


def _find_last_index(lines: Sequence[str], token: str) -> int:
    for i in range(len(lines) - 1, -1, -1):
        if token in lines[i]:
            return i
    return -1


def _float_tokens(line: str) -> List[float]:
    vals: List[float] = []
    for tok in re.findall(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?|[-+]?\d+(?:[Ee][-+]?\d+)?", line):
        vals.append(float(tok))
    return vals


def _parse_energy(lines: Sequence[str]) -> float:
    for line in reversed(lines):
        if "TOTEN" in line and "energy" in line:
            m = re.search(r"TOTEN\s*=\s*([-+]?\d*\.\d+(?:[Ee][-+]?\d+)?|[-+]?\d+(?:[Ee][-+]?\d+)?)", line)
            if m:
                return float(m.group(1))
    raise ValueError("Cannot find TOTEN line")


def _parse_cell(lines: Sequence[str]) -> List[float]:
    idx = _find_last_index(lines, "VOLUME and BASIS-vectors are now")
    if idx < 0:
        raise ValueError("Cannot find cell block")

    cell: List[float] = []
    for i in range(3):
        row_idx = idx + 5 + i
        if row_idx >= len(lines):
            raise ValueError("Cell block truncated")
        vals = _float_tokens(lines[row_idx])
        if len(vals) < 3:
            raise ValueError("Invalid cell row: {0}".format(lines[row_idx]))
        cell.extend(vals[:3])
    return cell


def _parse_coord_force(lines: Sequence[str], natoms: int) -> Tuple[List[float], List[float]]:
    idx = _find_last_index(lines, "TOTAL-FORCE (eV/Angst)")
    if idx < 0:
        raise ValueError("Cannot find TOTAL-FORCE block")

    coords: List[float] = []
    forces: List[float] = []
    i = idx + 2
    while i < len(lines) and len(coords) < natoms * 3:
        vals = _float_tokens(lines[i])
        if len(vals) >= 6:
            coords.extend(vals[0:3])
            forces.extend(vals[3:6])
        i += 1

    if len(coords) != natoms * 3 or len(forces) != natoms * 3:
        raise ValueError("Incomplete coord/force block: got {0} atoms expected {1}".format(len(coords) // 3, natoms))
    return coords, forces


def _parse_virial(lines: Sequence[str]) -> List[float]:
    # OUTCAR line format near stress block:
    # Total    XX   YY   ZZ   XY   YZ   ZX
    for line in reversed(lines):
        if not re.match(r"^\s*Total\s+", line):
            continue
        vals = _float_tokens(line)
        if len(vals) < 6:
            continue
        xx, yy, zz, xy, yz, zx = vals[:6]
        return [xx, xy, zx, xy, yy, yz, zx, yz, zz]
    raise ValueError("Cannot parse virial from OUTCAR")


def _parse_outcar(outcar_path: Path, natoms: int) -> Tuple[float, List[float], List[float], List[float], List[float]]:
    lines = outcar_path.read_text(errors="ignore").splitlines()
    energy = _parse_energy(lines)
    cell = _parse_cell(lines)
    coords, forces = _parse_coord_force(lines, natoms)
    virial = _parse_virial(lines)
    return energy, virial, forces, coords, cell


def _collect_frame_dirs(
    input_root: Path,
    require_finished_task_info: bool,
    require_tag_finished: bool,
    max_frames: int,
) -> List[Path]:
    frames: List[Path] = []
    for frame_dir in input_root.glob("system_*/frame_*"):
        outcar = frame_dir / "OUTCAR"
        poscar = frame_dir / "POSCAR"
        if not outcar.is_file() or not poscar.is_file():
            continue

        if require_tag_finished and not (frame_dir / "tag_finished").is_file():
            continue

        if require_finished_task_info:
            ti = frame_dir / "task_info.json"
            if not ti.is_file():
                continue
            try:
                info = json.loads(ti.read_text())
            except Exception:
                continue
            if str(info.get("status", "")).strip().lower() != "finished":
                continue

        frames.append(frame_dir)

    frames = sorted(frames, key=_frame_sort_key)
    if max_frames > 0:
        frames = frames[:max_frames]
    return frames


def _write_raw(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), rows)


def _write_type_files(output_dir: Path, elements: List[str], counts: List[int], type_order: List[str]) -> None:
    unknown = [e for e in elements if e not in type_order]
    if unknown:
        raise ValueError("Elements not in type order: {0}".format(unknown))

    map_index: Dict[str, int] = {e: i for i, e in enumerate(type_order)}

    with (output_dir / "type_map.raw").open("w") as f:
        for e in type_order:
            f.write(e + "\n")

    with (output_dir / "type.raw").open("w") as f:
        for elem, cnt in zip(elements, counts):
            t = map_index[elem]
            for _ in range(cnt):
                f.write(str(t) + "\n")


def _group_dir_name(group_idx_1based: int, elements: List[str], counts: List[int]) -> str:
    natoms = int(sum(counts))
    elems = "_".join(e.lower() for e in elements)
    return "group_{0:03d}_n{1}_{2}".format(group_idx_1based, natoms, elems)


def _convert_one_group(
    output_dir: Path,
    frame_dirs: List[Path],
    elements: List[str],
    counts: List[int],
    type_order: List[str],
) -> Dict[str, object]:
    natoms = int(sum(counts))

    all_energy: List[float] = []
    all_virial: List[List[float]] = []
    all_force: List[List[float]] = []
    all_coord: List[List[float]] = []
    all_box: List[List[float]] = []
    skipped: List[Dict[str, str]] = []

    for frame_dir in frame_dirs:
        try:
            e, v, f, c, b = _parse_outcar(frame_dir / "OUTCAR", natoms)
            all_energy.append(e)
            all_virial.append(v)
            all_force.append(f)
            all_coord.append(c)
            all_box.append(b)
        except Exception as exc:
            skipped.append({"frame": str(frame_dir), "error": str(exc)})

    if not all_energy:
        return {
            "output_dir": str(output_dir),
            "natoms": natoms,
            "elements": elements,
            "counts": counts,
            "frames_selected": len(frame_dirs),
            "frames_converted": 0,
            "frames_skipped": len(skipped),
            "skipped_examples": skipped[:10],
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    set_dir = output_dir / "set.000"
    set_dir.mkdir(parents=True, exist_ok=True)

    energy_arr = np.array(all_energy, dtype=float).reshape(-1, 1)
    box_arr = np.array(all_box, dtype=float)
    coord_arr = np.array(all_coord, dtype=float)
    force_arr = np.array(all_force, dtype=float)
    virial_arr = np.array(all_virial, dtype=float)

    _write_raw(output_dir / "energy.raw", energy_arr)
    _write_raw(output_dir / "box.raw", box_arr)
    _write_raw(output_dir / "coord.raw", coord_arr)
    _write_raw(output_dir / "force.raw", force_arr)
    _write_raw(output_dir / "virial.raw", virial_arr)

    np.save(str(set_dir / "energy.npy"), energy_arr.reshape(-1))
    np.save(str(set_dir / "box.npy"), box_arr)
    np.save(str(set_dir / "coord.npy"), coord_arr)
    np.save(str(set_dir / "force.npy"), force_arr)
    np.save(str(set_dir / "virial.npy"), virial_arr)

    _write_type_files(output_dir, elements, counts, type_order)

    summary = {
        "output_dir": str(output_dir),
        "natoms": natoms,
        "elements": elements,
        "counts": counts,
        "frames_selected": len(frame_dirs),
        "frames_converted": len(all_energy),
        "frames_skipped": len(skipped),
        "type_order": type_order,
        "skipped_examples": skipped[:10],
    }
    with (output_dir / "convert_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return summary


def add_vasp_sp2dpdata_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-root", required=True, help="Stage-2 root containing system_*/frame_* on server 15")
    parser.add_argument("--output-dir", required=True, help="Output DP dataset directory")
    parser.add_argument(
        "--type-order",
        default=",".join(DEFAULT_TYPE_ORDER),
        help="Comma-separated type_map order, default H,O,N,Na,Cl,Ti,C,Si",
    )
    parser.add_argument("--require-finished-task-info", action="store_true", help="Require task_info.json status=finished")
    parser.add_argument("--require-tag-finished", action="store_true", help="Require tag_finished file")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit number of frames (0 means all)")


def run_vasp_sp2dpdata(args: argparse.Namespace) -> Dict[str, object]:
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    type_order = [x.strip() for x in str(args.type_order).split(",") if x.strip()]

    if not input_root.is_dir():
        raise ValueError("input-root does not exist: {0}".format(input_root))

    frame_dirs = _collect_frame_dirs(
        input_root=input_root,
        require_finished_task_info=bool(args.require_finished_task_info),
        require_tag_finished=bool(args.require_tag_finished),
        max_frames=int(args.max_frames),
    )
    if not frame_dirs:
        raise ValueError("No eligible frames found under: {0}".format(input_root))

    skipped_poscar: List[Dict[str, str]] = []
    grouped: "OrderedDict[Tuple[Tuple[str, ...], Tuple[int, ...]], List[Path]]" = OrderedDict()

    for frame_dir in frame_dirs:
        try:
            elems, cnts = _parse_poscar(frame_dir / "POSCAR")
        except Exception as exc:
            skipped_poscar.append({"frame": str(frame_dir), "error": str(exc)})
            continue
        key = (tuple(elems), tuple(cnts))
        grouped.setdefault(key, []).append(frame_dir)

    if not grouped:
        raise ValueError("No frames with valid POSCAR under: {0}".format(input_root))

    group_items = sorted(grouped.items(), key=lambda kv: (sum(kv[0][1]), list(kv[0][0])))

    summaries: List[Dict[str, object]] = []

    single_group = len(group_items) == 1
    for idx, (key, g_frames) in enumerate(group_items, start=1):
        elems = list(key[0])
        cnts = list(key[1])
        if single_group:
            out_group_dir = output_dir
        else:
            out_group_dir = output_dir / _group_dir_name(idx, elems, cnts)

        s = _convert_one_group(
            output_dir=out_group_dir,
            frame_dirs=g_frames,
            elements=elems,
            counts=cnts,
            type_order=type_order,
        )
        summaries.append(s)

    total_converted = int(sum(int(x.get("frames_converted", 0)) for x in summaries))
    total_selected = int(sum(int(x.get("frames_selected", 0)) for x in summaries))
    total_skipped_parse = int(sum(int(x.get("frames_skipped", 0)) for x in summaries))

    if total_converted <= 0:
        raise ValueError("All frames failed to parse under: {0}".format(input_root))

    summary = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "groups": summaries,
        "group_count": len(summaries),
        "frames_selected": total_selected,
        "frames_converted": total_converted,
        "frames_skipped": total_skipped_parse,
        "frames_skipped_invalid_poscar": len(skipped_poscar),
        "invalid_poscar_examples": skipped_poscar[:10],
        "type_order": type_order,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "convert_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Stage-2 VASP OUTCAR/POSCAR frames to DP data")
    add_vasp_sp2dpdata_arguments(parser)
    args = parser.parse_args()
    run_vasp_sp2dpdata(args)


if __name__ == "__main__":
    main()
