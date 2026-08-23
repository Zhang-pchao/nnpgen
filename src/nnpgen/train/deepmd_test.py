"""Audited wrapper around DeePMD-kit's native ``dp test`` command."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..utils import write_json


_METRIC_KEYS = {
    "Energy MAE": "energy_mae_eV",
    "Energy RMSE": "energy_rmse_eV",
    "Energy MAE/Natoms": "energy_mae_per_atom_eV",
    "Energy RMSE/Natoms": "energy_rmse_per_atom_eV",
    "Force MAE": "force_mae_eV_per_A",
    "Force RMSE": "force_rmse_eV_per_A",
    "Virial MAE": "virial_mae_eV",
    "Virial RMSE": "virial_rmse_eV",
    "Virial MAE/Natoms": "virial_mae_per_atom_eV",
    "Virial RMSE/Natoms": "virial_rmse_per_atom_eV",
    "Stress MAE": "stress_mae_eV_per_A3",
    "Stress RMSE": "stress_rmse_eV_per_A3",
}
_FLOAT = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_line(line: str) -> str:
    if "DEEPMD INFO" in line:
        line = line.split("DEEPMD INFO", 1)[1]
    return " ".join(line.strip().split())


def parse_dp_test_log(text: str) -> Dict[str, Any]:
    """Parse tested-frame counts and DeePMD's final weighted-average block."""

    frames = 0
    mixed_nloc_groups: Optional[int] = None
    system_count: Optional[int] = None
    metrics: Dict[str, float] = {}
    in_weighted = False

    for raw_line in text.splitlines():
        line = _metric_line(raw_line)
        match = re.search(r"# number of test data\s*:\s*(\d+)", line)
        if match and not in_weighted:
            frames += int(match.group(1))
        match = re.search(r"# mixed-nloc LMDB: testing\s+(\d+)\s+groups", line)
        if match:
            mixed_nloc_groups = int(match.group(1))
        if "weighted average of errors" in line:
            in_weighted = True
            continue
        if not in_weighted:
            continue
        match = re.search(r"# number of systems\s*:\s*(\d+)", line)
        if match:
            system_count = int(match.group(1))
            continue
        if ":" not in line:
            continue
        label, value_text = (part.strip() for part in line.split(":", 1))
        key = _METRIC_KEYS.get(" ".join(label.split()))
        value_match = _FLOAT.search(value_text)
        if key and value_match:
            metrics[key] = float(value_match.group(0))

    result: Dict[str, Any] = {
        "frames_tested": frames,
        "system_count": system_count,
        "mixed_nloc_group_count": mixed_nloc_groups,
        "metrics": metrics,
    }
    if "energy_rmse_per_atom_eV" in metrics:
        result["energy_rmse_meV_per_atom"] = metrics["energy_rmse_per_atom_eV"] * 1000.0
    if "force_rmse_eV_per_A" in metrics:
        result["force_rmse_meV_per_A"] = metrics["force_rmse_eV_per_A"] * 1000.0
    return result


def _version(executable: str, pt_expt: bool) -> str:
    candidates = [[executable, "--version"]]
    if pt_expt:
        candidates.insert(0, [executable, "--pt-expt", "--version"])
    for command in candidates:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return "unknown"


def _audit_files(system: Path, extras: Sequence[Path]) -> List[Path]:
    files: List[Path] = []
    if system.is_file():
        files.append(system)
    elif (system / "data.mdb").is_file():
        files.append(system / "data.mdb")
    files.extend(Path(path).expanduser().resolve() for path in extras)
    unique = {str(path): path for path in files}
    for path in unique.values():
        if not path.is_file():
            raise ValueError(f"audit input is not a file: {path}")
    return [unique[key] for key in sorted(unique)]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, required=True, help="Frozen DeePMD model path.")
    parser.add_argument("--system", type=Path, required=True, help="NPY root, one DP system, or DeePMD-compatible LMDB directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty directory for the log and summary.")
    parser.add_argument("--numb-test", type=int, default=0, help="Frames per DeePMD test group; 0 means all frames.")
    parser.add_argument("--chunk-atoms", type=int, default=0, help="Set DP_TEST_CHUNK_ATOMS for this process; 0 preserves the environment.")
    parser.add_argument("--dp-command", default="dp", help="DeePMD executable name or path.")
    parser.add_argument("--pt-expt", action="store_true", help="Run the PyTorch-exportable backend as `dp --pt-expt test`.")
    parser.add_argument("--detail-file", type=Path, default=None, help="Optional native DeePMD detail-file prefix.")
    parser.add_argument("--audit-input", type=Path, action="append", default=[], help="Additional input file to hash; repeat as needed.")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    model = Path(args.model).expanduser().resolve()
    system = Path(args.system).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"model not found: {model}")
    if not system.exists():
        raise ValueError(f"system not found: {system}")
    if int(args.numb_test) < 0:
        raise ValueError("numb-test must be non-negative")
    if int(args.chunk_atoms) < 0:
        raise ValueError("chunk-atoms must be non-negative")

    executable = shutil.which(str(args.dp_command))
    if executable is None:
        raise ValueError(f"DeePMD executable not found: {args.dp_command}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "dp-test.log"
    summary_path = output_dir / "summary.json"
    if log_path.exists() or summary_path.exists():
        raise FileExistsError(f"refusing existing dp-test outputs in: {output_dir}")

    command = [executable]
    if bool(args.pt_expt):
        command.append("--pt-expt")
    command.extend(["test", "--model", str(model), "--system", str(system), "--numb-test", str(int(args.numb_test))])
    if args.detail_file:
        detail_file = Path(args.detail_file).expanduser().resolve()
        detail_file.parent.mkdir(parents=True, exist_ok=True)
        command.extend(["--detail-file", str(detail_file)])

    environment = os.environ.copy()
    if int(args.chunk_atoms) > 0:
        environment["DP_TEST_CHUNK_ATOMS"] = str(int(args.chunk_atoms))
    recorded_environment = {
        key: environment[key]
        for key in ("CUDA_VISIBLE_DEVICES", "DP_INTERFACE_PREC", "DP_TEST_CHUNK_ATOMS", "OMP_NUM_THREADS")
        if key in environment
    }
    audit_paths = _audit_files(system, list(args.audit_input))
    started_at = _now()
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True, env=environment, check=False)
    wall_time = time.perf_counter() - started
    parsed = parse_dp_test_log(log_path.read_text(encoding="utf-8", errors="replace"))
    valid = result.returncode == 0 and parsed["frames_tested"] > 0 and bool(parsed["metrics"])
    summary: Dict[str, Any] = {
        "schema": "nnpgen.deepmd-test.v1",
        "status": "PASS" if valid else "FAIL",
        "started_at": started_at,
        "finished_at": _now(),
        "host": socket.gethostname(),
        "command": shlex.join(command),
        "dp_executable": executable,
        "deepmd_version": _version(executable, bool(args.pt_expt)),
        "returncode": result.returncode,
        "model": str(model),
        "model_sha256": _sha256(model),
        "system": str(system),
        "input_sha256": {str(path): _sha256(path) for path in audit_paths},
        "numb_test": int(args.numb_test),
        "all_frames_requested": int(args.numb_test) == 0,
        "environment": recorded_environment,
        "log": str(log_path),
        "wall_time_seconds": wall_time,
        "frames_per_second": parsed["frames_tested"] / wall_time if wall_time > 0 else None,
        **parsed,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not valid:
        raise RuntimeError(f"native dp test failed or produced an incomplete weighted summary; see {log_path}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run and audit DeePMD-kit's native dp test command.")
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
