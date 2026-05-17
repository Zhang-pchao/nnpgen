import argparse
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import (
    DEFAULT_CONDA_ENV,
    DEFAULT_CPUS_PER_TASK,
    DEFAULT_DEVICE,
    DEFAULT_FRICTION_PER_FS,
    DEFAULT_FRAME_MAX,
    DEFAULT_FRAME_START,
    DEFAULT_FRAME_STRIDE,
    DEFAULT_GPUS_PER_NODE,
    DEFAULT_INTERVAL,
    DEFAULT_MODEL_PATH,
    DEFAULT_NODES,
    DEFAULT_NTASKS,
    DEFAULT_OMP_NUM_THREADS,
    DEFAULT_PARTITION,
    DEFAULT_POSCAR_POOL_ROOT,
    DEFAULT_POST_CPUS_PER_TASK,
    DEFAULT_POST_NODES,
    DEFAULT_POST_NTASKS,
    DEFAULT_POST_OMP_NUM_THREADS,
    DEFAULT_POST_TEMPLATE,
    DEFAULT_POST_WALLTIME,
    DEFAULT_QOS,
    DEFAULT_STEPS,
    DEFAULT_TEMPERATURE_K,
    DEFAULT_TEMPLATE,
    DEFAULT_TIMESTEP_FS,
    DEFAULT_WALLTIME,
    PROJECT_ROOT,
    RUN_ROOT,
)
from ..utils import (
    abs_path,
    ensure_dir,
    frame_dir_name,
    read_json,
    read_text,
    slugify,
    str_to_bool,
    write_json,
    write_text,
)


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _assert_under_project(path: Path) -> None:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path is outside project root: {resolved}")


def _render_sbatch(template_text: str, context: dict) -> str:
    return template_text.format(**context)


def _parse_job_id(sbatch_stdout: str) -> str:
    m = re.search(r"(\d+)\s*$", sbatch_stdout.strip())
    if not m:
        raise ValueError(f"Could not parse sbatch job id from output: {sbatch_stdout!r}")
    return m.group(1)


def _validate_job_name(job_name: str) -> str:
    clean = slugify(job_name)
    if clean != job_name:
        raise ValueError("--job-name must contain only letters, digits, and underscores")
    if not clean:
        raise ValueError("--job-name is empty after validation")
    return clean


def _derive_system_name(index_1based: int, input_path: Path, used: set) -> str:
    stem = input_path.stem
    prefix = stem.split("__")[0] if "__" in stem else stem
    prefix = slugify(prefix).lower()

    base = f"system_{index_1based:04d}"
    name = f"{base}_{prefix}" if prefix else base

    if name not in used:
        used.add(name)
        return name

    k = 2
    while f"{name}_{k}" in used:
        k += 1
    unique = f"{name}_{k}"
    used.add(unique)
    return unique


def _select_frame_indices(total_frames: int, frame_start: int, frame_stride: int, frame_max: Optional[int]) -> List[int]:
    if total_frames <= 0:
        return []
    if frame_start < 0:
        raise ValueError("--frame-start must be >= 0")
    if frame_stride <= 0:
        raise ValueError("--frame-stride must be > 0")
    if frame_max is not None and frame_max <= 0:
        raise ValueError("--frame-max must be > 0 when provided")

    if total_frames == 1:
        indices = [0]
    else:
        indices = [i for i in range(frame_start, total_frames, frame_stride)]
        if not indices:
            raise ValueError(
                f"No frames selected (total={total_frames}, start={frame_start}, stride={frame_stride})"
            )

    if frame_max is not None:
        indices = indices[:frame_max]

    return indices


def _discover_xyz_files(input_root: Path) -> List[Path]:
    return sorted([p.resolve() for p in input_root.rglob("*.xyz") if p.is_file()])


def _resolve_input_paths(args: argparse.Namespace) -> Tuple[List[Path], int]:
    manual = [abs_path(p) for p in args.input]

    discovered = []
    if args.input_root:
        root = abs_path(args.input_root)
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"--input-root not found or not a directory: {root}")
        discovered = _discover_xyz_files(root)

    combined = []
    seen = set()
    for p in manual + discovered:
        if not p.exists() or not p.is_file():
            continue
        key = str(p)
        if key not in seen:
            seen.add(key)
            combined.append(p)

    if args.input_max_files is not None:
        if args.input_max_files <= 0:
            raise ValueError("--input-max-files must be > 0")
        combined = combined[: args.input_max_files]

    if not combined:
        raise ValueError("No input xyz files resolved. Use --input and/or --input-root")

    return combined, len(discovered)


def _collect_frame_info_paths(run_root: Path) -> List[Path]:
    return sorted(run_root.glob("system_*/frame_*/frame_info.json"))


def _collect_status_json_paths(run_root: Path) -> List[Path]:
    a = sorted(run_root.glob("system_*/frame_*/frame_info.json"))
    b = sorted(run_root.glob("system_*/frame_records/*.json"))
    return a + b


def _update_frame_info(frame_info_path: Path, updates: Dict) -> None:
    info = read_json(frame_info_path)
    info.update(updates)
    info["updated_at"] = _now()
    write_json(frame_info_path, info)


def add_prepare_submit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--job-name", required=True, help="Run name under the configured run root")
    parser.add_argument("--input", nargs="+", default=[], help="One or more absolute XYZ input paths")
    parser.add_argument("--input-root", default=None, help="Recursive root for xyz discovery")
    parser.add_argument("--input-max-files", type=int, default=None, help="Optional cap on number of input xyz files")

    parser.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE)
    parser.add_argument("--frame-start", type=int, default=DEFAULT_FRAME_START)
    parser.add_argument("--frame-max", type=int, default=DEFAULT_FRAME_MAX)
    parser.add_argument("--force", action="store_true", help="Overwrite existing run/<job_name>")

    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument("--timestep-fs", type=float, default=DEFAULT_TIMESTEP_FS)
    parser.add_argument("--friction", type=float, default=DEFAULT_FRICTION_PER_FS)
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    parser.add_argument("--device", choices=["gpu", "cpu"], default=DEFAULT_DEVICE)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))

    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE), help="Absolute MD sbatch template path")
    parser.add_argument("--post-template", default=str(DEFAULT_POST_TEMPLATE), help="Absolute post sbatch template")
    parser.add_argument("--conda-env", default=str(DEFAULT_CONDA_ENV), help="Absolute conda env path")
    parser.add_argument("--python-exe", default="python", help="Python executable used inside sbatch")

    parser.add_argument("--nodes", type=int, default=DEFAULT_NODES)
    parser.add_argument("--ntasks", type=int, default=DEFAULT_NTASKS)
    parser.add_argument("--cpus-per-task", type=int, default=DEFAULT_CPUS_PER_TASK)
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--qos", default=DEFAULT_QOS)
    parser.add_argument("--walltime", default=DEFAULT_WALLTIME)
    parser.add_argument("--gpus-per-node", default=DEFAULT_GPUS_PER_NODE)
    parser.add_argument("--omp-num-threads", type=int, default=DEFAULT_OMP_NUM_THREADS)

    parser.add_argument("--lightweight-poscar-only", action="store_true")
    parser.add_argument("--poscar-pool-root", default=str(DEFAULT_POSCAR_POOL_ROOT))
    parser.add_argument("--cleanup-frame-dir", choices=["true", "false"], default="false")
    parser.add_argument("--delete-initial", choices=["true", "false"], default="true")
    parser.add_argument("--delete-slurm-logs", choices=["true", "false"], default="false")

    parser.add_argument("--dry-run", action="store_true", help="Prepare only; do not call sbatch")


def _prepare_run_root(job_name: str, force: bool) -> Path:
    run_root = RUN_ROOT / job_name
    _assert_under_project(run_root)

    if run_root.exists():
        if not force:
            raise FileExistsError(f"Run root already exists: {run_root}. Use --force to overwrite")
        shutil.rmtree(str(run_root))

    ensure_dir(run_root)
    return run_root


def prepare_runs(args: argparse.Namespace) -> Tuple[Path, List[Dict], List[Dict], int]:
    from ..io_xyz import read_frames, write_frame_xyz

    job_name = _validate_job_name(args.job_name)
    run_root = _prepare_run_root(job_name, force=args.force)

    cleanup_frame_dir = str_to_bool(args.cleanup_frame_dir)
    delete_initial = str_to_bool(args.delete_initial)
    delete_slurm_logs = str_to_bool(args.delete_slurm_logs)

    if cleanup_frame_dir and not args.lightweight_poscar_only:
        raise ValueError("--cleanup-frame-dir true requires --lightweight-poscar-only")

    input_paths, discovered_count = _resolve_input_paths(args)

    template_path = abs_path(args.template)
    post_template_path = abs_path(args.post_template)
    model_path = abs_path(args.model_path)
    conda_env = abs_path(args.conda_env)
    project_root = PROJECT_ROOT.resolve()
    template_text = read_text(template_path)
    post_template_text = read_text(post_template_path)

    poscar_pool_job_dir = None
    if args.lightweight_poscar_only:
        poscar_pool_job_dir = ensure_dir(abs_path(args.poscar_pool_root) / job_name)
        _assert_under_project(poscar_pool_job_dir)

    prepared: List[Dict] = []
    system_summaries: List[Dict] = []
    used_system_names = set()

    for sys_i, input_xyz in enumerate(input_paths, start=1):
        frames = read_frames(input_xyz)
        total_frames = len(frames)
        selected_indices = _select_frame_indices(
            total_frames=total_frames,
            frame_start=args.frame_start,
            frame_stride=args.frame_stride,
            frame_max=args.frame_max,
        )

        system_name = _derive_system_name(sys_i, input_xyz, used_system_names)
        system_dir = ensure_dir(run_root / system_name)
        frame_records_dir = ensure_dir(system_dir / "frame_records")

        for local_i, source_idx in enumerate(selected_indices, start=1):
            atoms = frames[source_idx]
            frame_name = frame_dir_name(local_i)
            frame_dir = ensure_dir(system_dir / frame_name)
            initial_xyz = frame_dir / "initial.xyz"
            traj_xyz = frame_dir / "traj.xyz"
            final_xyz = frame_dir / "final.xyz"
            final_poscar = frame_dir / "POSCAR"
            sbatch_path = frame_dir / "run.sbatch"
            frame_info_path = frame_dir / "frame_info.json"
            post_sbatch_path = frame_dir / "post.sbatch"
            frame_record_path = frame_records_dir / f"{frame_name}.json"

            pooled_poscar = None
            if args.lightweight_poscar_only:
                pooled_system_dir = ensure_dir(poscar_pool_job_dir / system_name)
                pooled_poscar = pooled_system_dir / f"POSCAR_{system_name}_{frame_name}.vasp"

            write_frame_xyz(atoms, initial_xyz)

            slurm_job_name = f"{job_name}_s{sys_i:02d}f{local_i:04d}"
            if len(slurm_job_name) > 120:
                slurm_job_name = slurm_job_name[:120]

            context = {
                "job_name": slurm_job_name,
                "nodes": args.nodes,
                "ntasks": args.ntasks,
                "cpus_per_task": args.cpus_per_task,
                "partition": args.partition,
                "qos": args.qos,
                "walltime": args.walltime,
                "gpus_per_node": args.gpus_per_node,
                "omp_num_threads": args.omp_num_threads,
                "frame_dir": str(frame_dir),
                "project_root": str(project_root),
                "conda_env": str(conda_env),
                "python_exe": args.python_exe,
                "input_xyz": str(initial_xyz),
                "output_xyz": str(traj_xyz),
                "final_xyz": str(final_xyz),
                "final_poscar": str(final_poscar),
                "steps": args.steps,
                "temperature": args.temperature,
                "timestep_fs": args.timestep_fs,
                "friction": args.friction,
                "interval": args.interval,
                "device": args.device,
                "model_path": str(model_path),
            }
            write_text(sbatch_path, _render_sbatch(template_text, context))
            sbatch_path.chmod(0o755)

            post_context = {
                "job_name": f"post_{slurm_job_name}"[:120],
                "nodes": DEFAULT_POST_NODES,
                "ntasks": DEFAULT_POST_NTASKS,
                "cpus_per_task": DEFAULT_POST_CPUS_PER_TASK,
                "partition": args.partition,
                "qos": args.qos,
                "walltime": DEFAULT_POST_WALLTIME,
                "omp_num_threads": DEFAULT_POST_OMP_NUM_THREADS,
                "frame_dir": str(frame_dir),
                "project_root": str(project_root),
                "conda_env": str(conda_env),
                "python_exe": args.python_exe,
                "frame_info_path": str(frame_info_path),
                "frame_record_path": str(frame_record_path),
                "pooled_poscar_path": str(pooled_poscar) if pooled_poscar is not None else "",
                "lightweight_poscar_only": "true" if args.lightweight_poscar_only else "false",
                "cleanup_frame_dir": "true" if cleanup_frame_dir else "false",
                "delete_initial": "true" if delete_initial else "false",
                "delete_slurm_logs": "true" if delete_slurm_logs else "false",
            }
            write_text(post_sbatch_path, _render_sbatch(post_template_text, post_context))
            post_sbatch_path.chmod(0o755)

            frame_info = {
                "job_name": job_name,
                "system_index_1based": sys_i,
                "system_name": system_name,
                "source_input": str(input_xyz),
                "source_filename": input_xyz.name,
                "source_frame_index_0based": source_idx,
                "source_frame_index_1based": source_idx + 1,
                "selected_frame_index_1based": local_i,
                "sampling": {
                    "frame_start": args.frame_start,
                    "frame_stride": args.frame_stride,
                    "frame_max": args.frame_max,
                },
                "md_parameters": {
                    "steps": args.steps,
                    "interval": args.interval,
                    "temperature": args.temperature,
                    "timestep_fs": args.timestep_fs,
                    "friction": args.friction,
                    "device": args.device,
                    "model_path": str(model_path),
                },
                "lightweight_poscar_only": args.lightweight_poscar_only,
                "cleanup_options": {
                    "cleanup_frame_dir": cleanup_frame_dir,
                    "delete_initial": delete_initial,
                    "delete_slurm_logs": delete_slurm_logs,
                },
                "paths": {
                    "frame_dir": str(frame_dir),
                    "initial_xyz": str(initial_xyz),
                    "traj_xyz": str(traj_xyz),
                    "final_xyz": str(final_xyz),
                    "final_poscar": str(final_poscar),
                    "run_sbatch": str(sbatch_path),
                    "post_sbatch": str(post_sbatch_path),
                    "frame_record": str(frame_record_path),
                    "pooled_poscar": str(pooled_poscar) if pooled_poscar is not None else "",
                },
                "job_id": None,
                "status": "prepared",
                "post_job_id": None,
                "post_status": "not_submitted",
                "updated_at": _now(),
            }
            write_json(frame_info_path, frame_info)

            prepared.append(
                {
                    "system_name": system_name,
                    "frame_dir": frame_dir,
                    "frame_info_path": frame_info_path,
                    "run_sbatch": sbatch_path,
                    "post_sbatch": post_sbatch_path,
                    "source_frame_index_0based": source_idx,
                    "lightweight": args.lightweight_poscar_only,
                }
            )

        system_summary = {
            "system_name": system_name,
            "source_input": str(input_xyz),
            "total_frames": total_frames,
            "selected_count": len(selected_indices),
            "selected_indices_0based": selected_indices,
        }
        write_json(system_dir / "system_info.json", system_summary)
        system_summaries.append(system_summary)

    manifest = {
        "schema_version": "nnpgen.md.run_manifest.v1",
        "job_name": job_name,
        "run_root": str(run_root),
        "created_at": _now(),
        "discovered_xyz_count": discovered_count,
        "input_xyz_used_count": len(input_paths),
        "system_count": len(system_summaries),
        "total_selected_frames": len(prepared),
        "lightweight_poscar_only": args.lightweight_poscar_only,
        "systems": system_summaries,
    }
    write_json(run_root / "run_manifest.json", manifest)

    return run_root, prepared, system_summaries, discovered_count


def submit_runs(prepared: List[Dict], dry_run: bool, lightweight: bool) -> Dict[str, List[str]]:
    md_job_ids: List[str] = []
    post_job_ids: List[str] = []

    for entry in prepared:
        frame_dir = entry["frame_dir"]
        sbatch_path = entry["run_sbatch"]
        post_sbatch_path = entry["post_sbatch"]
        info_path = entry["frame_info_path"]

        if dry_run:
            print(f"Prepared {frame_dir} (dry-run)")
            continue

        md_result = subprocess.run(
            ["sbatch", str(sbatch_path)],
            cwd=str(frame_dir),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        md_job_id = _parse_job_id(md_result.stdout)
        _update_frame_info(info_path, {"status": "submitted", "job_id": md_job_id})
        md_job_ids.append(md_job_id)
        print(f"Submitted MD {frame_dir.name}: job_id={md_job_id}")

        if lightweight:
            post_result = subprocess.run(
                ["sbatch", f"--dependency=afterok:{md_job_id}", str(post_sbatch_path)],
                cwd=str(frame_dir),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            post_job_id = _parse_job_id(post_result.stdout)
            _update_frame_info(info_path, {"post_job_id": post_job_id, "post_status": "submitted"})
            post_job_ids.append(post_job_id)
            print(f"Submitted POST {frame_dir.name}: job_id={post_job_id} dependency=afterok:{md_job_id}")

    return {"md": md_job_ids, "post": post_job_ids}


def run_prepare_or_submit(args: argparse.Namespace, do_submit: bool) -> Tuple[Path, List[Dict], Dict[str, List[str]], List[Dict], int]:
    run_root, prepared, system_summaries, discovered_count = prepare_runs(args)
    print(f"Prepared run root: {run_root}")
    print(f"Discovered xyz: {discovered_count}, systems: {len(system_summaries)}, selected frames: {len(prepared)}")
    jobs = submit_runs(prepared, dry_run=(args.dry_run or not do_submit), lightweight=args.lightweight_poscar_only)
    return run_root, prepared, jobs, system_summaries, discovered_count


def _map_slurm_state(raw: str) -> str:
    state = raw.strip().upper()
    if not state:
        return "submitted"

    if state.startswith("COMPLETED"):
        return "finished"

    if state in {"RUNNING", "COMPLETING", "STAGE_OUT", "SIGNALING", "CONFIGURING"}:
        return "running"

    if state in {"PENDING", "SUSPENDED", "RESV_DEL_HOLD", "REQUEUED"}:
        return "submitted"

    failed_prefixes = (
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
    )
    if state.startswith(failed_prefixes):
        return "failed"

    return "submitted"


def _query_job_status_with_reason(job_id: str) -> Tuple[str, str]:
    squeue = subprocess.run(
        ["squeue", "-h", "-j", str(job_id), "-o", "%T|%r"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    qout = squeue.stdout.strip()
    if qout:
        first = qout.splitlines()[0].strip()
        state_raw, reason = (first.split("|", 1) + [""])[:2]
        mapped = _map_slurm_state(state_raw)
        if mapped == "submitted" and reason.strip().upper() == "DEPENDENCYNEVERSATISFIED":
            return "failed", "DependencyNeverSatisfied"
        return mapped, reason.strip()

    sacct = subprocess.run(
        ["sacct", "-n", "-X", "-j", str(job_id), "--format=State"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if sacct.returncode == 0:
        for line in sacct.stdout.splitlines():
            s = line.strip()
            if not s:
                continue
            state = s.split()[0]
            return _map_slurm_state(state), ""

    return "submitted", ""


def _query_job_status(job_id: str) -> str:
    state, _ = _query_job_status_with_reason(job_id)
    return state


def _cancel_job(job_id: str) -> bool:
    result = subprocess.run(
        ["scancel", str(job_id)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return result.returncode == 0


def add_status_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", required=True, help="Absolute run root path")


def run_status(args: argparse.Namespace) -> Dict[str, int]:
    run_root = abs_path(args.run_root)
    _assert_under_project(run_root)

    status_paths = _collect_status_json_paths(run_root)
    counts = {
        "total": 0,
        "prepared": 0,
        "submitted": 0,
        "running": 0,
        "finished": 0,
        "failed": 0,
    }
    cancelled_stuck_posts = 0

    for p in status_paths:
        info = read_json(p)
        paths = info.get("paths", {})
        status = info.get("status", "prepared")

        post_job_id = info.get("post_job_id")
        job_id = info.get("job_id")

        md_state = None
        if job_id:
            md_state = _query_job_status(str(job_id))

        if post_job_id:
            post_state, post_reason = _query_job_status_with_reason(str(post_job_id))

            if md_state == "failed":
                if post_state in {"submitted", "running"} or post_reason.upper() == "DEPENDENCYNEVERSATISFIED":
                    if _cancel_job(str(post_job_id)):
                        cancelled_stuck_posts += 1
                        info["post_status"] = "cancelled_md_failed"
                    else:
                        info["post_status"] = "failed"
                else:
                    info["post_status"] = post_state
                status = "failed"

            elif md_state in {"submitted", "running"}:
                info["post_status"] = post_state
                status = md_state

            elif post_state == "finished":
                info["post_status"] = "finished"
                pooled = paths.get("pooled_poscar", "")
                if pooled and Path(pooled).exists():
                    status = "finished"
                else:
                    status = "failed"

            elif post_state == "failed":
                if post_reason.upper() == "DEPENDENCYNEVERSATISFIED":
                    if _cancel_job(str(post_job_id)):
                        cancelled_stuck_posts += 1
                        info["post_status"] = "cancelled_dependency_never_satisfied"
                    else:
                        info["post_status"] = "failed"
                else:
                    info["post_status"] = "failed"
                status = "failed"

            else:
                info["post_status"] = post_state
                status = post_state

        elif job_id:
            if md_state == "finished":
                pooled = paths.get("pooled_poscar", "")
                if pooled:
                    status = "finished" if Path(pooled).exists() else "failed"
                else:
                    traj = Path(paths.get("traj_xyz", ""))
                    final_xyz = Path(paths.get("final_xyz", ""))
                    poscar = Path(paths.get("final_poscar", ""))
                    status = "finished" if (traj.exists() and final_xyz.exists() and poscar.exists()) else "failed"
            elif md_state == "failed":
                status = "failed"
            else:
                status = md_state

        info["status"] = status
        info["updated_at"] = _now()
        write_json(p, info)

        counts["total"] += 1
        counts[status] = counts.get(status, 0) + 1

    print(
        f"run_root={run_root} total={counts['total']} submitted={counts['submitted']} "
        f"running={counts['running']} finished={counts['finished']} failed={counts['failed']}"
    )
    if cancelled_stuck_posts:
        print(f"run_root={run_root} cancelled_stuck_post_jobs={cancelled_stuck_posts}")
    return counts


def add_submit_post_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", required=True, help="Absolute run root path")
    parser.add_argument("--template", default=str(DEFAULT_POST_TEMPLATE), help="Absolute post sbatch template")
    parser.add_argument("--conda-env", default=str(DEFAULT_CONDA_ENV), help="Absolute conda env path")
    parser.add_argument("--python-exe", default="python", help="Python executable in sbatch")

    parser.add_argument("--nodes", type=int, default=DEFAULT_POST_NODES)
    parser.add_argument("--ntasks", type=int, default=DEFAULT_POST_NTASKS)
    parser.add_argument("--cpus-per-task", type=int, default=DEFAULT_POST_CPUS_PER_TASK)
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--qos", default=DEFAULT_QOS)
    parser.add_argument("--walltime", default=DEFAULT_POST_WALLTIME)
    parser.add_argument("--omp-num-threads", type=int, default=DEFAULT_POST_OMP_NUM_THREADS)

    parser.add_argument("--job-name-prefix", default="nnpgenpost")
    parser.add_argument("--poscar-pool-root", default=str(DEFAULT_POSCAR_POOL_ROOT))
    parser.add_argument("--dry-run", action="store_true")


def run_submit_post(args: argparse.Namespace) -> List[str]:
    run_root = abs_path(args.run_root)
    _assert_under_project(run_root)

    template_text = read_text(abs_path(args.template))
    project_root = PROJECT_ROOT.resolve()
    conda_env = abs_path(args.conda_env)
    poscar_pool_job_dir = ensure_dir(abs_path(args.poscar_pool_root) / run_root.name)
    _assert_under_project(poscar_pool_job_dir)

    info_paths = _collect_frame_info_paths(run_root)
    submitted: List[str] = []

    for i, info_path in enumerate(info_paths, start=1):
        info = read_json(info_path)
        frame_dir = info_path.parent
        final_xyz = frame_dir / "final.xyz"
        post_sbatch = frame_dir / "post.sbatch"

        if not final_xyz.exists():
            info["post_status"] = "skipped_no_final"
            info["updated_at"] = _now()
            write_json(info_path, info)
            continue

        system_name = info.get("system_name", "system_unknown")
        frame_name = frame_dir.name
        pooled_system_dir = ensure_dir(poscar_pool_job_dir / system_name)
        pooled_poscar_path = pooled_system_dir / f"POSCAR_{system_name}_{frame_name}.vasp"

        context = {
            "job_name": f"{args.job_name_prefix}_{i:06d}",
            "nodes": args.nodes,
            "ntasks": args.ntasks,
            "cpus_per_task": args.cpus_per_task,
            "partition": args.partition,
            "qos": args.qos,
            "walltime": args.walltime,
            "omp_num_threads": args.omp_num_threads,
            "frame_dir": str(frame_dir),
            "project_root": str(project_root),
            "conda_env": str(conda_env),
            "python_exe": args.python_exe,
            "frame_info_path": str(info_path),
            "frame_record_path": str(frame_dir / "frame_record.json"),
            "pooled_poscar_path": str(pooled_poscar_path),
            "lightweight_poscar_only": "false",
            "cleanup_frame_dir": "false",
            "delete_initial": "false",
            "delete_slurm_logs": "false",
        }
        write_text(post_sbatch, _render_sbatch(template_text, context))
        post_sbatch.chmod(0o755)

        if args.dry_run:
            info["post_status"] = "prepared"
            info["updated_at"] = _now()
            write_json(info_path, info)
            print(f"Prepared post.sbatch for {frame_dir}")
            continue

        result = subprocess.run(
            ["sbatch", str(post_sbatch)],
            cwd=str(frame_dir),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        job_id = _parse_job_id(result.stdout)
        info["post_job_id"] = job_id
        info["post_status"] = "submitted"
        info["updated_at"] = _now()
        write_json(info_path, info)
        submitted.append(job_id)
        print(f"Submitted post job for {frame_dir.name}: job_id={job_id}")

    print(f"Post jobs submitted: {len(submitted)}")
    return submitted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and optionally submit per-frame MACE MD jobs")
    add_prepare_submit_arguments(parser)
    parser.add_argument("--submit", action="store_true", help="Call sbatch after preparation")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_prepare_or_submit(args, do_submit=args.submit)


if __name__ == "__main__":
    main()
