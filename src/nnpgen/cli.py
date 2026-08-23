"""Domain-oriented command line interface for nnpgen."""

from __future__ import annotations

import argparse
import importlib
import json
import runpy
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .config import apply_config_environment, load_config


@dataclass(frozen=True)
class CommandSpec:
    help: str
    module: Optional[str] = None
    argv_prefix: tuple = ()
    add_args: Optional[str] = None
    runner: Optional[str] = None
    runner_kwargs: tuple = ()


def _function_runner(add_args: str, runner: str, argv: List[str], prog: str, runner_kwargs: tuple = ()) -> None:
    add_module_name, add_name = add_args.rsplit(":", 1)
    run_module_name, run_name = runner.rsplit(":", 1)
    add_module = importlib.import_module(add_module_name)
    run_module = importlib.import_module(run_module_name)

    parser = argparse.ArgumentParser(prog=prog)
    getattr(add_module, add_name)(parser)
    args = parser.parse_args(argv)
    result = getattr(run_module, run_name)(args, *runner_kwargs)
    if result is not None and not isinstance(result, (str, bytes)):
        try:
            json.dumps(result)
        except TypeError:
            return


def _module_runner(module: str, argv: List[str]) -> None:
    old_argv = sys.argv[:]
    sys.argv = [module] + argv
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = old_argv


COMMANDS: Dict[str, Dict[str, CommandSpec]] = {
    "md": {
        "run": CommandSpec("Run ASE/MACE MD for one frame.", module="nnpgen.md.mace"),
        "postprocess": CommandSpec("Post-process one MD frame directory.", module="nnpgen.md.postprocess"),
        "prepare": CommandSpec("Prepare MD frame directories.", module="nnpgen.md.submit", argv_prefix=("prepare",)),
        "submit": CommandSpec("Prepare and submit MD jobs.", module="nnpgen.md.submit", argv_prefix=("submit",)),
        "status": CommandSpec("Summarize MD run status.", module="nnpgen.md.submit", argv_prefix=("status",)),
        "submit-post": CommandSpec("Submit post-processing jobs.", module="nnpgen.md.submit", argv_prefix=("submit-post",)),
        "build-manifest": CommandSpec(
            "Build a Stage-1 candidate manifest from MD outputs.",
            add_args="nnpgen.md.stage1_manifest:add_stage1_manifest_arguments",
            runner="nnpgen.md.stage1_manifest:run_build_stage1_manifest",
        ),
        "control": CommandSpec(
            "Run the bounded Stage-1 submission controller.",
            add_args="nnpgen.md.stage1_control:add_stage1_control_arguments",
            runner="nnpgen.md.stage1_control:run_stage1_submit_controlled",
        ),
        "progress": CommandSpec(
            "Summarize controlled Stage-1 progress.",
            add_args="nnpgen.md.stage1_control:add_stage1_progress_arguments",
            runner="nnpgen.md.stage1_control:run_stage1_progress",
        ),
    },
    "dft": {
        "plan": CommandSpec(
            "Build a VASP planning manifest.",
            add_args="nnpgen.dft.plan:add_vasp_plan_arguments",
            runner="nnpgen.dft.plan:run_vasp_plan",
        ),
        "prepare-from-manifest": CommandSpec(
            "Materialize VASP task directories from a manifest.",
            add_args="nnpgen.dft.prepare_vasp_manifest:add_prepare_vasp_from_manifest_arguments",
            runner="nnpgen.dft.prepare_vasp_manifest:run_prepare_vasp_from_manifest",
        ),
        "prepare-seq-pbs": CommandSpec(
            "Build grouped sequential PBS task directories.",
            add_args="nnpgen.dft.pbs_flow:add_prepare_seq_pbs_arguments",
            runner="nnpgen.dft.pbs_flow:run_prepare_seq_pbs_from_manifest",
        ),
        "submit-pbs": CommandSpec(
            "Submit grouped PBS jobs.",
            add_args="nnpgen.dft.pbs_flow:add_submit_pbs_arguments",
            runner="nnpgen.dft.pbs_flow:run_submit_pbs",
        ),
        "collect-results": CommandSpec(
            "Collect completed DFT outputs.",
            add_args="nnpgen.dft.pbs_flow:add_collect_results_arguments",
            runner="nnpgen.dft.pbs_flow:run_collect_results",
        ),
        "convert-dataset": CommandSpec(
            "Convert completed DFT results into DP dataset format.",
            add_args="nnpgen.dft.pbs_flow:add_convert_dataset_arguments",
            runner="nnpgen.dft.pbs_flow:run_convert_dataset",
        ),
        "select": CommandSpec(
            "Select Stage-1 structures for DFT.",
            add_args="nnpgen.dft.stage2:add_stage2_select_arguments",
            runner="nnpgen.dft.stage2:run_stage2_select",
        ),
        "prepare": CommandSpec(
            "Prepare task-per-structure VASP inputs.",
            add_args="nnpgen.dft.stage2:add_stage2_prepare_arguments",
            runner="nnpgen.dft.stage2:run_stage2_prepare",
        ),
        "submit": CommandSpec(
            "Submit task-per-structure VASP jobs.",
            add_args="nnpgen.dft.stage2:add_stage2_submit_arguments",
            runner="nnpgen.dft.stage2:run_stage2_submit",
        ),
        "status": CommandSpec(
            "Refresh and summarize DFT statuses.",
            add_args="nnpgen.dft.stage2:add_stage2_status_arguments",
            runner="nnpgen.dft.stage2:run_stage2_status",
        ),
        "completed-to-dpdata": CommandSpec(
            "Convert completed controller runs into DP data.",
            add_args="nnpgen.dft.completed_to_dpdata:add_stage2_completed_to_dpdata_arguments",
            runner="nnpgen.dft.completed_to_dpdata:run_stage2_completed_to_dpdata",
        ),
        "controller": CommandSpec(
            "Run the DFT controller.",
            add_args="nnpgen.dft.controller:add_controller_arguments",
            runner="nnpgen.dft.controller:run_controller",
        ),
        "controller-status": CommandSpec(
            "Summarize a DFT controller.",
            add_args="nnpgen.dft.controller:add_controller_status_arguments",
            runner="nnpgen.dft.controller:run_controller_status",
        ),
        "archive": CommandSpec(
            "Scan and stage terminal remote DFT frames.",
            add_args="nnpgen.dft.archive:add_archive_arguments",
            runner="nnpgen.dft.archive:run_archive",
        ),
        "recover": CommandSpec(
            "Plan or apply safe recovery after a scheduler outage.",
            add_args="nnpgen.dft.recovery:add_recovery_arguments",
            runner="nnpgen.dft.recovery:run_recovery",
        ),
    },
    "dataset": {
        "dpdata-to-extxyz": CommandSpec("Convert DP data to Extended XYZ.", module="nnpgen.dataset.dpdata_to_extxyz"),
        "vasp-to-dpdata": CommandSpec("Convert VASP OUTCAR/POSCAR outputs to DP data.", module="nnpgen.dataset.vasp_sp2dpdata"),
        "validate-dpdata": CommandSpec(
            "Validate RAW, NPY, set.000, and type metadata.",
            add_args="nnpgen.dataset.validation:add_validation_arguments",
            runner="nnpgen.dataset.validation:run_validation",
        ),
        "split-train-test": CommandSpec("Split DP datasets into train/test roots.", module="nnpgen.dataset.split_train_test"),
        "check-run-systems": CommandSpec("Validate DeepMD run.json system paths.", module="nnpgen.dataset.check_run_systems"),
        "build-exclude-list": CommandSpec("Build explicit DFT exclusion keys.", module="nnpgen.dataset.build_exclude_list"),
        "inspect": CommandSpec(
            "Inspect DPData/NPY groups and optionally write a manifest.",
            add_args="nnpgen.dataset.inspect:add_inspect_arguments",
            runner="nnpgen.dataset.inspect:run_inspect",
        ),
        "build-manifest": CommandSpec(
            "Build a deterministic DPData/NPY TSV manifest.",
            add_args="nnpgen.dataset.inspect:add_manifest_arguments",
            runner="nnpgen.dataset.inspect:run_build_manifest",
        ),
        "convert-npy-to-lmdb": CommandSpec(
            "Convert a DPData/NPY manifest to DPA4C-compatible LMDB.",
            add_args="nnpgen.dataset.lmdb:add_convert_arguments",
            runner="nnpgen.dataset.lmdb:run_convert",
        ),
        "validate-lmdb": CommandSpec(
            "Validate DPA4C-compatible LMDB metadata and frames.",
            add_args="nnpgen.dataset.lmdb:add_validate_arguments",
            runner="nnpgen.dataset.lmdb:run_validate",
        ),
        "compare-npy-lmdb": CommandSpec(
            "Compare source NPY frames with LMDB frames.",
            add_args="nnpgen.dataset.lmdb:add_compare_arguments",
            runner="nnpgen.dataset.lmdb:run_compare",
        ),
    },
    "geo": {
        "build-sio2-nanobubble": CommandSpec(
            "Build an alpha-SiO2 slab plus hemispherical N2 nanobubble solution model.",
            module="nnpgen.geo.sio2_nanobubble",
        ),
    },
    "train": {
        "fill-finetune-systems": CommandSpec("Fill DeepMD fine-tuning systems arrays.", module="nnpgen.train.finetune"),
        "predict-poscar": CommandSpec("Predict POSCAR structures with DeepMD and export DP data.", module="nnpgen.train.predict_poscar"),
        "dp-test": CommandSpec(
            "Run and audit DeePMD-kit's native dp test command.",
            add_args="nnpgen.train.deepmd_test:add_arguments",
            runner="nnpgen.train.deepmd_test:run",
        ),
        "benchmark-predict": CommandSpec(
            "Run DeepMD predictions over DP test data.",
            add_args="nnpgen.train.benchmark_predict:add_arguments",
            runner="nnpgen.train.benchmark_predict:run",
        ),
        "benchmark-plot": CommandSpec(
            "Plot parity metrics from benchmark arrays.",
            add_args="nnpgen.train.benchmark_plot:add_arguments",
            runner="nnpgen.train.benchmark_plot:run",
        ),
        "sync-dataset": CommandSpec("Sync train/test datasets and optional training inputs.", module="nnpgen.train.sync"),
    },
    "monitor": {
        "watchdog": CommandSpec("Run the controller watchdog.", module="nnpgen.monitor.watchdog"),
        "summary": CommandSpec("Summarize manifests, frame runs, and XYZ coverage.", module="nnpgen.monitor.summary"),
        "schedulers": CommandSpec(
            "Summarize PBS and Slurm jobs for explicit run roots.",
            add_args="nnpgen.monitor.schedulers:add_scheduler_arguments",
            runner="nnpgen.monitor.schedulers:run_scheduler_summary",
        ),
    },
}


def _top_help() -> str:
    lines = [
        "usage: nnpgen [--config CONFIG] <domain> <command> [args...]",
        "",
        "Neural network potential generation workflows.",
        "",
        "domains:",
    ]
    for domain in sorted(COMMANDS):
        lines.append(f"  {domain:<8} {len(COMMANDS[domain])} commands")
    lines.extend(["", "Use 'nnpgen <domain> --help' for domain commands."])
    return "\n".join(lines)


def _domain_help(domain: str) -> str:
    specs = COMMANDS[domain]
    lines = [f"usage: nnpgen {domain} <command> [args...]", "", "commands:"]
    for name in sorted(specs):
        lines.append(f"  {name:<22} {specs[name].help}")
    return "\n".join(lines)


def _dispatch(domain: str, command: str, argv: List[str]) -> None:
    try:
        spec = COMMANDS[domain][command]
    except KeyError:
        raise SystemExit(f"Unknown command: nnpgen {domain} {command}")

    if spec.module:
        _module_runner(spec.module, list(spec.argv_prefix) + argv)
        return
    if spec.add_args and spec.runner:
        _function_runner(
            spec.add_args,
            spec.runner,
            argv,
            prog=f"nnpgen {domain} {command}",
            runner_kwargs=spec.runner_kwargs,
        )
        return
    raise SystemExit(f"Command is not wired: nnpgen {domain} {command}")


def main(argv: Optional[List[str]] = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    config_path = ""
    rest: List[str] = []
    i = 0
    while i < len(raw):
        item = raw[i]
        if item == "--config":
            if i + 1 >= len(raw):
                raise SystemExit("--config requires a path")
            config_path = raw[i + 1]
            i += 2
            continue
        if item.startswith("--config="):
            config_path = item.split("=", 1)[1]
            i += 1
            continue
        rest = raw[i:]
        break

    if not rest or rest[0] in {"-h", "--help"}:
        print(_top_help())
        return

    apply_config_environment(config_path or None)

    domain = rest[0]
    if domain == "config":
        if len(rest) >= 2 and rest[1] == "show":
            cfg = load_config(config_path or None)
            print(json.dumps(cfg.as_env(), indent=2, sort_keys=True))
            return
        raise SystemExit("usage: nnpgen [--config CONFIG] config show")

    if domain not in COMMANDS:
        raise SystemExit(f"Unknown domain: {domain}")

    if len(rest) == 1 or rest[1] in {"-h", "--help"}:
        print(_domain_help(domain))
        return

    _dispatch(domain, rest[1], rest[2:])


if __name__ == "__main__":
    main()
