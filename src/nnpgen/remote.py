"""Small subprocess wrappers for local and SSH commands."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class CommandResult:
    args: List[str]
    stdout: str
    stderr: str
    returncode: int


def run_command(args: List[str], cwd: Optional[Path] = None, check: bool = True) -> CommandResult:
    proc = subprocess.run(
        args,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, args, output=proc.stdout, stderr=proc.stderr)
    return CommandResult(list(args), proc.stdout, proc.stderr, proc.returncode)


def ssh(host: str, command: str, extra_args: Optional[List[str]] = None) -> str:
    if not host:
        raise ValueError("SSH host is required")
    args = ["ssh"] + list(extra_args or []) + [host, command]
    return run_command(args).stdout


def scp_to(local_path: Path, host: str, remote_path: str, extra_args: Optional[List[str]] = None) -> None:
    if not host:
        raise ValueError("SCP host is required")
    run_command(["scp"] + list(extra_args or []) + [str(local_path), f"{host}:{remote_path}"])


def scp_from(host: str, remote_path: str, local_path: Path, extra_args: Optional[List[str]] = None) -> None:
    if not host:
        raise ValueError("SCP host is required")
    run_command(["scp"] + list(extra_args or []) + [f"{host}:{remote_path}", str(local_path)])
