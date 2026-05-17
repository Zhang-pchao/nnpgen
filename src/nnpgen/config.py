"""Configuration defaults and TOML/env loading for nnpgen.

The package defaults are intentionally portable. Site-specific paths should be
provided through a TOML config file, environment variables, or explicit CLI
arguments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

try:  # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.9/3.10
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - dependency-free fallback
        tomllib = None  # type: ignore


SCHEMA_VERSION = "nnpgen.config.v1"
PACKAGE_ROOT = Path(__file__).resolve().parent
TEMPLATE_ROOT = PACKAGE_ROOT / "templates"


def _path_from_env(name: str, default: Union[Path, str]) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


def _str_from_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class NnpgenConfig:
    project_root: Path
    run_root: Path
    plan_root: Path
    poscar_pool_root: Path
    model_path: Path
    conda_env: str
    md_template: Path
    post_template: Path
    vasp_root: Path
    vasp_smoke_root: Path
    vasp_run_root: Path
    vasp_incar_template: Path
    vasp_pbs_template: Path
    vasp_potcar_org_dir: Path
    vasp_potcar_cat_dir: Path
    default_remote_host: str

    def as_env(self) -> Dict[str, str]:
        return {
            "NNPGEN_PROJECT_ROOT": str(self.project_root),
            "NNPGEN_RUN_ROOT": str(self.run_root),
            "NNPGEN_PLAN_ROOT": str(self.plan_root),
            "NNPGEN_POSCAR_POOL_ROOT": str(self.poscar_pool_root),
            "NNPGEN_MODEL_PATH": str(self.model_path),
            "NNPGEN_CONDA_ENV": self.conda_env,
            "NNPGEN_MD_TEMPLATE": str(self.md_template),
            "NNPGEN_POST_TEMPLATE": str(self.post_template),
            "NNPGEN_VASP_ROOT": str(self.vasp_root),
            "NNPGEN_VASP_SMOKE_ROOT": str(self.vasp_smoke_root),
            "NNPGEN_VASP_RUN_ROOT": str(self.vasp_run_root),
            "NNPGEN_VASP_INCAR_TEMPLATE": str(self.vasp_incar_template),
            "NNPGEN_VASP_PBS_TEMPLATE": str(self.vasp_pbs_template),
            "NNPGEN_VASP_POTCAR_ORG_DIR": str(self.vasp_potcar_org_dir),
            "NNPGEN_VASP_POTCAR_CAT_DIR": str(self.vasp_potcar_cat_dir),
            "NNPGEN_DEFAULT_REMOTE_HOST": self.default_remote_host,
        }


DEFAULT_CONFIG = NnpgenConfig(
    project_root=Path.cwd(),
    run_root=Path.cwd() / "runs",
    plan_root=Path.cwd() / "plans",
    poscar_pool_root=Path.cwd() / "poscar_pool",
    model_path=Path.cwd() / "models" / "mace.model",
    conda_env="",
    md_template=TEMPLATE_ROOT / "run_md_single_frame.sbatch",
    post_template=TEMPLATE_ROOT / "run_post_single_frame.sbatch",
    vasp_root=Path.cwd() / "remote_vasp",
    vasp_smoke_root=Path.cwd() / "remote_vasp" / "smoke",
    vasp_run_root=Path.cwd() / "remote_vasp" / "run",
    vasp_incar_template=Path.cwd() / "remote_vasp" / "template" / "incar" / "INCAR",
    vasp_pbs_template=Path.cwd() / "remote_vasp" / "template" / "script" / "vasp.pbs",
    vasp_potcar_org_dir=Path.cwd() / "remote_vasp" / "template" / "potcar" / "org",
    vasp_potcar_cat_dir=Path.cwd() / "remote_vasp" / "template" / "potcar" / "cat",
    default_remote_host="",
)


ENV_KEYS = {
    "paths.project_root": "NNPGEN_PROJECT_ROOT",
    "paths.run_root": "NNPGEN_RUN_ROOT",
    "paths.plan_root": "NNPGEN_PLAN_ROOT",
    "paths.poscar_pool_root": "NNPGEN_POSCAR_POOL_ROOT",
    "md.model_path": "NNPGEN_MODEL_PATH",
    "md.conda_env": "NNPGEN_CONDA_ENV",
    "md.template": "NNPGEN_MD_TEMPLATE",
    "md.post_template": "NNPGEN_POST_TEMPLATE",
    "dft.vasp_root": "NNPGEN_VASP_ROOT",
    "dft.vasp_smoke_root": "NNPGEN_VASP_SMOKE_ROOT",
    "dft.vasp_run_root": "NNPGEN_VASP_RUN_ROOT",
    "dft.incar_template": "NNPGEN_VASP_INCAR_TEMPLATE",
    "dft.pbs_template": "NNPGEN_VASP_PBS_TEMPLATE",
    "dft.potcar_org_dir": "NNPGEN_VASP_POTCAR_ORG_DIR",
    "dft.potcar_cat_dir": "NNPGEN_VASP_POTCAR_CAT_DIR",
    "remote.default_host": "NNPGEN_DEFAULT_REMOTE_HOST",
}


def _flatten_config(data: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for section, values in data.items():
        if not isinstance(values, Mapping):
            continue
        for key, value in values.items():
            flat[f"{section}.{key}"] = value
    return flat


def _parse_simple_toml(text: str) -> Dict[str, Any]:
    data: Dict[str, Dict[str, Any]] = {}
    section = ""
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            data.setdefault(section, {})
            continue
        if "=" not in line or not section:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        data.setdefault(section, {})[key] = value
    return data


def read_config_file(path: Union[Path, str]) -> Dict[str, Any]:
    config_path = Path(path).expanduser()
    if tomllib is not None:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    else:
        data = _parse_simple_toml(config_path.read_text())
    return _flatten_config(data)


def apply_config_environment(
    config_path: Optional[Union[Path, str]] = None,
    env: MutableMapping[str, str] | None = None,
) -> Dict[str, str]:
    """Apply TOML config values to env without overriding existing variables."""

    target_env = env if env is not None else os.environ
    applied: Dict[str, str] = {}
    if not config_path:
        return applied

    flat = read_config_file(config_path)
    for key, env_name in ENV_KEYS.items():
        if env_name in target_env:
            continue
        if key in flat and flat[key] is not None:
            value = str(flat[key])
            target_env[env_name] = value
            applied[env_name] = value
    return applied


def load_config(
    config_path: Optional[Union[Path, str]] = None,
    cli_overrides: Optional[Mapping[str, Any]] = None,
    env: Mapping[str, str] | None = None,
) -> NnpgenConfig:
    """Load config with precedence: CLI > environment > TOML file > defaults."""

    source_env: Dict[str, str] = dict(env or os.environ)
    if config_path:
        flat = read_config_file(config_path)
        for key, env_name in ENV_KEYS.items():
            if env_name not in source_env and key in flat and flat[key] is not None:
                source_env[env_name] = str(flat[key])

    overrides = dict(cli_overrides or {})

    def pick_path(attr: str, env_name: str, default: Path) -> Path:
        if attr in overrides and overrides[attr] is not None:
            return Path(str(overrides[attr])).expanduser()
        return Path(source_env.get(env_name, str(default))).expanduser()

    def pick_str(attr: str, env_name: str, default: str) -> str:
        if attr in overrides and overrides[attr] is not None:
            return str(overrides[attr])
        return source_env.get(env_name, default)

    return NnpgenConfig(
        project_root=pick_path("project_root", "NNPGEN_PROJECT_ROOT", DEFAULT_CONFIG.project_root),
        run_root=pick_path("run_root", "NNPGEN_RUN_ROOT", DEFAULT_CONFIG.run_root),
        plan_root=pick_path("plan_root", "NNPGEN_PLAN_ROOT", DEFAULT_CONFIG.plan_root),
        poscar_pool_root=pick_path("poscar_pool_root", "NNPGEN_POSCAR_POOL_ROOT", DEFAULT_CONFIG.poscar_pool_root),
        model_path=pick_path("model_path", "NNPGEN_MODEL_PATH", DEFAULT_CONFIG.model_path),
        conda_env=pick_str("conda_env", "NNPGEN_CONDA_ENV", DEFAULT_CONFIG.conda_env),
        md_template=pick_path("md_template", "NNPGEN_MD_TEMPLATE", DEFAULT_CONFIG.md_template),
        post_template=pick_path("post_template", "NNPGEN_POST_TEMPLATE", DEFAULT_CONFIG.post_template),
        vasp_root=pick_path("vasp_root", "NNPGEN_VASP_ROOT", DEFAULT_CONFIG.vasp_root),
        vasp_smoke_root=pick_path("vasp_smoke_root", "NNPGEN_VASP_SMOKE_ROOT", DEFAULT_CONFIG.vasp_smoke_root),
        vasp_run_root=pick_path("vasp_run_root", "NNPGEN_VASP_RUN_ROOT", DEFAULT_CONFIG.vasp_run_root),
        vasp_incar_template=pick_path("vasp_incar_template", "NNPGEN_VASP_INCAR_TEMPLATE", DEFAULT_CONFIG.vasp_incar_template),
        vasp_pbs_template=pick_path("vasp_pbs_template", "NNPGEN_VASP_PBS_TEMPLATE", DEFAULT_CONFIG.vasp_pbs_template),
        vasp_potcar_org_dir=pick_path("vasp_potcar_org_dir", "NNPGEN_VASP_POTCAR_ORG_DIR", DEFAULT_CONFIG.vasp_potcar_org_dir),
        vasp_potcar_cat_dir=pick_path("vasp_potcar_cat_dir", "NNPGEN_VASP_POTCAR_CAT_DIR", DEFAULT_CONFIG.vasp_potcar_cat_dir),
        default_remote_host=pick_str("default_remote_host", "NNPGEN_DEFAULT_REMOTE_HOST", DEFAULT_CONFIG.default_remote_host),
    )


_CONFIG = load_config()

PROJECT_ROOT = _CONFIG.project_root
RUN_ROOT = _CONFIG.run_root
DEFAULT_POSCAR_POOL_ROOT = _CONFIG.poscar_pool_root
DEFAULT_PLAN_ROOT = _CONFIG.plan_root

DEFAULT_MODEL_PATH = _CONFIG.model_path
DEFAULT_CONDA_ENV = _CONFIG.conda_env
DEFAULT_TEMPLATE = _CONFIG.md_template
DEFAULT_POST_TEMPLATE = _CONFIG.post_template

DEFAULT_STEPS = int(_str_from_env("NNPGEN_DEFAULT_STEPS", "100"))
DEFAULT_TEMPERATURE_K = float(_str_from_env("NNPGEN_DEFAULT_TEMPERATURE_K", "300.0"))
DEFAULT_TIMESTEP_FS = float(_str_from_env("NNPGEN_DEFAULT_TIMESTEP_FS", "1.0"))
DEFAULT_FRICTION_PER_FS = float(_str_from_env("NNPGEN_DEFAULT_FRICTION_PER_FS", "0.004"))
DEFAULT_INTERVAL = int(_str_from_env("NNPGEN_DEFAULT_INTERVAL", "20"))
DEFAULT_DEVICE = _str_from_env("NNPGEN_DEFAULT_DEVICE", "gpu")

DEFAULT_FRAME_STRIDE = int(_str_from_env("NNPGEN_DEFAULT_FRAME_STRIDE", "1"))
DEFAULT_FRAME_START = int(_str_from_env("NNPGEN_DEFAULT_FRAME_START", "0"))
DEFAULT_FRAME_MAX = None

DEFAULT_PARTITION = _str_from_env("NNPGEN_DEFAULT_PARTITION", "standard")
DEFAULT_QOS = _str_from_env("NNPGEN_DEFAULT_QOS", "normal")
DEFAULT_GPUS_PER_NODE = _str_from_env("NNPGEN_DEFAULT_GPUS_PER_NODE", "1")
DEFAULT_NODES = int(_str_from_env("NNPGEN_DEFAULT_NODES", "1"))
DEFAULT_NTASKS = int(_str_from_env("NNPGEN_DEFAULT_NTASKS", "4"))
DEFAULT_CPUS_PER_TASK = int(_str_from_env("NNPGEN_DEFAULT_CPUS_PER_TASK", "1"))
DEFAULT_WALLTIME = _str_from_env("NNPGEN_DEFAULT_WALLTIME", "24:00:00")
DEFAULT_OMP_NUM_THREADS = int(_str_from_env("NNPGEN_DEFAULT_OMP_NUM_THREADS", "1"))

DEFAULT_POST_NODES = int(_str_from_env("NNPGEN_DEFAULT_POST_NODES", "1"))
DEFAULT_POST_NTASKS = int(_str_from_env("NNPGEN_DEFAULT_POST_NTASKS", "1"))
DEFAULT_POST_CPUS_PER_TASK = int(_str_from_env("NNPGEN_DEFAULT_POST_CPUS_PER_TASK", "1"))
DEFAULT_POST_WALLTIME = _str_from_env("NNPGEN_DEFAULT_POST_WALLTIME", "02:00:00")
DEFAULT_POST_OMP_NUM_THREADS = int(_str_from_env("NNPGEN_DEFAULT_POST_OMP_NUM_THREADS", "1"))

ELEMENT_ORDER = _str_from_env("NNPGEN_ELEMENT_ORDER", "H,O,N,Na,Cl,Ti,C,Si").split(",")

VASP15_ROOT = _CONFIG.vasp_root
VASP15_SMOKE_ROOT = _CONFIG.vasp_smoke_root
VASP15_RUN_ROOT = _CONFIG.vasp_run_root
VASP15_INCAR_TEMPLATE = _CONFIG.vasp_incar_template
VASP15_PBS_TEMPLATE = _CONFIG.vasp_pbs_template
VASP15_POTCAR_ORG_DIR = _CONFIG.vasp_potcar_org_dir
VASP15_POTCAR_CAT_DIR = _CONFIG.vasp_potcar_cat_dir
DEFAULT_REMOTE_HOST = _CONFIG.default_remote_host

VASP_PLAN_STATUSES = [
    "planned",
    "transferred",
    "prepared",
    "submitted",
    "running",
    "finished",
    "failed",
]
