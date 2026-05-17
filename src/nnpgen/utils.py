import json
import os
import re
from pathlib import Path
from typing import Any, Dict


def abs_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def frame_dir_name(index_1based: int) -> str:
    return f"frame_{index_1based:06d}"


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def write_text(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(content)


def slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def str_to_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v == "true":
        return True
    if v == "false":
        return False
    raise ValueError(f"Invalid boolean string: {value}")
