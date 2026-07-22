from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def candidate_files():
    if (ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        return [ROOT / line for line in result.stdout.splitlines() if line.strip()]
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(ROOT).parts
        and ".venv" not in path.relative_to(ROOT).parts
        and ".pytest_cache" not in path.relative_to(ROOT).parts
        and "__pycache__" not in path.relative_to(ROOT).parts
    ]


def test_no_generated_artifacts_committed():
    forbidden_suffixes = {".pyc", ".npy", ".npz"}
    forbidden_names = {"__pycache__", "OUTCAR", "CONTCAR", "XDATCAR", "vasprun.xml", "WAVECAR", "CHGCAR"}
    offenders = []
    for path in candidate_files():
        rel = path.relative_to(ROOT)
        if path.name in forbidden_names or path.suffix in forbidden_suffixes or ".bak" in path.name:
            offenders.append(str(rel))
    assert offenders == []


def test_no_private_paths_or_tokens_in_sources():
    private_patterns = [
        "/" + "home/" + "peng" + "chao",
        "/" + "data/HOME_BACKUP/" + "peng" + "chao",
        "/" + "home/" + "xu" + "xf",
        "/" + "home/" + "thu-xu" + "xuefei",
        "github_" + "to" + "ken",
        "ssh " + "11",
        "ssh " + "15",
        "ssh " + "27",
        "101" + ".6.61.27",
    ]
    offenders = []
    for path in candidate_files():
        rel = path.relative_to(ROOT)
        if path.suffix not in {".py", ".md", ".toml", ".sbatch", ".txt"} and path.name not in {"README.md"}:
            continue
        text = path.read_text(errors="ignore")
        for pattern in private_patterns:
            if pattern in text:
                offenders.append(f"{rel}: {pattern}")
    assert offenders == []
