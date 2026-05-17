import subprocess
import sys


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "nnpgen", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_top_level_help():
    result = run_cli("--help")
    assert "nnpgen [--config CONFIG]" in result.stdout
    assert "md" in result.stdout
    assert "dft" in result.stdout


def test_domain_help_is_lightweight():
    for domain in ["md", "dft", "dataset", "train", "monitor"]:
        result = run_cli(domain, "--help")
        assert f"nnpgen {domain}" in result.stdout
        assert "commands:" in result.stdout
