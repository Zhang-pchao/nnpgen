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
    for domain in ["md", "dft", "dataset", "geo", "train", "monitor"]:
        result = run_cli(domain, "--help")
        assert f"nnpgen {domain}" in result.stdout
        assert "commands:" in result.stdout


def test_new_generic_commands_are_listed():
    result = run_cli("dft", "--help")
    assert "archive" in result.stdout
    assert "recover" in result.stdout
    result = run_cli("monitor", "--help")
    assert "summary" in result.stdout
    assert "schedulers" in result.stdout
    result = run_cli("dataset", "--help")
    assert "validate-dpdata" in result.stdout
    assert "inspect" in result.stdout
    assert "build-manifest" in result.stdout
    assert "convert-npy-to-lmdb" in result.stdout
    assert "validate-lmdb" in result.stdout
    assert "compare-npy-lmdb" in result.stdout
