# nnpgen

`nnpgen` is a Python workflow toolkit for neural network potential generation. It connects structure preparation, ASE/MACE molecular dynamics sampling, VASP DFT task generation and monitoring, DP-data conversion, and DeepMD/DPA-style fine-tuning utilities.

The repository is intentionally source-only: no model checkpoints, VASP outputs, generated DP arrays, private run manifests, or site-specific paths are committed.

## Install

```bash
python -m pip install -e .
```

Feature extras are optional:

```bash
python -m pip install -e ".[md,dataset,plot,dev]"
```

Use a Python 3.9+ environment. Cluster system Python installations can be older; activate a suitable conda or virtual environment before installing.

## CLI

```bash
nnpgen --help
nnpgen md --help
nnpgen dft --help
nnpgen dataset --help
nnpgen train --help
nnpgen monitor --help
```

Main command groups:

- `nnpgen md`: prepare, submit, monitor, and post-process ASE/MACE MD jobs.
- `nnpgen dft`: plan, prepare, submit, monitor, collect, and convert VASP DFT jobs.
- `nnpgen dataset`: validate and convert DP, EXTXYZ, and VASP-derived datasets.
- `nnpgen train`: prepare fine-tuning inputs and run DeepMD/DPA benchmark helpers.
- `nnpgen monitor`: run controller watchdog utilities.

## Configuration

Public defaults are portable. Real project roots, remote roots, conda environments, model paths, and scheduler hosts should be supplied through:

1. explicit CLI arguments,
2. environment variables,
3. a TOML file passed with `--config`,
4. built-in defaults.

Example:

```bash
nnpgen --config examples/configs/server11_server15.example.toml config show
nnpgen --config examples/configs/server11_server15.example.toml md prepare --help
```

Copy an example config and replace placeholder paths with your own cluster paths before running jobs.

## Development Checks

```bash
python -m pip install -e ".[dev]"
pytest
python -m compileall src
```

Before publishing, also check that private paths and generated files are absent:

```bash
find . -path './.git' -prune -o -name '__pycache__' -o -name '*.pyc' -o -name '*.bak*' -print
grep -RIn 'replace-with-your-own-cluster-path' .
```
