# nnpgen

`nnpgen` is a portable Python toolkit for neural-network-potential workflows.
It organizes structure preparation, ASE/MACE sampling, VASP task generation,
dataset conversion, training helpers, and monitoring into reusable command
groups.

The repository is source-only: checkpoints, generated datasets, private
manifests, credentials, server aliases, IP addresses, and site-specific
absolute paths are not part of the project.

## Installation

Use Python 3.9 or newer in a virtual environment or conda environment:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install only the optional capabilities you need:

```bash
python -m pip install -e ".[dev]"       # tests and local development
python -m pip install -e ".[md,geo]"     # ASE/MACE and geometry builders
python -m pip install -e ".[dataset]"    # NumPy-backed dataset utilities
python -m pip install -e ".[plot]"       # benchmark plots
```

The `deepmd` extra installs DeepMD-kit, which is intentionally not a base
dependency because it is environment- and accelerator-specific.

## Command groups

```bash
nnpgen --help
nnpgen md --help
nnpgen dft --help
nnpgen dataset --help
nnpgen geo --help
nnpgen train --help
nnpgen monitor --help
```

- `md`: prepare, submit, monitor, and post-process MD frames.
- `dft`: plan, prepare, submit, monitor, archive, recover, and convert DFT data.
- `dataset`: validate and convert DP, EXTXYZ, and VASP-derived datasets.
- `geo`: build reusable structure/solvent geometries, including the SiO₂
  nanobubble builder.
- `train`: prepare fine-tuning inputs and benchmark predictions.
- `monitor`: summarize workflow state or run the optional watchdog.

## Portable configuration

Public defaults use the current working directory. Supply real paths, model
locations, scheduler hosts, and environments through explicit arguments,
environment variables, or a copied TOML file:

```bash
cp examples/configs/cluster.example.toml my-workflow.toml
# Edit my-workflow.toml with paths for your own environment.
nnpgen --config my-workflow.toml config show
```

Never commit a config containing credentials or organization-specific host
details. Keep those values in a private, ignored file or in environment
variables.

## Generic DFT operations

Remote targets are named at invocation time, so the same manifest can move
between clusters without code edits:

```bash
nnpgen dft archive \
  --manifest plans/dft_manifest.json \
  --remote primary=login.example:/work/project/dft \
  --output-root datasets/archive \
  --dry-run

nnpgen dft recover \
  --manifest plans/dft_manifest.json \
  --remote primary=login.example:/work/project/dft \
  --backend primary=slurm
# Add --apply only after reviewing the recovery report.
```

`archive` requires finished markers and `OUTCAR` by default, records pending
and failed frames, and stages successful frames without deleting source data.
Add `--convert` to invoke the package's VASP-to-DP-data converter after a
successful staging pass.
`recover` preserves active jobs and explicit failures; it only changes a
manifest when `--apply` is supplied. Use `--allow-lost-target NAME` only after
confirming that a target's scheduler is genuinely unavailable.

For campaign-level status:

```bash
nnpgen monitor summary \
  --manifest plans/dft_manifest.json \
  --run-root runs/stage1 \
  --xyz datasets/samples.xyz \
  --output reports/status.json
```

## Development and hygiene checks

```bash
python -m pip install -e ".[dev]"
pytest
python -m compileall src
```

Before publishing, verify that generated artifacts and private values are
absent from the Git-tracked files:

```bash
git ls-files | rg '(__pycache__|\.pyc$|\.bak|OUTCAR$|CONTCAR$|\.npy$|\.npz$)'
rg -n -i '(/home/|/data/|@[^ ]+:[^ ]+|token|password|secret)' $(git ls-files)
```

The second check is a review aid; inspect matches rather than blindly deleting
generic documentation examples.

## Extension rules

New functionality should live under a domain package (`geo`, `md`, `dft`,
`dataset`, `train`, or `monitor`) with a small CLI adapter, explicit paths and
parameters, deterministic summaries, and tests for pure logic. Keep cluster
integration behind arguments/configuration, use descriptive generic names,
and do not copy timestamped backups or one-off campaign scripts into `src/`.
