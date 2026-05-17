from pathlib import Path

from nnpgen.config import apply_config_environment, load_config


def test_config_precedence_cli_env_file_defaults(tmp_path):
    cfg_path = tmp_path / "nnpgen.toml"
    cfg_path.write_text(
        "\n".join(
            [
                "[paths]",
                f"project_root = {str(tmp_path / 'from_file')!r}",
                "[md]",
                "conda_env = '/env/from-file'",
            ]
        )
        + "\n"
    )

    env = {"NNPGEN_CONDA_ENV": "/env/from-env"}
    cfg = load_config(
        cfg_path,
        cli_overrides={"project_root": tmp_path / "from_cli"},
        env=env,
    )

    assert cfg.project_root == tmp_path / "from_cli"
    assert cfg.conda_env == "/env/from-env"
    assert cfg.run_root.name == "runs"


def test_apply_config_environment_does_not_override_existing_env(tmp_path):
    cfg_path = tmp_path / "nnpgen.toml"
    cfg_path.write_text("[remote]\ndefault_host = 'pbs-cluster'\n")
    env = {"NNPGEN_DEFAULT_REMOTE_HOST": "already-set"}

    applied = apply_config_environment(cfg_path, env=env)

    assert env["NNPGEN_DEFAULT_REMOTE_HOST"] == "already-set"
    assert "NNPGEN_DEFAULT_REMOTE_HOST" not in applied


def test_apply_config_environment_sets_missing_values(tmp_path):
    cfg_path = tmp_path / "nnpgen.toml"
    project_root = tmp_path / "project"
    cfg_path.write_text(f"[paths]\nproject_root = {str(project_root)!r}\n")
    env = {}

    apply_config_environment(cfg_path, env=env)

    assert env["NNPGEN_PROJECT_ROOT"] == str(project_root)
