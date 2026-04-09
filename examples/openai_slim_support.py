"""Shared bootstrap, environment, and Slim runtime helpers for examples."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from collections.abc import Sequence
from pathlib import Path

ALLOWED_ENV_VARS: dict[str, str] = {
    "OPENAI_API_KEY": "",
    "SLIM_PLUGIN_ID": "",
    "SLIM_BINARY_PATH": "",
    "SLIM_PROVIDER": "openai",
    "SLIM_PLUGIN_FOLDER": "../../.slim/plugins",
    "SLIM_PLUGIN_ROOT": "",
}
PATH_ENV_VARS = {
    "SLIM_BINARY_PATH",
    "SLIM_PLUGIN_FOLDER",
    "SLIM_PLUGIN_ROOT",
}
BOOTSTRAP_ENV_VAR = "GRAPHON_EXAMPLE_BOOTSTRAPPED"
RUNTIME_MODULES = ("pydantic", "httpx", "yaml")
MIN_QUOTED_VALUE_LENGTH = 2


def repo_root_for(example_file: Path) -> Path:
    return example_file.resolve().parents[2]


def local_src_dir_for(example_file: Path) -> Path:
    return repo_root_for(example_file) / "src"


def local_venv_python_for(example_file: Path) -> Path:
    return repo_root_for(example_file) / ".venv" / "bin" / "python"


def prepare_example_imports(
    example_file: Path,
    *,
    argv: Sequence[str] | None = None,
) -> None:
    bootstrap_local_python(example_file, argv=argv)
    ensure_local_src_on_path(example_file)


def bootstrap_local_python(
    example_file: Path,
    *,
    argv: Sequence[str] | None = None,
) -> None:
    if os.environ.get(BOOTSTRAP_ENV_VAR) == "1":
        return
    if all(importlib.util.find_spec(module) is not None for module in RUNTIME_MODULES):
        return

    local_python = local_venv_python_for(example_file)
    if not local_python.is_file():
        return

    env = dict(os.environ)
    env[BOOTSTRAP_ENV_VAR] = "1"
    os.execve(  # noqa: S606
        str(local_python),
        [
            str(local_python),
            str(example_file.resolve()),
            *(argv if argv is not None else sys.argv[1:]),
        ],
        env,
    )


def ensure_local_src_on_path(example_file: Path) -> None:
    local_src_dir = local_src_dir_for(example_file)
    if (
        importlib.util.find_spec("graphon") is None
        and str(local_src_dir) not in sys.path
    ):
        sys.path.insert(0, str(local_src_dir))


def load_default_env_file(example_dir: Path) -> None:
    env_file = example_dir / ".env"
    if env_file.is_file():
        load_env_file(env_file)


def load_env_file(path: Path) -> None:
    env_dir = path.resolve().parent
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            msg = f"Invalid .env line {line_number} in {path}: {raw_line}"
            raise ValueError(msg)

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            msg = f"Invalid .env key on line {line_number} in {path}"
            raise ValueError(msg)
        if key not in ALLOWED_ENV_VARS:
            msg = f"Unsupported .env key {key!r} on line {line_number} in {path}"
            raise ValueError(msg)

        os.environ.setdefault(
            key,
            normalize_env_value(
                key,
                strip_optional_quotes(value.strip()),
                base_dir=env_dir,
            ),
        )


def strip_optional_quotes(value: str) -> str:
    if (
        len(value) >= MIN_QUOTED_VALUE_LENGTH
        and value[0] == value[-1]
        and value[0] in {'"', "'"}
    ):
        return value[1:-1]
    return value


def normalize_env_value(name: str, value: str, *, base_dir: Path) -> str:
    if name not in PATH_ENV_VARS or not value:
        return value

    path_value = Path(value).expanduser()
    if not path_value.is_absolute():
        path_value = (base_dir / path_value).resolve()
    else:
        path_value = path_value.resolve()
    return str(path_value)


def env_value(name: str, *, example_dir: Path) -> str:
    raw_value = os.environ.get(name)
    if raw_value is not None:
        return raw_value.strip()
    return normalize_env_value(
        name,
        ALLOWED_ENV_VARS[name],
        base_dir=example_dir,
    ).strip()


def require_env(name: str, *, example_dir: Path) -> str:
    value = env_value(name, example_dir=example_dir)
    if value:
        return value
    msg = f"{name} is required."
    raise ValueError(msg)


def optional_path(name: str, *, example_dir: Path) -> Path | None:
    value = env_value(name, example_dir=example_dir)
    return Path(value).expanduser() if value else None


def build_runtime(*, example_dir: Path) -> tuple[object, str]:
    slim_module = importlib.import_module("graphon.model_runtime.slim")
    slim_config = slim_module.SlimConfig
    slim_local_settings = slim_module.SlimLocalSettings
    slim_provider_binding = slim_module.SlimProviderBinding
    slim_runtime = slim_module.SlimRuntime

    provider = env_value("SLIM_PROVIDER", example_dir=example_dir)
    plugin_folder = Path(
        env_value("SLIM_PLUGIN_FOLDER", example_dir=example_dir),
    ).expanduser()
    plugin_root = optional_path("SLIM_PLUGIN_ROOT", example_dir=example_dir)

    runtime = slim_runtime(
        slim_config(
            bindings=[
                slim_provider_binding(
                    plugin_id=require_env("SLIM_PLUGIN_ID", example_dir=example_dir),
                    provider=provider,
                    plugin_root=plugin_root,
                ),
            ],
            local=slim_local_settings(folder=plugin_folder),
        ),
    )
    return runtime, provider


class PassthroughPromptMessageSerializer:
    def serialize(
        self,
        *,
        model_mode: object,
        prompt_messages: Sequence[object],
    ) -> object:
        _ = model_mode
        return list(prompt_messages)


class TextOnlyFileSaver:
    def save_binary_string(
        self,
        data: bytes,
        mime_type: str,
        file_type: object,
        extension_override: str | None = None,
    ) -> object:
        _ = data, mime_type, file_type, extension_override
        msg = "This example only supports text responses."
        raise RuntimeError(msg)

    def save_remote_url(self, url: str, file_type: object) -> object:
        _ = url, file_type
        msg = "This example only supports text responses."
        raise RuntimeError(msg)
