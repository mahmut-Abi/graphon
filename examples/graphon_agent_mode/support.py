"""Shared runtime helpers for the agent example."""

from __future__ import annotations

# ruff: noqa: E402
import importlib.util
import os
import re
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
LOCAL_SRC_DIR = REPO_ROOT / "src"
DEFAULT_ENV_FILE = EXAMPLE_DIR / ".env"

if importlib.util.find_spec("graphon") is None and str(LOCAL_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC_DIR))

from graphon.file.enums import FileType
from graphon.file.models import File
from graphon.model_runtime.entities.llm_entities import LLMMode
from graphon.model_runtime.entities.message_entities import PromptMessage
from graphon.model_runtime.slim import (
    SlimConfig,
    SlimLocalSettings,
    SlimProviderBinding,
    SlimRuntime,
)
from graphon.model_runtime.slim.package_loader import SlimPackageLoader

from .workspace_tools import WorkspaceToolSettings

ALLOWED_ENV_VARS: dict[str, str] = {
    "OPENAI_API_KEY": "",
    "SLIM_PLUGIN_ID": "",
    "SLIM_BINARY_PATH": "",
    "SLIM_PROVIDER": "openai",
    "SLIM_PLUGIN_FOLDER": "../../.slim/plugins",
    "SLIM_PLUGIN_ROOT": "",
    "SLIM_PYTHON_ENV_INIT_TIMEOUT": "300",
    "SLIM_MAX_EXECUTION_TIMEOUT": "1800",
    "SLIM_OPENAI_CONNECT_TIMEOUT_SECONDS": "30",
    "SLIM_OPENAI_WRITE_TIMEOUT_SECONDS": "30",
    "AGENT_BASH_COMMAND_TIMEOUT_SECONDS": "120",
}
PATH_ENV_VARS = {
    "SLIM_BINARY_PATH",
    "SLIM_PLUGIN_FOLDER",
    "SLIM_PLUGIN_ROOT",
}
INTEGER_ENV_VARS = {
    "SLIM_PYTHON_ENV_INIT_TIMEOUT",
    "SLIM_MAX_EXECUTION_TIMEOUT",
    "SLIM_OPENAI_CONNECT_TIMEOUT_SECONDS",
    "SLIM_OPENAI_WRITE_TIMEOUT_SECONDS",
    "AGENT_BASH_COMMAND_TIMEOUT_SECONDS",
}
OPENAI_TIMEOUT_CALL_PATTERN = re.compile(r'"timeout":\s*Timeout\([^\n]+\),')
OPENAI_PATCHED_PLUGIN_ROOT = REPO_ROOT / ".slim" / "patched_plugins"


def load_default_env_file() -> None:
    if DEFAULT_ENV_FILE.is_file():
        load_env_file(DEFAULT_ENV_FILE)


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
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
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


def env_value(name: str) -> str:
    raw_value = os.environ.get(name)
    if raw_value is not None:
        return raw_value.strip()
    return normalize_env_value(
        name,
        ALLOWED_ENV_VARS[name],
        base_dir=EXAMPLE_DIR,
    ).strip()


def require_env(name: str) -> str:
    value = env_value(name)
    if value:
        return value
    msg = f"{name} is required."
    raise ValueError(msg)


def optional_path(name: str) -> Path | None:
    value = env_value(name)
    return Path(value).expanduser() if value else None


def env_int(name: str, *, minimum: int = 1) -> int:
    if name not in INTEGER_ENV_VARS:
        msg = f"{name} is not configured as an integer environment variable."
        raise ValueError(msg)

    value = env_value(name)
    try:
        parsed = int(value)
    except ValueError as exc:
        msg = f"{name} must be an integer, got {value!r}."
        raise ValueError(msg) from exc

    if parsed < minimum:
        msg = f"{name} must be >= {minimum}, got {parsed}."
        raise ValueError(msg)
    return parsed


def build_local_settings() -> SlimLocalSettings:
    plugin_folder = Path(env_value("SLIM_PLUGIN_FOLDER")).expanduser()
    return SlimLocalSettings(
        folder=plugin_folder,
        python_env_init_timeout=env_int("SLIM_PYTHON_ENV_INIT_TIMEOUT"),
        max_execution_timeout=env_int("SLIM_MAX_EXECUTION_TIMEOUT"),
    )


def build_workspace_tool_settings(*, workspace_root: Path) -> WorkspaceToolSettings:
    return WorkspaceToolSettings(
        workspace_root=workspace_root,
        command_timeout_seconds=env_int("AGENT_BASH_COMMAND_TIMEOUT_SECONDS"),
    )


def patch_openai_timeout_source(
    source_text: str,
    *,
    read_timeout_seconds: int,
    connect_timeout_seconds: int,
    write_timeout_seconds: int,
) -> str:
    default_timeout_seconds = max(
        read_timeout_seconds,
        connect_timeout_seconds,
        write_timeout_seconds,
    )
    replacement = (
        '"timeout": Timeout('
        f"{float(default_timeout_seconds):.1f}, "
        f"read={float(read_timeout_seconds):.1f}, "
        f"write={float(write_timeout_seconds):.1f}, "
        f"connect={float(connect_timeout_seconds):.1f}),"
    )
    patched_text, replacements = OPENAI_TIMEOUT_CALL_PATTERN.subn(
        replacement,
        source_text,
        count=1,
    )
    if replacements != 1:
        msg = "Unable to locate the OpenAI client timeout configuration."
        raise ValueError(msg)
    return patched_text


def resolve_plugin_root(
    *,
    provider: str,
    plugin_id: str,
    configured_plugin_root: Path | None,
    local_settings: SlimLocalSettings,
) -> Path | None:
    if provider != "openai":
        return configured_plugin_root
    return prepare_openai_plugin_root(
        provider=provider,
        plugin_id=plugin_id,
        configured_plugin_root=configured_plugin_root,
        local_settings=local_settings,
    )


def prepare_openai_plugin_root(
    *,
    provider: str,
    plugin_id: str,
    configured_plugin_root: Path | None,
    local_settings: SlimLocalSettings,
) -> Path:
    binding = SlimProviderBinding(
        plugin_id=plugin_id,
        provider=provider,
        plugin_root=configured_plugin_root,
    )
    config = SlimConfig(
        bindings=[binding],
        local=local_settings,
    )
    source_root = SlimPackageLoader(config).load(binding).plugin_root

    source_file = source_root / "models" / "common_openai.py"
    if not source_file.is_file():
        msg = f"OpenAI plugin timeout source file not found: {source_file}"
        raise FileNotFoundError(msg)

    patched_source = patch_openai_timeout_source(
        source_file.read_text(encoding="utf-8"),
        read_timeout_seconds=env_int("SLIM_MAX_EXECUTION_TIMEOUT"),
        connect_timeout_seconds=env_int("SLIM_OPENAI_CONNECT_TIMEOUT_SECONDS"),
        write_timeout_seconds=env_int("SLIM_OPENAI_WRITE_TIMEOUT_SECONDS"),
    )

    patched_root = OPENAI_PATCHED_PLUGIN_ROOT / plugin_id.replace(":", "-")
    if source_root == patched_root:
        source_file.write_text(patched_source, encoding="utf-8")
        return source_root

    target_file = patched_root / "models" / "common_openai.py"
    if patched_root.exists() and target_file.is_file():
        existing_source = target_file.read_text(encoding="utf-8")
        if existing_source == patched_source:
            return patched_root
        shutil.rmtree(patched_root)
    elif patched_root.exists():
        shutil.rmtree(patched_root)

    patched_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, patched_root)
    target_file.write_text(patched_source, encoding="utf-8")
    return patched_root


def build_runtime() -> tuple[SlimRuntime, str]:
    provider = env_value("SLIM_PROVIDER")
    plugin_id = require_env("SLIM_PLUGIN_ID")
    local_settings = build_local_settings()
    plugin_root = resolve_plugin_root(
        provider=provider,
        plugin_id=plugin_id,
        configured_plugin_root=optional_path("SLIM_PLUGIN_ROOT"),
        local_settings=local_settings,
    )

    runtime = SlimRuntime(
        SlimConfig(
            bindings=[
                SlimProviderBinding(
                    plugin_id=plugin_id,
                    provider=provider,
                    plugin_root=plugin_root,
                ),
            ],
            local=local_settings,
        ),
    )
    return runtime, provider


class PassthroughPromptMessageSerializer:
    def serialize(
        self,
        *,
        model_mode: LLMMode,
        prompt_messages: Sequence[PromptMessage],
    ) -> object:
        _ = model_mode
        return list(prompt_messages)


class TextOnlyFileSaver:
    def save_binary_string(
        self,
        data: bytes,
        mime_type: str,
        file_type: FileType,
        extension_override: str | None = None,
    ) -> File:
        _ = data, mime_type, file_type, extension_override
        msg = "This example only supports text responses."
        raise RuntimeError(msg)

    def save_remote_url(self, url: str, file_type: FileType) -> File:
        _ = url, file_type
        msg = "This example only supports text responses."
        raise RuntimeError(msg)
