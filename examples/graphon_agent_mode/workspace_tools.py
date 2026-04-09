"""Workspace-local tool runtime used by the agent example."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphon.model_runtime.entities.llm_entities import LLMUsage
from graphon.nodes.runtime import ToolNodeRuntimeProtocol
from graphon.nodes.tool.entities import ToolNodeData
from graphon.nodes.tool_runtime_entities import (
    ToolRuntimeHandle,
    ToolRuntimeMessage,
    ToolRuntimeParameter,
)


@dataclass(frozen=True, slots=True)
class WorkspaceToolSettings:
    workspace_root: Path
    shell: str = "/bin/zsh"
    command_timeout_seconds: int = 20
    max_result_chars: int = 12000


class WorkspaceToolRuntime(ToolNodeRuntimeProtocol):
    def __init__(self, settings: WorkspaceToolSettings) -> None:
        self._settings = settings

    def get_runtime(
        self,
        *,
        node_id: str,
        node_data: ToolNodeData,
        variable_pool: object | None,
    ) -> ToolRuntimeHandle:
        _ = node_id, node_data, variable_pool
        return ToolRuntimeHandle(raw=self._settings)

    def get_runtime_parameters(
        self,
        *,
        tool_runtime: ToolRuntimeHandle,
    ) -> Sequence[ToolRuntimeParameter]:
        _ = tool_runtime
        return (
            ToolRuntimeParameter("tool_name"),
            ToolRuntimeParameter("path"),
            ToolRuntimeParameter("content"),
            ToolRuntimeParameter("command"),
        )

    def invoke(
        self,
        *,
        tool_runtime: ToolRuntimeHandle,
        tool_parameters: Mapping[str, Any],
        workflow_call_depth: int,
        provider_name: str,
    ) -> Generator[ToolRuntimeMessage, None, None]:
        _ = workflow_call_depth, provider_name
        settings = self._coerce_settings(tool_runtime)
        result = self._dispatch(
            settings=settings,
            tool_name=str(tool_parameters.get("tool_name", "")).strip(),
            path=str(tool_parameters.get("path", "")).strip(),
            content=str(tool_parameters.get("content", "")),
            command=str(tool_parameters.get("command", "")).strip(),
        )
        result_text = json.dumps(result, indent=2, ensure_ascii=False)
        yield ToolRuntimeMessage(
            type=ToolRuntimeMessage.MessageType.TEXT,
            message=ToolRuntimeMessage.TextMessage(text=result_text),
        )
        yield ToolRuntimeMessage(
            type=ToolRuntimeMessage.MessageType.VARIABLE,
            message=ToolRuntimeMessage.VariableMessage(
                variable_name="result",
                variable_value=result_text,
                stream=False,
            ),
        )

    def get_usage(
        self,
        *,
        tool_runtime: ToolRuntimeHandle,
    ) -> LLMUsage:
        _ = tool_runtime
        return LLMUsage.empty_usage()

    def build_file_reference(self, *, mapping: Mapping[str, Any]) -> Any:
        _ = mapping
        msg = "File outputs are not supported by this example tool runtime."
        raise NotImplementedError(msg)

    def _dispatch(
        self,
        *,
        settings: WorkspaceToolSettings,
        tool_name: str,
        path: str,
        content: str,
        command: str,
    ) -> dict[str, Any]:
        try:
            match tool_name:
                case "read_file":
                    return self._read_file(settings=settings, raw_path=path)
                case "write_file":
                    return self._write_file(
                        settings=settings,
                        raw_path=path,
                        content=content,
                    )
                case "delete_file":
                    return self._delete_file(settings=settings, raw_path=path)
                case "run_bash":
                    return self._run_bash(settings=settings, command=command)
                case _:
                    return {
                        "status": "error",
                        "tool_name": tool_name,
                        "message": (
                            "Unsupported tool. Use read_file, write_file, "
                            "delete_file, or run_bash."
                        ),
                    }
        except (OSError, TypeError, ValueError, subprocess.SubprocessError) as exc:
            return {
                "status": "error",
                "tool_name": tool_name,
                "message": str(exc),
            }

    @staticmethod
    def _coerce_settings(tool_runtime: ToolRuntimeHandle) -> WorkspaceToolSettings:
        settings = tool_runtime.raw
        if not isinstance(settings, WorkspaceToolSettings):
            msg = f"Unexpected tool runtime payload: {type(settings).__name__}"
            raise TypeError(msg)
        return settings

    def _read_file(
        self,
        *,
        settings: WorkspaceToolSettings,
        raw_path: str,
    ) -> dict[str, Any]:
        path = self._resolve_path(settings=settings, raw_path=raw_path)
        if not path.is_file():
            msg = f"{path.relative_to(settings.workspace_root)} is not a file."
            raise FileNotFoundError(msg)

        content = path.read_text(encoding="utf-8", errors="replace")
        truncated_content, truncated = self._truncate_text(
            content,
            max_chars=settings.max_result_chars,
        )
        return {
            "status": "ok",
            "tool_name": "read_file",
            "path": str(path.relative_to(settings.workspace_root)),
            "truncated": truncated,
            "content": truncated_content,
        }

    def _write_file(
        self,
        *,
        settings: WorkspaceToolSettings,
        raw_path: str,
        content: str,
    ) -> dict[str, Any]:
        path = self._resolve_path(settings=settings, raw_path=raw_path)
        if path.exists() and path.is_dir():
            msg = f"{path.relative_to(settings.workspace_root)} is a directory."
            raise IsADirectoryError(msg)

        created = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "status": "ok",
            "tool_name": "write_file",
            "path": str(path.relative_to(settings.workspace_root)),
            "created": created,
            "bytes_written": len(content.encode("utf-8")),
        }

    def _delete_file(
        self,
        *,
        settings: WorkspaceToolSettings,
        raw_path: str,
    ) -> dict[str, Any]:
        path = self._resolve_path(settings=settings, raw_path=raw_path)
        if not path.exists():
            msg = f"{path.relative_to(settings.workspace_root)} does not exist."
            raise FileNotFoundError(msg)
        if path.is_dir():
            msg = f"{path.relative_to(settings.workspace_root)} is a directory."
            raise IsADirectoryError(msg)

        path.unlink()
        return {
            "status": "ok",
            "tool_name": "delete_file",
            "path": str(path.relative_to(settings.workspace_root)),
        }

    def _run_bash(
        self,
        *,
        settings: WorkspaceToolSettings,
        command: str,
    ) -> dict[str, Any]:
        if not command:
            msg = "run_bash requires a non-empty command."
            raise ValueError(msg)

        completed = subprocess.run(
            [settings.shell, "-lc", command],
            cwd=settings.workspace_root,
            capture_output=True,
            text=True,
            timeout=settings.command_timeout_seconds,
            check=False,
        )
        stdout, stdout_truncated = self._truncate_text(
            completed.stdout,
            max_chars=settings.max_result_chars,
        )
        stderr, stderr_truncated = self._truncate_text(
            completed.stderr,
            max_chars=settings.max_result_chars,
        )
        return {
            "status": "ok" if completed.returncode == 0 else "error",
            "tool_name": "run_bash",
            "command": command,
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }

    @staticmethod
    def _truncate_text(text: str, *, max_chars: int) -> tuple[str, bool]:
        if len(text) <= max_chars:
            return text, False
        suffix = "\n...[truncated]"
        return text[:max_chars] + suffix, True

    @staticmethod
    def _ensure_inside_workspace(path: Path, workspace_root: Path) -> None:
        if path == workspace_root or workspace_root in path.parents:
            return
        msg = "Path must stay inside the workspace root."
        raise ValueError(msg)

    def _resolve_path(
        self,
        *,
        settings: WorkspaceToolSettings,
        raw_path: str,
    ) -> Path:
        if not raw_path:
            msg = "Tool call requires a non-empty path."
            raise ValueError(msg)

        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (settings.workspace_root / path).resolve()
        else:
            path = path.resolve()
        self._ensure_inside_workspace(path, settings.workspace_root)
        return path


class NoopToolFileManager:
    def create_file_by_raw(
        self,
        *,
        file_binary: bytes,
        mimetype: str,
        filename: str | None = None,
    ) -> Any:
        _ = file_binary, mimetype, filename
        msg = "File creation is not supported by this example."
        raise NotImplementedError(msg)

    def get_file_generator_by_tool_file_id(
        self,
        tool_file_id: str,
    ) -> tuple[Generator | None, Any | None]:
        _ = tool_file_id
        return None, None
