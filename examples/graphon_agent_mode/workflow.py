"""Graphon agent-mode example powered by a loop and local workspace tools."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, cast

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
LOCAL_SRC_DIR = REPO_ROOT / "src"
LOCAL_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
BOOTSTRAP_ENV_VAR = "GRAPHON_EXAMPLE_BOOTSTRAPPED"
RUNTIME_MODULES = ("pydantic", "httpx", "yaml")


def bootstrap_local_python() -> None:
    if os.environ.get(BOOTSTRAP_ENV_VAR) == "1":
        return
    if all(importlib.util.find_spec(module) is not None for module in RUNTIME_MODULES):
        return
    if not LOCAL_VENV_PYTHON.is_file():
        return

    env = dict(os.environ)
    env[BOOTSTRAP_ENV_VAR] = "1"
    os.execve(
        str(LOCAL_VENV_PYTHON),
        [str(LOCAL_VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        env,
    )


bootstrap_local_python()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if importlib.util.find_spec("graphon") is None and str(LOCAL_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC_DIR))

# ruff: noqa: E402
from examples.graphon_agent_mode.support import (
    REPO_ROOT as WORKSPACE_ROOT,
)
from examples.graphon_agent_mode.support import (
    PassthroughPromptMessageSerializer,
    TextOnlyFileSaver,
    build_runtime,
    build_workspace_tool_settings,
    load_default_env_file,
    require_env,
)
from examples.graphon_agent_mode.workspace_tools import (
    NoopToolFileManager,
    WorkspaceToolRuntime,
)
from graphon.entities.graph_init_params import GraphInitParams
from graphon.enums import BuiltinNodeTypes, WorkflowNodeExecutionMetadataKey
from graphon.graph.edge import Edge
from graphon.graph.graph import Graph
from graphon.graph.validation import get_graph_validator
from graphon.graph_engine.command_channels import InMemoryChannel
from graphon.graph_engine.graph_engine import GraphEngine
from graphon.graph_events.node import NodeRunStartedEvent, NodeRunStreamChunkEvent
from graphon.model_runtime.entities.llm_entities import LLMMode
from graphon.model_runtime.entities.message_entities import PromptMessageRole
from graphon.model_runtime.slim import SlimPreparedLLM
from graphon.nodes.base.entities import OutputVariableEntity, VariableSelector
from graphon.nodes.code.code_node import CodeNode, WorkflowCodeExecutor
from graphon.nodes.code.entities import CodeLanguage, CodeNodeData
from graphon.nodes.code.limits import CodeNodeLimits
from graphon.nodes.end.end_node import EndNode
from graphon.nodes.end.entities import EndNodeData
from graphon.nodes.if_else.entities import IfElseNodeData
from graphon.nodes.if_else.if_else_node import IfElseNode
from graphon.nodes.llm import (
    LLMNode,
    LLMNodeChatModelMessage,
    LLMNodeData,
    ModelConfig,
)
from graphon.nodes.llm.entities import ContextConfig
from graphon.nodes.loop import LoopEndNode, LoopNode, LoopStartNode
from graphon.nodes.loop.entities import (
    LoopEndNodeData,
    LoopNodeData,
    LoopStartNodeData,
    LoopVariableData,
)
from graphon.nodes.start import StartNode
from graphon.nodes.start.entities import StartNodeData
from graphon.nodes.tool.entities import ToolNodeData, ToolProviderType
from graphon.nodes.tool.tool_node import ToolNode
from graphon.nodes.variable_assigner.v2.entities import (
    VariableAssignerNodeData,
    VariableOperationItem,
)
from graphon.nodes.variable_assigner.v2.enums import InputType, Operation
from graphon.nodes.variable_assigner.v2.node import VariableAssignerNode
from graphon.runtime.graph_runtime_state import (
    ChildGraphNotFoundError,
    GraphRuntimeState,
)
from graphon.runtime.variable_pool import VariablePool
from graphon.utils.condition.entities import Condition
from graphon.utils.json_in_md_parser import parse_json_markdown
from graphon.variables.input_entities import VariableEntity, VariableEntityType
from graphon.variables.types import SegmentType

WORKFLOW_ID = "example-agent-mode"
LOOP_NODE_ID = "agent_loop"
LOOP_START_NODE_ID = "loop_start"
PLANNER_NODE_ID = "planner"
PLANNER_DECISION_NODE_ID = "planner_decision"
ROUND_LOG_NODE_ID = "round_log"
ROUTE_ACTION_NODE_ID = "route_action"
WORKSPACE_TOOL_NODE_ID = "workspace_tool"
STORE_TOOL_RESULT_NODE_ID = "store_tool_result"
STORE_FINAL_RESULT_NODE_ID = "store_final_result"
LOOP_END_NODE_ID = "loop_end"
FINAL_OUTPUT_NODE_ID = "final_output"
FINAL_OUTPUT_KEY = "result"
INITIAL_TOOL_RESULT = "No tool has been called yet."
MAX_ROUNDS_EXCEEDED_MESSAGE = (
    "Agent stopped after reaching 100 rounds without a final answer."
)
PLANNER_STREAM_SELECTOR = (PLANNER_NODE_ID, "text")
THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"
PLANNER_DECISION_PARSE_CODE = """
def main(raw_text):
    return {
        "decision": normalize_planner_decision(
            raw_text=raw_text,
        ),
    }
"""
CODE_NODE_LIMITS = CodeNodeLimits(
    max_string_length=20000,
    max_number=10_000_000,
    min_number=-10_000_000,
    max_precision=8,
    max_depth=8,
    max_number_array_length=1000,
    max_string_array_length=1000,
    max_object_array_length=1000,
)

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action_type": {
            "type": "string",
            "enum": ["tool_call", "answer_question"],
        },
        "summary": {"type": "string"},
        "tool_name": {
            "type": "string",
            "enum": ["", "read_file", "write_file", "delete_file", "run_bash"],
        },
        "path": {"type": "string"},
        "content": {"type": "string"},
        "command": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": [
        "action_type",
        "summary",
        "tool_name",
        "path",
        "content",
        "command",
        "answer",
    ],
}


@dataclass(frozen=True, slots=True)
class AgentExampleDependencies:
    provider: str
    prepared_llm: SlimPreparedLLM
    tool_runtime: WorkspaceToolRuntime
    tool_file_manager: NoopToolFileManager
    prompt_message_serializer: PassthroughPromptMessageSerializer
    llm_file_saver: TextOnlyFileSaver
    code_executor: WorkflowCodeExecutor


class LocalPythonCodeExecutor:
    def execute(
        self,
        *,
        language: CodeLanguage,
        code: str,
        inputs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if language != CodeLanguage.PYTHON3:
            msg = f"Unsupported code language for example: {language}"
            raise ValueError(msg)

        globals_dict = {
            "__builtins__": __builtins__,
            "normalize_planner_decision": normalize_planner_decision,
        }
        locals_dict: dict[str, Any] = {}
        exec(code, globals_dict, locals_dict)
        main = locals_dict.get("main")
        if not callable(main):
            msg = "Code snippet must define a callable `main`."
            raise ValueError(msg)

        result = main(**dict(inputs))
        if not isinstance(result, Mapping):
            msg = f"Code executor expected a mapping result, got {type(result)}."
            raise TypeError(msg)
        return dict(result)

    def is_execution_error(self, _error: Exception) -> bool:
        return True


def _coerce_text_field(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _infer_tool_name(*, tool_name: str, path: str, content: str, command: str) -> str:
    if tool_name:
        return tool_name
    if command:
        return "run_bash"
    if path and content:
        return "write_file"
    if path:
        return "read_file"
    return ""


def _looks_like_decision_payload(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    action_type = _coerce_text_field(payload.get("action_type"))
    return action_type in {"tool_call", "answer_question"}


def _extract_last_json_object(raw_text: str) -> Mapping[str, object] | None:
    decoder = json.JSONDecoder()
    decision_candidate: Mapping[str, object] | None = None
    last_object: Mapping[str, object] | None = None

    for start_index, char in enumerate(raw_text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw_text[start_index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, Mapping):
            continue
        last_object = parsed
        if _looks_like_decision_payload(parsed):
            decision_candidate = parsed

    return decision_candidate or last_object


def _longest_suffix_prefix_length(text: str, pattern: str) -> int:
    max_length = min(len(text), len(pattern) - 1)
    for length in range(max_length, 0, -1):
        if text.endswith(pattern[:length]):
            return length
    return 0


@dataclass(slots=True)
class PlannerThinkingStreamState:
    current_round: int = 0
    rounds_by_execution_id: dict[str, int] = field(default_factory=dict)
    chunk_buffers_by_execution_id: dict[str, str] = field(default_factory=dict)
    inside_think_by_execution_id: dict[str, bool] = field(default_factory=dict)
    prefixed_execution_ids: set[str] = field(default_factory=set)


def extract_thinking_text(
    state: PlannerThinkingStreamState,
    *,
    execution_id: str,
    chunk: str,
) -> str:
    buffer = state.chunk_buffers_by_execution_id.get(execution_id, "") + chunk
    inside_think = state.inside_think_by_execution_id.get(execution_id, False)
    extracted_parts: list[str] = []

    while buffer:
        lowered = buffer.lower()
        if inside_think:
            close_index = lowered.find(THINK_CLOSE_TAG)
            if close_index == -1:
                suffix_length = _longest_suffix_prefix_length(lowered, THINK_CLOSE_TAG)
                flush_upto = len(buffer) - suffix_length
                if flush_upto == 0:
                    break
                extracted_parts.append(buffer[:flush_upto])
                buffer = buffer[flush_upto:]
                break

            extracted_parts.append(buffer[:close_index])
            buffer = buffer[close_index + len(THINK_CLOSE_TAG) :]
            inside_think = False
            continue

        open_index = lowered.find(THINK_OPEN_TAG)
        if open_index == -1:
            suffix_length = _longest_suffix_prefix_length(lowered, THINK_OPEN_TAG)
            buffer = buffer[-suffix_length:] if suffix_length else ""
            break
        buffer = buffer[open_index + len(THINK_OPEN_TAG) :]
        inside_think = True

    state.chunk_buffers_by_execution_id[execution_id] = buffer
    state.inside_think_by_execution_id[execution_id] = inside_think
    return "".join(extracted_parts)


def write_planner_thinking_chunk(
    event: object,
    *,
    stream_state: PlannerThinkingStreamState,
    log_output: IO[str],
) -> bool:
    if isinstance(event, NodeRunStartedEvent) and event.node_id == PLANNER_NODE_ID:
        stream_state.current_round += 1
        stream_state.rounds_by_execution_id[event.id] = stream_state.current_round
        return False

    if not isinstance(event, NodeRunStreamChunkEvent):
        return False
    if event.node_id != PLANNER_NODE_ID:
        return False
    if tuple(event.selector) != PLANNER_STREAM_SELECTOR:
        return False

    execution_id = event.id
    round_number = stream_state.rounds_by_execution_id.get(
        execution_id,
        stream_state.current_round or 1,
    )
    thinking_text = extract_thinking_text(
        stream_state,
        execution_id=execution_id,
        chunk=event.chunk,
    )

    wrote_output = False
    if thinking_text:
        if execution_id not in stream_state.prefixed_execution_ids:
            log_output.write(f"[round {round_number} thinking] ")
            stream_state.prefixed_execution_ids.add(execution_id)
        log_output.write(thinking_text)
        log_output.flush()
        wrote_output = True

    if event.is_final:
        if execution_id in stream_state.prefixed_execution_ids:
            log_output.write("\n")
            log_output.flush()
            stream_state.prefixed_execution_ids.discard(execution_id)
        stream_state.rounds_by_execution_id.pop(execution_id, None)
        stream_state.chunk_buffers_by_execution_id.pop(execution_id, None)
        stream_state.inside_think_by_execution_id.pop(execution_id, None)

    return wrote_output


def normalize_planner_decision(
    *,
    raw_text: object,
    structured_output: object = None,
) -> dict[str, str]:
    payload: object = structured_output
    if not isinstance(payload, Mapping):
        raw_text_string = _coerce_text_field(raw_text)
        if not raw_text_string:
            msg = "Planner returned neither structured output nor JSON text."
            raise ValueError(msg)
        try:
            payload = json.loads(raw_text_string)
        except json.JSONDecodeError:
            payload = _extract_last_json_object(raw_text_string)
            if payload is None:
                try:
                    payload = parse_json_markdown(raw_text_string)
                except (ValueError, json.JSONDecodeError):
                    payload = {
                        "action_type": "answer_question",
                        "summary": "Planner returned a plain-text answer.",
                        "tool_name": "",
                        "path": "",
                        "content": "",
                        "command": "",
                        "answer": raw_text_string,
                    }

    if not isinstance(payload, Mapping):
        msg = f"Planner output must be an object, got {type(payload).__name__}."
        raise TypeError(msg)

    action_type = _coerce_text_field(payload.get("action_type"))
    summary = _coerce_text_field(
        payload.get("summary") or payload.get("justification"),
    )
    path = _coerce_text_field(payload.get("path"))
    content = _coerce_text_field(payload.get("content"))
    command = _coerce_text_field(payload.get("command"))
    answer = _coerce_text_field(payload.get("answer"))
    tool_name = _infer_tool_name(
        tool_name=_coerce_text_field(payload.get("tool_name")),
        path=path,
        content=content,
        command=command,
    )

    if action_type not in {"tool_call", "answer_question"}:
        msg = f"Unsupported action_type: {action_type!r}"
        raise ValueError(msg)

    if not summary:
        if action_type == "tool_call":
            summary = "Use a tool to gather missing repository context."
        else:
            summary = "Return a direct answer."

    if action_type == "tool_call":
        if tool_name not in {"read_file", "write_file", "delete_file", "run_bash"}:
            msg = f"Unsupported tool_name: {tool_name!r}"
            raise ValueError(msg)
        if tool_name in {"read_file", "write_file", "delete_file"} and not path:
            msg = f"{tool_name} requires a path."
            raise ValueError(msg)
        if tool_name == "run_bash" and not command:
            msg = "run_bash requires a command."
            raise ValueError(msg)
    elif not answer:
        answer = summary

    return {
        "action_type": action_type,
        "summary": summary,
        "tool_name": tool_name,
        "path": path,
        "content": content,
        "command": command,
        "answer": answer,
    }


class AgentChildEngineBuilder:
    def __init__(self, dependencies: AgentExampleDependencies) -> None:
        self._dependencies = dependencies

    def build_child_engine(
        self,
        *,
        workflow_id: str,
        graph_init_params: GraphInitParams,
        parent_graph_runtime_state: GraphRuntimeState,
        root_node_id: str,
        variable_pool: VariablePool | None = None,
    ) -> GraphEngine:
        node_ids = {
            str(node_config["id"])
            for node_config in graph_init_params.graph_config.get("nodes", [])
            if "id" in node_config
        }
        if root_node_id not in node_ids:
            msg = f"Unknown child graph root node: {root_node_id}"
            raise ChildGraphNotFoundError(msg)

        child_runtime_state = GraphRuntimeState(
            variable_pool=variable_pool or parent_graph_runtime_state.variable_pool,
            start_at=time.time(),
        )
        child_graph = build_graph(
            graph_config=graph_init_params.graph_config,
            graph_init_params=graph_init_params,
            graph_runtime_state=child_runtime_state,
            dependencies=self._dependencies,
            root_node_id=root_node_id,
        )
        return GraphEngine(
            workflow_id=workflow_id,
            graph=child_graph,
            graph_runtime_state=child_runtime_state,
            command_channel=InMemoryChannel(),
            child_engine_builder=self,
        )


def build_graph_config(*, provider: str, workspace_root: Path) -> dict[str, object]:
    system_prompt = (
        "You are a repository agent. "
        f"You operate inside the workspace root `{workspace_root}`. "
        "You may read files, overwrite files, delete files, and run bash commands. "
        "Choose exactly one next action per round. "
        "Available tools are read_file(path), write_file(path, content), "
        "delete_file(path), and run_bash(command). "
        "Paths are relative to the workspace root unless they are already absolute "
        "and still inside that workspace. "
        "First, write a brief user-visible planning note inside exact lowercase "
        "<think> and </think> tags. Keep that note under three short sentences. "
        "After the closing </think>, output exactly one JSON object and nothing "
        "else. Do not wrap the JSON in code fences and do not repeat it. "
        "If you can answer the user directly, use action_type `answer_question`. "
        "If you need a tool, use action_type `tool_call`. "
        "Return a JSON object with exactly these keys: action_type, summary, "
        "tool_name, path, content, command, answer. "
        "Always fill every field in the schema and use empty strings for fields "
        "that do not apply to the chosen action."
    )
    user_prompt = (
        "User question:\n"
        "{{#start.query#}}\n\n"
        "Previous tool result:\n"
        "{{#agent_loop.tool_result#}}\n\n"
        "Return only the next structured decision."
    )

    nodes: list[dict[str, object]] = [
        {
            "id": "start",
            "data": StartNodeData(
                title="Start",
                variables=[
                    VariableEntity(
                        variable="query",
                        label="Query",
                        type=VariableEntityType.PARAGRAPH,
                        required=True,
                    ),
                ],
            ),
        },
        {
            "id": LOOP_NODE_ID,
            "data": LoopNodeData(
                title="Agent Loop",
                start_node_id=LOOP_START_NODE_ID,
                loop_count=100,
                break_conditions=[],
                logical_operator="and",
                loop_variables=[
                    LoopVariableData(
                        label="tool_result",
                        var_type=SegmentType.STRING,
                        value_type="constant",
                        value=INITIAL_TOOL_RESULT,
                    ),
                    LoopVariableData(
                        label="final_result",
                        var_type=SegmentType.STRING,
                        value_type="constant",
                        value=MAX_ROUNDS_EXCEEDED_MESSAGE,
                    ),
                ],
            ),
        },
        {
            "id": FINAL_OUTPUT_NODE_ID,
            "data": EndNodeData(
                title="Output",
                outputs=[
                    OutputVariableEntity(
                        variable=FINAL_OUTPUT_KEY,
                        value_selector=[LOOP_NODE_ID, "final_result"],
                    ),
                ],
            ),
        },
        {
            "id": LOOP_START_NODE_ID,
            "data": LoopStartNodeData(
                title="Loop Start",
                loop_id=LOOP_NODE_ID,
            ),
        },
        {
            "id": PLANNER_NODE_ID,
            "data": LLMNodeData(
                title="Planner",
                loop_id=LOOP_NODE_ID,
                model=ModelConfig(
                    provider=provider,
                    name="gpt-5.4",
                    mode=LLMMode.CHAT,
                ),
                prompt_template=[
                    LLMNodeChatModelMessage(
                        role=PromptMessageRole.SYSTEM,
                        text=system_prompt,
                    ),
                    LLMNodeChatModelMessage(
                        role=PromptMessageRole.USER,
                        text=user_prompt,
                    ),
                ],
                context=ContextConfig(enabled=False),
                reasoning_format="separated",
            ),
        },
        {
            "id": PLANNER_DECISION_NODE_ID,
            "data": CodeNodeData(
                title="Parse Planner Decision",
                loop_id=LOOP_NODE_ID,
                variables=[
                    VariableSelector(
                        variable="raw_text",
                        value_selector=[PLANNER_NODE_ID, "text"],
                    ),
                ],
                code_language=CodeLanguage.PYTHON3,
                code=PLANNER_DECISION_PARSE_CODE,
                outputs={
                    "decision": CodeNodeData.Output(
                        type=SegmentType.OBJECT,
                        children={
                            "action_type": CodeNodeData.Output(
                                type=SegmentType.STRING,
                            ),
                            "summary": CodeNodeData.Output(type=SegmentType.STRING),
                            "tool_name": CodeNodeData.Output(
                                type=SegmentType.STRING,
                            ),
                            "path": CodeNodeData.Output(type=SegmentType.STRING),
                            "content": CodeNodeData.Output(type=SegmentType.STRING),
                            "command": CodeNodeData.Output(type=SegmentType.STRING),
                            "answer": CodeNodeData.Output(type=SegmentType.STRING),
                        },
                    ),
                },
            ),
        },
        {
            "id": ROUND_LOG_NODE_ID,
            "data": EndNodeData(
                title="Round Log",
                loop_id=LOOP_NODE_ID,
                outputs=[
                    OutputVariableEntity(
                        variable="agent_log",
                        value_selector=[PLANNER_DECISION_NODE_ID, "decision"],
                    ),
                ],
            ),
        },
        {
            "id": ROUTE_ACTION_NODE_ID,
            "data": IfElseNodeData(
                title="Route Action",
                loop_id=LOOP_NODE_ID,
                cases=[
                    IfElseNodeData.Case(
                        case_id="answer_question",
                        logical_operator="and",
                        conditions=[
                            Condition(
                                variable_selector=[
                                    PLANNER_DECISION_NODE_ID,
                                    "decision",
                                    "action_type",
                                ],
                                comparison_operator="is",
                                value="answer_question",
                            ),
                        ],
                    ),
                    IfElseNodeData.Case(
                        case_id="tool_call",
                        logical_operator="and",
                        conditions=[
                            Condition(
                                variable_selector=[
                                    PLANNER_DECISION_NODE_ID,
                                    "decision",
                                    "action_type",
                                ],
                                comparison_operator="is",
                                value="tool_call",
                            ),
                        ],
                    ),
                ],
            ),
        },
        {
            "id": WORKSPACE_TOOL_NODE_ID,
            "data": ToolNodeData(
                title="Workspace Tool",
                loop_id=LOOP_NODE_ID,
                provider_id="workspace",
                provider_type=ToolProviderType.BUILT_IN,
                provider_name="workspace",
                tool_name="workspace_agent_tools",
                tool_label="Workspace Agent Tools",
                tool_configurations={},
                tool_parameters={
                    "tool_name": ToolNodeData.ToolInput(
                        type="mixed",
                        value="{{#planner_decision.decision.tool_name#}}",
                    ),
                    "path": ToolNodeData.ToolInput(
                        type="mixed",
                        value="{{#planner_decision.decision.path#}}",
                    ),
                    "content": ToolNodeData.ToolInput(
                        type="mixed",
                        value="{{#planner_decision.decision.content#}}",
                    ),
                    "command": ToolNodeData.ToolInput(
                        type="mixed",
                        value="{{#planner_decision.decision.command#}}",
                    ),
                },
            ),
        },
        {
            "id": STORE_TOOL_RESULT_NODE_ID,
            "data": VariableAssignerNodeData(
                title="Store Tool Result",
                loop_id=LOOP_NODE_ID,
                items=[
                    VariableOperationItem(
                        variable_selector=[LOOP_NODE_ID, "tool_result"],
                        input_type=InputType.VARIABLE,
                        operation=Operation.OVER_WRITE,
                        value=[WORKSPACE_TOOL_NODE_ID, "result"],
                    ),
                ],
            ),
        },
        {
            "id": STORE_FINAL_RESULT_NODE_ID,
            "data": VariableAssignerNodeData(
                title="Store Final Result",
                loop_id=LOOP_NODE_ID,
                items=[
                    VariableOperationItem(
                        variable_selector=[LOOP_NODE_ID, "final_result"],
                        input_type=InputType.VARIABLE,
                        operation=Operation.OVER_WRITE,
                        value=[PLANNER_DECISION_NODE_ID, "decision", "answer"],
                    ),
                ],
            ),
        },
        {
            "id": LOOP_END_NODE_ID,
            "data": LoopEndNodeData(
                title="Loop End",
                loop_id=LOOP_NODE_ID,
            ),
        },
    ]
    edges = [
        {"source": "start", "target": LOOP_NODE_ID, "sourceHandle": "source"},
        {
            "source": LOOP_NODE_ID,
            "target": FINAL_OUTPUT_NODE_ID,
            "sourceHandle": "source",
        },
        {
            "source": LOOP_START_NODE_ID,
            "target": PLANNER_NODE_ID,
            "sourceHandle": "source",
        },
        {
            "source": PLANNER_NODE_ID,
            "target": PLANNER_DECISION_NODE_ID,
            "sourceHandle": "source",
        },
        {
            "source": PLANNER_DECISION_NODE_ID,
            "target": ROUND_LOG_NODE_ID,
            "sourceHandle": "source",
        },
        {
            "source": ROUND_LOG_NODE_ID,
            "target": ROUTE_ACTION_NODE_ID,
            "sourceHandle": "source",
        },
        {
            "source": ROUTE_ACTION_NODE_ID,
            "target": STORE_FINAL_RESULT_NODE_ID,
            "sourceHandle": "answer_question",
        },
        {
            "source": ROUTE_ACTION_NODE_ID,
            "target": WORKSPACE_TOOL_NODE_ID,
            "sourceHandle": "tool_call",
        },
        {
            "source": STORE_FINAL_RESULT_NODE_ID,
            "target": LOOP_END_NODE_ID,
            "sourceHandle": "source",
        },
        {
            "source": WORKSPACE_TOOL_NODE_ID,
            "target": STORE_TOOL_RESULT_NODE_ID,
            "sourceHandle": "source",
        },
    ]
    return {"nodes": nodes, "edges": edges}


def build_graph(
    *,
    graph_config: Mapping[str, Any],
    graph_init_params: GraphInitParams,
    graph_runtime_state: GraphRuntimeState,
    dependencies: AgentExampleDependencies,
    root_node_id: str,
) -> Graph:
    nodes: dict[str, object] = {}
    ordered_ids: list[str] = []
    for node_config in graph_config.get("nodes", []):
        node_id = cast("str", node_config["id"])
        ordered_ids.append(node_id)
        nodes[node_id] = _build_node(
            node_config=cast("dict[str, object]", node_config),
            graph_init_params=graph_init_params,
            graph_runtime_state=graph_runtime_state,
            dependencies=dependencies,
        )

    if root_node_id not in nodes:
        msg = f"Unknown graph root node: {root_node_id}"
        raise ValueError(msg)

    edges: dict[str, Edge] = {}
    in_edges: dict[str, list[str]] = defaultdict(list)
    out_edges: dict[str, list[str]] = defaultdict(list)
    for index, edge_config in enumerate(graph_config.get("edges", [])):
        source = cast("str", edge_config["source"])
        target = cast("str", edge_config["target"])
        edge = Edge(
            id=f"edge_{index}",
            tail=source,
            head=target,
            source_handle=str(edge_config.get("sourceHandle", "source")),
        )
        edges[edge.id] = edge
        out_edges[source].append(edge.id)
        in_edges[target].append(edge.id)

    graph = Graph(
        nodes={node_id: cast("Any", nodes[node_id]) for node_id in ordered_ids},
        edges=edges,
        in_edges=dict(in_edges),
        out_edges=dict(out_edges),
        root_node=cast("Any", nodes[root_node_id]),
    )
    get_graph_validator().validate(graph)
    return graph


def _build_node(
    *,
    node_config: dict[str, object],
    graph_init_params: GraphInitParams,
    graph_runtime_state: GraphRuntimeState,
    dependencies: AgentExampleDependencies,
) -> object:
    node_id = cast("str", node_config["id"])
    node_data = node_config["data"]
    node_type = node_data.type

    shared_kwargs = {
        "node_id": node_id,
        "config": node_config,
        "graph_init_params": graph_init_params,
        "graph_runtime_state": graph_runtime_state,
    }

    match node_type:
        case BuiltinNodeTypes.START:
            node_cls = StartNode
            extra_kwargs: dict[str, object] = {}
        case BuiltinNodeTypes.LOOP:
            node_cls = LoopNode
            extra_kwargs = {}
        case BuiltinNodeTypes.LOOP_START:
            node_cls = LoopStartNode
            extra_kwargs = {}
        case BuiltinNodeTypes.LLM:
            node_cls = LLMNode
            extra_kwargs = {
                "model_instance": dependencies.prepared_llm,
                "llm_file_saver": dependencies.llm_file_saver,
                "prompt_message_serializer": (dependencies.prompt_message_serializer),
            }
        case BuiltinNodeTypes.CODE:
            node_cls = CodeNode
            extra_kwargs = {
                "code_executor": dependencies.code_executor,
                "code_limits": CODE_NODE_LIMITS,
            }
        case BuiltinNodeTypes.END:
            node_cls = EndNode
            extra_kwargs = {}
        case BuiltinNodeTypes.IF_ELSE:
            node_cls = IfElseNode
            extra_kwargs = {}
        case BuiltinNodeTypes.TOOL:
            node_cls = ToolNode
            extra_kwargs = {
                "runtime": dependencies.tool_runtime,
                "tool_file_manager_factory": dependencies.tool_file_manager,
            }
        case BuiltinNodeTypes.VARIABLE_ASSIGNER:
            node_cls = VariableAssignerNode
            extra_kwargs = {}
        case BuiltinNodeTypes.LOOP_END:
            node_cls = LoopEndNode
            extra_kwargs = {}
        case _:
            msg = f"Unsupported node type in example graph: {node_type}"
            raise ValueError(msg)

    return node_cls(**shared_kwargs, **extra_kwargs)


def format_round_log(event: object) -> str | None:
    if not hasattr(event, "node_id") or event.node_id != ROUND_LOG_NODE_ID:
        return None

    node_run_result = getattr(event, "node_run_result", None)
    if node_run_result is None:
        return None
    outputs = getattr(node_run_result, "outputs", {})
    if not isinstance(outputs, Mapping):
        return None

    raw_log = outputs.get("agent_log")
    if not isinstance(raw_log, Mapping):
        return None

    metadata = getattr(node_run_result, "metadata", {})
    round_index = 0
    if isinstance(metadata, Mapping):
        raw_round_index = metadata.get(WorkflowNodeExecutionMetadataKey.LOOP_INDEX, 0)
        if isinstance(raw_round_index, int):
            round_index = raw_round_index

    action_type = str(raw_log.get("action_type", "")).strip() or "unknown"
    summary = _truncate_for_log(str(raw_log.get("summary", "")).strip(), 120)
    if action_type == "tool_call":
        tool_name = str(raw_log.get("tool_name", "")).strip()
        target = str(raw_log.get("path", "") or raw_log.get("command", "")).strip()
        details = _truncate_for_log(target, 120)
        return (
            f"[round {round_index + 1}] {action_type} {tool_name} {summary} {details}"
        ).strip()
    answer = _truncate_for_log(str(raw_log.get("answer", "")).strip(), 120)
    return f"[round {round_index + 1}] {action_type} {summary} {answer}".strip()


def _truncate_for_log(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "..."


def _execute_workflow(
    query: str,
    *,
    stream_output: IO[str] | None = None,
    log_output: IO[str] | None = None,
) -> str:
    load_default_env_file()
    runtime, provider = build_runtime()
    graph_config = build_graph_config(provider=provider, workspace_root=WORKSPACE_ROOT)
    graph_init_params = GraphInitParams(
        workflow_id=WORKFLOW_ID,
        graph_config=graph_config,
        run_context={},
        call_depth=0,
    )
    graph_runtime_state = GraphRuntimeState(
        variable_pool=VariablePool(),
        start_at=time.time(),
    )
    graph_runtime_state.variable_pool.add(("start", "query"), query)

    dependencies = AgentExampleDependencies(
        provider=provider,
        prepared_llm=SlimPreparedLLM(
            runtime=runtime,
            provider=provider,
            model_name="gpt-5.4",
            credentials={"openai_api_key": require_env("OPENAI_API_KEY")},
            parameters={},
        ),
        tool_runtime=WorkspaceToolRuntime(
            build_workspace_tool_settings(workspace_root=WORKSPACE_ROOT),
        ),
        tool_file_manager=NoopToolFileManager(),
        prompt_message_serializer=PassthroughPromptMessageSerializer(),
        llm_file_saver=TextOnlyFileSaver(),
        code_executor=LocalPythonCodeExecutor(),
    )
    graph = build_graph(
        graph_config=graph_config,
        graph_init_params=graph_init_params,
        graph_runtime_state=graph_runtime_state,
        dependencies=dependencies,
        root_node_id="start",
    )
    engine = GraphEngine(
        workflow_id=WORKFLOW_ID,
        graph=graph,
        graph_runtime_state=graph_runtime_state,
        command_channel=InMemoryChannel(),
        child_engine_builder=AgentChildEngineBuilder(dependencies),
    )

    planner_stream_state = PlannerThinkingStreamState()
    for event in engine.run():
        if log_output is None:
            continue
        write_planner_thinking_chunk(
            event,
            stream_state=planner_stream_state,
            log_output=log_output,
        )
        round_log = format_round_log(event)
        if round_log is None:
            continue
        log_output.write(round_log + "\n")
        log_output.flush()

    result = graph_runtime_state.get_output(FINAL_OUTPUT_KEY)
    if not isinstance(result, str):
        msg = "Workflow did not produce a text result."
        raise TypeError(msg)
    if stream_output is not None:
        stream_output.write(result + "\n")
        stream_output.flush()
    return result


def run_workflow(query: str) -> str:
    return _execute_workflow(query)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Graphon agent-mode example with local workspace tools.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="Read README.md and summarize the repository in two sentences.",
        help="User input passed into the Start node.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _execute_workflow(
        args.query,
        stream_output=sys.stdout,
        log_output=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
