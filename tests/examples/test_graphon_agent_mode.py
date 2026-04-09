from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

from examples.graphon_agent_mode.support import (
    ALLOWED_ENV_VARS,
    build_local_settings,
    build_workspace_tool_settings,
    patch_openai_timeout_source,
)
from examples.graphon_agent_mode.workflow import (
    FINAL_OUTPUT_KEY,
    LOOP_NODE_ID,
    MAX_ROUNDS_EXCEEDED_MESSAGE,
    PLANNER_DECISION_NODE_ID,
    PLANNER_NODE_ID,
    PLANNER_STREAM_SELECTOR,
    ROUND_LOG_NODE_ID,
    WORKSPACE_ROOT,
    PlannerThinkingStreamState,
    build_graph_config,
    extract_thinking_text,
    format_round_log,
    normalize_planner_decision,
    write_planner_thinking_chunk,
)
from examples.graphon_agent_mode.workspace_tools import (
    WorkspaceToolRuntime,
    WorkspaceToolSettings,
)
from graphon.enums import (
    BuiltinNodeTypes,
    WorkflowNodeExecutionMetadataKey,
    WorkflowNodeExecutionStatus,
)
from graphon.graph_events.node import NodeRunStartedEvent, NodeRunStreamChunkEvent
from graphon.node_events.base import NodeRunResult
from graphon.nodes.tool_runtime_entities import ToolRuntimeHandle


def test_env_example_matches_allowed_env_vars() -> None:
    env_example = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "graphon_agent_mode"
        / ".env.example"
    )
    keys = {
        line.split("=", 1)[0].removeprefix("export ").strip()
        for line in env_example.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert keys == set(ALLOWED_ENV_VARS)


def test_build_local_settings_uses_relaxed_timeout_defaults(monkeypatch) -> None:
    monkeypatch.delenv("SLIM_PLUGIN_FOLDER", raising=False)
    monkeypatch.delenv("SLIM_PYTHON_ENV_INIT_TIMEOUT", raising=False)
    monkeypatch.delenv("SLIM_MAX_EXECUTION_TIMEOUT", raising=False)

    settings = build_local_settings()

    assert settings.python_env_init_timeout == 300
    assert settings.max_execution_timeout == 1800


def test_build_workspace_tool_settings_uses_relaxed_timeout_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("AGENT_BASH_COMMAND_TIMEOUT_SECONDS", raising=False)

    settings = build_workspace_tool_settings(workspace_root=tmp_path)

    assert settings.workspace_root == tmp_path
    assert settings.command_timeout_seconds == 120


def test_patch_openai_timeout_source_relaxes_plugin_timeouts() -> None:
    source_text = (
        "credentials_kwargs = {\n"
        '    "api_key": credentials["openai_api_key"],\n'
        '    "timeout": Timeout(315.0, read=300.0, write=10.0, connect=5.0),\n'
        '    "max_retries": 1,\n'
        "}\n"
    )

    patched = patch_openai_timeout_source(
        source_text,
        read_timeout_seconds=1800,
        connect_timeout_seconds=30,
        write_timeout_seconds=30,
    )

    assert (
        '"timeout": Timeout(1800.0, read=1800.0, write=30.0, connect=30.0),' in patched
    )


def test_build_graph_config_contains_requested_agent_structure() -> None:
    graph_config = build_graph_config(provider="openai", workspace_root=WORKSPACE_ROOT)
    nodes = {node["id"]: node for node in graph_config["nodes"]}
    edges = graph_config["edges"]

    loop_node = nodes[LOOP_NODE_ID]["data"]
    assert loop_node.loop_count == 100
    assert loop_node.start_node_id == "loop_start"
    assert loop_node.loop_variables[0].label == "tool_result"
    assert loop_node.loop_variables[1].value == MAX_ROUNDS_EXCEEDED_MESSAGE

    planner_node = nodes["planner"]["data"]
    assert planner_node.structured_output_switch_on is False
    assert planner_node.reasoning_format == "separated"
    assert "<think>" in planner_node.prompt_template[0].text

    parser_node = nodes[PLANNER_DECISION_NODE_ID]["data"]
    assert parser_node.code_language == "python3"
    assert parser_node.variables[0].value_selector == ["planner", "text"]
    assert len(parser_node.variables) == 1

    route_node = nodes["route_action"]["data"]
    assert [case.case_id for case in route_node.cases] == [
        "answer_question",
        "tool_call",
    ]

    final_output = nodes["final_output"]["data"]
    assert final_output.outputs[0].variable == FINAL_OUTPUT_KEY
    assert final_output.outputs[0].value_selector == [LOOP_NODE_ID, "final_result"]

    assert {
        (edge["source"], edge["target"], edge["sourceHandle"]) for edge in edges
    } >= {
        ("start", LOOP_NODE_ID, "source"),
        (LOOP_NODE_ID, "final_output", "source"),
        ("planner", PLANNER_DECISION_NODE_ID, "source"),
        (PLANNER_DECISION_NODE_ID, "round_log", "source"),
        ("route_action", "store_final_result", "answer_question"),
        ("route_action", "workspace_tool", "tool_call"),
    }


def test_normalize_planner_decision_parses_json_text_and_infers_read_file() -> None:
    decision = normalize_planner_decision(
        raw_text="""
        {
          "action_type": "tool_call",
          "justification": "Need to inspect README.md before summarizing.",
          "path": "README.md",
          "content": "",
          "command": "",
          "answer": ""
        }
        """,
        structured_output=None,
    )

    assert decision == {
        "action_type": "tool_call",
        "summary": "Need to inspect README.md before summarizing.",
        "tool_name": "read_file",
        "path": "README.md",
        "content": "",
        "command": "",
        "answer": "",
    }


def test_normalize_planner_decision_prefers_raw_json_before_markdown_heuristics() -> (
    None
):
    decision = normalize_planner_decision(
        raw_text=(
            '{"action_type":"answer_question","summary":"Done.","tool_name":"","path":"",'
            '"content":"","command":"","answer":"Use `src/graphon/graph` for the graph '
            'layer."}'
        ),
        structured_output=None,
    )

    assert decision["action_type"] == "answer_question"
    assert decision["answer"] == "Use `src/graphon/graph` for the graph layer."


def test_normalize_planner_decision_recovers_last_json_object_from_messy_text() -> None:
    decision = normalize_planner_decision(
        raw_text=(
            '{"action_type":"tool_call","summary":"Read README.md.",'
            '"tool_name":"read_file","path":"README.md","content":"",'
            '"command":"","answer":""}\n'
            "\n---\n"
            '{"action_type":"tool_call","summary":"Inspect src/graphon next.",'
            '"tool_name":"run_bash","path":"","content":"",'
            '"command":"find src/graphon -maxdepth 2 -type d | sort",'
            '"answer":""}'
        ),
        structured_output=None,
    )

    assert decision["action_type"] == "tool_call"
    assert decision["tool_name"] == "run_bash"
    assert decision["command"] == "find src/graphon -maxdepth 2 -type d | sort"


def test_normalize_planner_decision_falls_back_to_plain_text_answer() -> None:
    decision = normalize_planner_decision(
        raw_text="The repository is organized into src/, examples/, and tests/.",
        structured_output=None,
    )

    assert decision == {
        "action_type": "answer_question",
        "summary": "Planner returned a plain-text answer.",
        "tool_name": "",
        "path": "",
        "content": "",
        "command": "",
        "answer": "The repository is organized into src/, examples/, and tests/.",
    }


def test_workspace_tool_runtime_file_round_trip(tmp_path: Path) -> None:
    runtime = WorkspaceToolRuntime(WorkspaceToolSettings(workspace_root=tmp_path))
    handle = ToolRuntimeHandle(raw=WorkspaceToolSettings(workspace_root=tmp_path))

    write_messages = list(
        runtime.invoke(
            tool_runtime=handle,
            tool_parameters={
                "tool_name": "write_file",
                "path": "notes.txt",
                "content": "hello",
                "command": "",
            },
            workflow_call_depth=0,
            provider_name="workspace",
        ),
    )
    read_messages = list(
        runtime.invoke(
            tool_runtime=handle,
            tool_parameters={
                "tool_name": "read_file",
                "path": "notes.txt",
                "content": "",
                "command": "",
            },
            workflow_call_depth=0,
            provider_name="workspace",
        ),
    )
    delete_messages = list(
        runtime.invoke(
            tool_runtime=handle,
            tool_parameters={
                "tool_name": "delete_file",
                "path": "notes.txt",
                "content": "",
                "command": "",
            },
            workflow_call_depth=0,
            provider_name="workspace",
        ),
    )

    write_result = json.loads(write_messages[-1].message.variable_value)
    read_result = json.loads(read_messages[-1].message.variable_value)
    delete_result = json.loads(delete_messages[-1].message.variable_value)

    assert write_result["status"] == "ok"
    assert read_result["content"] == "hello"
    assert delete_result["status"] == "ok"
    assert not (tmp_path / "notes.txt").exists()


def test_workspace_tool_runtime_run_bash_captures_output(tmp_path: Path) -> None:
    settings = WorkspaceToolSettings(workspace_root=tmp_path)
    runtime = WorkspaceToolRuntime(settings)
    handle = ToolRuntimeHandle(raw=settings)

    messages = list(
        runtime.invoke(
            tool_runtime=handle,
            tool_parameters={
                "tool_name": "run_bash",
                "path": "",
                "content": "",
                "command": "printf 'graphon'",
            },
            workflow_call_depth=0,
            provider_name="workspace",
        ),
    )
    result = json.loads(messages[-1].message.variable_value)

    assert result["status"] == "ok"
    assert result["stdout"] == "graphon"
    assert result["returncode"] == 0


def test_extract_thinking_text_handles_split_tags() -> None:
    state = PlannerThinkingStreamState()

    first = extract_thinking_text(
        state,
        execution_id="planner-run",
        chunk="<th",
    )
    second = extract_thinking_text(
        state,
        execution_id="planner-run",
        chunk="ink>Inspect README",
    )
    third = extract_thinking_text(
        state,
        execution_id="planner-run",
        chunk=".md first</th",
    )
    fourth = extract_thinking_text(
        state,
        execution_id="planner-run",
        chunk='ink>{"action_type":"tool_call"}',
    )

    assert not first
    assert second == "Inspect README"
    assert third == ".md first"
    assert not fourth


def test_write_planner_thinking_chunk_streams_only_think_content() -> None:
    state = PlannerThinkingStreamState()
    log_output = io.StringIO()

    started = NodeRunStartedEvent(
        id="planner-run",
        node_id=PLANNER_NODE_ID,
        node_type=BuiltinNodeTypes.LLM,
        node_title="Planner",
        start_at=datetime.now(UTC),
    )
    thinking_chunk = NodeRunStreamChunkEvent(
        id="planner-run",
        node_id=PLANNER_NODE_ID,
        node_type=BuiltinNodeTypes.LLM,
        selector=PLANNER_STREAM_SELECTOR,
        chunk='<think>Inspect README.md</think>{"action_type":"tool_call"}',
        is_final=False,
    )
    final_chunk = NodeRunStreamChunkEvent(
        id="planner-run",
        node_id=PLANNER_NODE_ID,
        node_type=BuiltinNodeTypes.LLM,
        selector=PLANNER_STREAM_SELECTOR,
        chunk="",
        is_final=True,
    )

    assert (
        write_planner_thinking_chunk(
            started,
            stream_state=state,
            log_output=log_output,
        )
        is False
    )
    assert write_planner_thinking_chunk(
        thinking_chunk,
        stream_state=state,
        log_output=log_output,
    )
    assert not write_planner_thinking_chunk(
        final_chunk,
        stream_state=state,
        log_output=log_output,
    )
    assert log_output.getvalue() == "[round 1 thinking] Inspect README.md\n"


def test_format_round_log_renders_agent_decision() -> None:
    event = type(
        "RoundLogEvent",
        (),
        {
            "node_id": ROUND_LOG_NODE_ID,
            "node_run_result": NodeRunResult(
                status=WorkflowNodeExecutionStatus.SUCCEEDED,
                outputs={
                    "agent_log": {
                        "action_type": "tool_call",
                        "summary": "Inspect the README before answering.",
                        "tool_name": "read_file",
                        "path": "README.md",
                        "content": "",
                        "command": "",
                        "answer": "",
                    },
                },
                metadata={WorkflowNodeExecutionMetadataKey.LOOP_INDEX: 1},
            ),
        },
    )()

    assert format_round_log(event) == (
        "[round 2] tool_call read_file Inspect the README before answering. README.md"
    )
