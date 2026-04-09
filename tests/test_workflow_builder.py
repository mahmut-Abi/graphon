from __future__ import annotations

import time
from typing import cast

from graphon.entities.graph_init_params import GraphInitParams
from graphon.model_runtime.entities.llm_entities import LLMMode
from graphon.nodes.end.end_node import EndNode
from graphon.nodes.end.entities import EndNodeData
from graphon.nodes.llm import LLMNodeData, ModelConfig
from graphon.nodes.llm.runtime_protocols import PreparedLLMProtocol
from graphon.nodes.start.entities import StartNodeData
from graphon.runtime.graph_runtime_state import GraphRuntimeState
from graphon.runtime.variable_pool import VariablePool
from graphon.workflow_builder import (
    WorkflowBuilder,
    paragraph_input,
    system,
    user,
)


def test_llm_node_data_defaults_context_to_disabled() -> None:
    node_data = LLMNodeData(
        model=ModelConfig(
            provider="mock",
            name="mock-chat",
            mode=LLMMode.CHAT,
        ),
        prompt_template=[system("Translate this text.")],
    )

    assert node_data.context.enabled is False
    assert node_data.context.variable_selector is None


def test_workflow_builder_builds_parallel_translation_workflow() -> None:
    graph_init_params = GraphInitParams(
        workflow_id="parallel-translation",
        graph_config={"nodes": [], "edges": []},
        run_context={},
        call_depth=0,
    )
    graph_runtime_state = GraphRuntimeState(
        variable_pool=VariablePool(),
        start_at=time.time(),
    )
    builder = WorkflowBuilder(
        graph_init_params=graph_init_params,
        graph_runtime_state=graph_runtime_state,
        prepared_llm=cast(PreparedLLMProtocol, object()),
    )

    start = builder.root(
        "start",
        StartNodeData(
            variables=[paragraph_input("content", required=True)],
        ),
    )

    translation_model = ModelConfig(
        provider="mock",
        name="mock-chat",
        mode=LLMMode.CHAT,
    )

    chinese = start.then(
        "translate_zh",
        LLMNodeData(
            model=translation_model,
            prompt_template=[
                system("Translate the following text to Chinese."),
                user(start.ref("content")),
            ],
        ),
    )
    english = start.then(
        "translate_en",
        LLMNodeData(
            model=translation_model,
            prompt_template=[
                system("Translate the following text to English."),
                user(start.ref("content")),
            ],
        ),
    )
    japanese = start.then(
        "translate_ja",
        LLMNodeData(
            model=translation_model,
            prompt_template=[
                system("Translate the following text to Japanese."),
                user(start.ref("content")),
            ],
        ),
    )

    output = chinese.then(
        "output",
        EndNodeData(
            outputs=[
                chinese.ref("text").output("chinese"),
                english.ref("text").output("english"),
                japanese.ref("text").output("japanese"),
            ],
        ),
    )
    english.connect(output)
    japanese.connect(output)

    graph = builder.build()

    assert graph.root_node.id == "start"
    assert isinstance(graph.nodes["output"], EndNode)
    assert sorted((edge.tail, edge.head) for edge in graph.edges.values()) == [
        ("start", "translate_en"),
        ("start", "translate_ja"),
        ("start", "translate_zh"),
        ("translate_en", "output"),
        ("translate_ja", "output"),
        ("translate_zh", "output"),
    ]

    output_node = cast(EndNode, graph.nodes["output"])
    assert [item.variable for item in output_node.node_data.outputs] == [
        "chinese",
        "english",
        "japanese",
    ]
    assert [tuple(item.value_selector) for item in output_node.node_data.outputs] == [
        ("translate_zh", "text"),
        ("translate_en", "text"),
        ("translate_ja", "text"),
    ]
