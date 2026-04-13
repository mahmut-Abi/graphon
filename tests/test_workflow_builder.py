from __future__ import annotations

import time
from typing import cast

from graphon.model_runtime.entities.llm_entities import LLMMode
from graphon.nodes.base.entities import OutputVariableType
from graphon.nodes.end.end_node import EndNode
from graphon.nodes.end.entities import EndNodeData
from graphon.nodes.llm import LLMNodeData, ModelConfig
from graphon.nodes.llm.runtime_protocols import PreparedLLMProtocol
from graphon.nodes.start.entities import StartNodeData
from graphon.runtime.graph_runtime_state import GraphRuntimeState
from graphon.runtime.variable_pool import VariablePool
from graphon.workflow_builder import (
    NodeOutputRef,
    WorkflowBuilder,
    WorkflowRuntime,
    completion_prompt,
    paragraph_input,
    system,
    template,
    text_input,
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
    graph_runtime_state = GraphRuntimeState(
        variable_pool=VariablePool(),
        start_at=time.time(),
    )
    builder = WorkflowBuilder()

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

    workflow = builder.build()
    graph = workflow.materialize(
        WorkflowRuntime(
            workflow_id="parallel-translation",
            graph_runtime_state=graph_runtime_state,
            prepared_llm=cast(PreparedLLMProtocol, object()),
        ),
    )

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


def test_workflow_builder_helpers_produce_typed_authoring_values() -> None:
    ref = NodeOutputRef(node_id="llm", output_name="text")
    prompt = completion_prompt("Answer in one sentence: ", ref)
    binding = ref.output("answer", value_type=OutputVariableType.STRING)
    text = text_input("question", required=True, max_length=512)

    assert template("Result: ", ref).render() == "Result: {{#llm.text#}}"
    assert prompt.text == "Answer in one sentence: {{#llm.text#}}"
    assert binding.variable == "answer"
    assert tuple(binding.value_selector) == ("llm", "text")
    assert binding.value_type is OutputVariableType.STRING
    assert text.variable == "question"
    assert text.type.value == "text-input"
    assert text.max_length == 512
