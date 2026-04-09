from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, final

from graphon.entities.base_node_data import BaseNodeData
from graphon.entities.graph_init_params import GraphInitParams
from graphon.enums import BuiltinNodeTypes
from graphon.file.enums import FileType
from graphon.file.models import File
from graphon.graph.graph import Graph
from graphon.model_runtime.entities.llm_entities import LLMMode
from graphon.model_runtime.entities.message_entities import (
    PromptMessage,
    PromptMessageRole,
)
from graphon.nodes.answer.answer_node import AnswerNode
from graphon.nodes.base.entities import OutputVariableEntity, OutputVariableType
from graphon.nodes.base.node import Node
from graphon.nodes.end.end_node import EndNode
from graphon.nodes.llm import LLMNode, LLMNodeChatModelMessage
from graphon.nodes.llm.file_saver import LLMFileSaver
from graphon.nodes.llm.runtime_protocols import (
    PreparedLLMProtocol,
    PromptMessageSerializerProtocol,
)
from graphon.nodes.start import StartNode
from graphon.runtime.graph_runtime_state import GraphRuntimeState
from graphon.variables.input_entities import VariableEntity, VariableEntityType

type TemplatePart = object
type NodeBuilder = Callable[[NodeBuildContext], Node]


@dataclass(frozen=True, slots=True)
class NodeBuildContext:
    node_id: str
    data: BaseNodeData
    graph_init_params: GraphInitParams
    graph_runtime_state: GraphRuntimeState


@dataclass(frozen=True, slots=True)
class NodeOutputRef:
    node_id: str
    output_name: str

    @property
    def selector(self) -> tuple[str, str]:
        return (self.node_id, self.output_name)

    def as_template(self) -> str:
        return "{{#" + ".".join(self.selector) + "#}}"

    def output(
        self,
        variable: str | None = None,
        *,
        value_type: OutputVariableType = OutputVariableType.ANY,
    ) -> OutputVariableEntity:
        return OutputVariableEntity(
            variable=variable or self.output_name,
            value_type=value_type,
            value_selector=self.selector,
        )

    def __str__(self) -> str:
        return self.as_template()


@dataclass(frozen=True, slots=True)
class NodeHandle:
    _builder: WorkflowBuilder
    node_id: str

    def then(
        self,
        node_id: str,
        data: BaseNodeData,
        *,
        source_handle: str = "source",
    ) -> NodeHandle:
        return self._builder.add_node(
            node_id=node_id,
            data=data,
            from_node_id=self.node_id,
            source_handle=source_handle,
        )

    def connect(
        self,
        target: NodeHandle,
        *,
        source_handle: str = "source",
    ) -> NodeHandle:
        return self._builder.connect(
            tail=self,
            head=target,
            source_handle=source_handle,
        )

    def ref(self, output_name: str) -> NodeOutputRef:
        return NodeOutputRef(node_id=self.node_id, output_name=output_name)


@final
class _PassthroughPromptMessageSerializer:
    def serialize(
        self,
        *,
        model_mode: LLMMode,
        prompt_messages: Sequence[PromptMessage],
    ) -> object:
        _ = model_mode
        return list(prompt_messages)


@final
class _TextOnlyFileSaver:
    def save_binary_string(
        self,
        data: bytes,
        mime_type: str,
        file_type: FileType,
        extension_override: str | None = None,
    ) -> File:
        _ = data, mime_type, file_type, extension_override
        msg = "WorkflowBuilder default saver only supports text outputs."
        raise RuntimeError(msg)

    def save_remote_url(self, url: str, file_type: FileType) -> File:
        _ = url, file_type
        msg = "WorkflowBuilder default saver only supports text outputs."
        raise RuntimeError(msg)


class WorkflowBuilder:
    def __init__(
        self,
        *,
        graph_init_params: GraphInitParams,
        graph_runtime_state: GraphRuntimeState,
        prepared_llm: PreparedLLMProtocol | None = None,
        llm_file_saver: LLMFileSaver | None = None,
        prompt_message_serializer: PromptMessageSerializerProtocol | None = None,
        node_builders: Mapping[str, NodeBuilder] | None = None,
    ) -> None:
        self._graph_init_params = graph_init_params
        self._graph_runtime_state = graph_runtime_state
        self._prepared_llm = prepared_llm
        self._llm_file_saver = llm_file_saver or _TextOnlyFileSaver()
        self._prompt_message_serializer = (
            prompt_message_serializer or _PassthroughPromptMessageSerializer()
        )
        self._graph_builder = Graph.new()
        self._handles: dict[str, NodeHandle] = {}
        self._node_builders: dict[str, NodeBuilder] = dict(node_builders or {})

    def register_node_builder(self, node_type: str, builder: NodeBuilder) -> None:
        self._node_builders[node_type] = builder

    def register_node_class(
        self,
        node_cls: type[Node],
        *,
        extra_kwargs_factory: (
            Callable[[NodeBuildContext], Mapping[str, Any]] | None
        ) = None,
    ) -> None:
        def _builder(context: NodeBuildContext) -> Node:
            extra_kwargs = (
                dict(extra_kwargs_factory(context))
                if extra_kwargs_factory is not None
                else {}
            )
            return node_cls(
                **self._base_node_kwargs(context),
                **extra_kwargs,
            )

        self.register_node_builder(node_cls.node_type, _builder)

    def root(self, node_id: str, data: BaseNodeData) -> NodeHandle:
        node = self._build_node(node_id=node_id, data=data)
        self._graph_builder.add_root(node)
        return self._remember_handle(node_id)

    def add_node(
        self,
        *,
        node_id: str,
        data: BaseNodeData,
        from_node_id: str,
        source_handle: str = "source",
    ) -> NodeHandle:
        node = self._build_node(node_id=node_id, data=data)
        self._graph_builder.add_node(
            node,
            from_node_id=from_node_id,
            source_handle=source_handle,
        )
        return self._remember_handle(node_id)

    def connect(
        self,
        *,
        tail: NodeHandle,
        head: NodeHandle,
        source_handle: str = "source",
    ) -> NodeHandle:
        self._ensure_owned_handle(tail)
        self._ensure_owned_handle(head)
        self._graph_builder.connect(
            tail=tail.node_id,
            head=head.node_id,
            source_handle=source_handle,
        )
        return head

    def handle(self, node_id: str) -> NodeHandle:
        try:
            return self._handles[node_id]
        except KeyError as error:
            msg = f"Unknown node id {node_id!r}."
            raise KeyError(msg) from error

    def build(self) -> Graph:
        return self._graph_builder.build()

    def _remember_handle(self, node_id: str) -> NodeHandle:
        handle = NodeHandle(_builder=self, node_id=node_id)
        self._handles[node_id] = handle
        return handle

    def _build_node(self, *, node_id: str, data: BaseNodeData) -> Node:
        context = NodeBuildContext(
            node_id=node_id,
            data=data,
            graph_init_params=self._graph_init_params,
            graph_runtime_state=self._graph_runtime_state,
        )

        custom_builder = self._node_builders.get(data.type)
        if custom_builder is not None:
            return custom_builder(context)

        if data.type == BuiltinNodeTypes.START:
            return StartNode(**self._base_node_kwargs(context))
        if data.type == BuiltinNodeTypes.ANSWER:
            return AnswerNode(**self._base_node_kwargs(context))
        if data.type == BuiltinNodeTypes.END:
            return EndNode(**self._base_node_kwargs(context))
        if data.type == BuiltinNodeTypes.LLM:
            if self._prepared_llm is None:
                msg = "LLM nodes require `prepared_llm` when using WorkflowBuilder."
                raise ValueError(msg)
            return LLMNode(
                **self._base_node_kwargs(context),
                model_instance=self._prepared_llm,
                llm_file_saver=self._llm_file_saver,
                prompt_message_serializer=self._prompt_message_serializer,
            )

        msg = (
            f"No node builder registered for node type {data.type!r}. "
            "Use `register_node_builder()` or `register_node_class()`."
        )
        raise ValueError(msg)

    def _base_node_kwargs(self, context: NodeBuildContext) -> dict[str, object]:
        return {
            "node_id": context.node_id,
            "config": {"id": context.node_id, "data": context.data},
            "graph_init_params": context.graph_init_params,
            "graph_runtime_state": context.graph_runtime_state,
        }

    def _ensure_owned_handle(self, handle: NodeHandle) -> None:
        if handle._builder is not self:
            msg = "NodeHandle belongs to a different WorkflowBuilder instance."
            raise ValueError(msg)


def template(*parts: TemplatePart) -> str:
    rendered: list[str] = []
    for part in parts:
        if isinstance(part, NodeOutputRef):
            rendered.append(part.as_template())
        else:
            rendered.append(str(part))
    return "".join(rendered)


def chat_message(
    role: PromptMessageRole,
    *parts: TemplatePart,
) -> LLMNodeChatModelMessage:
    return LLMNodeChatModelMessage(role=role, text=template(*parts))


def system(*parts: TemplatePart) -> LLMNodeChatModelMessage:
    return chat_message(PromptMessageRole.SYSTEM, *parts)


def user(*parts: TemplatePart) -> LLMNodeChatModelMessage:
    return chat_message(PromptMessageRole.USER, *parts)


def assistant(*parts: TemplatePart) -> LLMNodeChatModelMessage:
    return chat_message(PromptMessageRole.ASSISTANT, *parts)


def input_variable(
    variable: str,
    *,
    label: str | None = None,
    variable_type: VariableEntityType = VariableEntityType.PARAGRAPH,
    required: bool = False,
    **kwargs: Any,
) -> VariableEntity:
    return VariableEntity(
        variable=variable,
        label=label or variable.replace("_", " ").title(),
        type=variable_type,
        required=required,
        **kwargs,
    )


def paragraph_input(
    variable: str,
    *,
    label: str | None = None,
    required: bool = False,
    **kwargs: Any,
) -> VariableEntity:
    return input_variable(
        variable,
        label=label,
        variable_type=VariableEntityType.PARAGRAPH,
        required=required,
        **kwargs,
    )


__all__ = [
    "NodeBuildContext",
    "NodeHandle",
    "NodeOutputRef",
    "WorkflowBuilder",
    "assistant",
    "chat_message",
    "input_variable",
    "paragraph_input",
    "system",
    "template",
    "user",
]
