from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast, final

from graphon.entities.base_node_data import BaseNodeData
from graphon.entities.graph_config import NodeConfigDict
from graphon.entities.graph_init_params import GraphInitParams
from graphon.enums import NodeType
from graphon.file.enums import FileTransferMethod, FileType
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
from graphon.nodes.llm import (
    LLMNode,
    LLMNodeChatModelMessage,
    LLMNodeCompletionModelPromptTemplate,
    LLMNodeData,
)
from graphon.nodes.llm.file_saver import LLMFileSaver
from graphon.nodes.llm.runtime_protocols import (
    PreparedLLMProtocol,
    PromptMessageSerializerProtocol,
)
from graphon.nodes.start import StartNode
from graphon.runtime.graph_runtime_state import GraphRuntimeState
from graphon.variables.input_entities import VariableEntity, VariableEntityType


@dataclass(frozen=True, slots=True)
class WorkflowRuntime:
    workflow_id: str
    graph_runtime_state: GraphRuntimeState
    run_context: Mapping[str, Any] = field(default_factory=dict)
    call_depth: int = 0
    prepared_llm: PreparedLLMProtocol | None = None
    llm_file_saver: LLMFileSaver | None = None
    prompt_message_serializer: PromptMessageSerializerProtocol | None = None

    @classmethod
    def from_graph_init_params(
        cls,
        graph_init_params: GraphInitParams,
        *,
        graph_runtime_state: GraphRuntimeState,
        prepared_llm: PreparedLLMProtocol | None = None,
        llm_file_saver: LLMFileSaver | None = None,
        prompt_message_serializer: PromptMessageSerializerProtocol | None = None,
    ) -> WorkflowRuntime:
        return cls(
            workflow_id=graph_init_params.workflow_id,
            graph_runtime_state=graph_runtime_state,
            run_context=graph_init_params.run_context,
            call_depth=graph_init_params.call_depth,
            prepared_llm=prepared_llm,
            llm_file_saver=llm_file_saver,
            prompt_message_serializer=prompt_message_serializer,
        )

    def create_graph_init_params(
        self,
        *,
        graph_config: Mapping[str, Any],
    ) -> GraphInitParams:
        return GraphInitParams(
            workflow_id=self.workflow_id,
            graph_config=graph_config,
            run_context=dict(self.run_context),
            call_depth=self.call_depth,
        )


@dataclass(frozen=True, slots=True)
class NodeMaterializationContext[NodeDataT: BaseNodeData]:
    node_id: str
    data: NodeDataT
    runtime: WorkflowRuntime
    graph_init_params: GraphInitParams
    graph_runtime_state: GraphRuntimeState


type NodeMaterializer[NodeDataT: BaseNodeData] = Callable[
    [NodeMaterializationContext[NodeDataT]],
    Node[NodeDataT],
]


@dataclass(frozen=True, slots=True)
class _RegisteredNodeMaterializer[NodeDataT: BaseNodeData]:
    data_type: type[NodeDataT]
    materializer: NodeMaterializer[NodeDataT]


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
    ) -> OutputBinding:
        return OutputBinding.from_ref(
            self,
            variable=variable,
            value_type=value_type,
        )

    def __str__(self) -> str:
        return self.as_template()


class OutputBinding(OutputVariableEntity):
    @classmethod
    def from_ref(
        cls,
        ref: NodeOutputRef,
        *,
        variable: str | None = None,
        value_type: OutputVariableType = OutputVariableType.ANY,
    ) -> OutputBinding:
        return cls(
            variable=variable or ref.output_name,
            value_type=value_type,
            value_selector=ref.selector,
        )

    @property
    def selector(self) -> tuple[str, ...]:
        return tuple(self.value_selector)


@dataclass(frozen=True, slots=True)
class TemplateExpr:
    parts: tuple[str | NodeOutputRef, ...]

    def render(self) -> str:
        return "".join(
            part.as_template() if isinstance(part, NodeOutputRef) else part
            for part in self.parts
        )

    def __str__(self) -> str:
        return self.render()

    @classmethod
    def from_parts(
        cls,
        *parts: str | NodeOutputRef | TemplateExpr,
    ) -> TemplateExpr:
        normalized_parts: list[str | NodeOutputRef] = []
        for part in parts:
            if isinstance(part, TemplateExpr):
                normalized_parts.extend(part.parts)
            elif isinstance(part, (str, NodeOutputRef)):
                normalized_parts.append(part)
            else:
                msg = (
                    "Template expressions only support string literals, "
                    "NodeOutputRef values, or other TemplateExpr instances."
                )
                raise TypeError(msg)
        return cls(parts=tuple(normalized_parts))


@dataclass(frozen=True, slots=True)
class WorkflowNodeSpec[NodeDataT: BaseNodeData]:
    node_id: str
    data: NodeDataT

    def as_node_config(self) -> NodeConfigDict:
        return {"id": self.node_id, "data": self.data}


@dataclass(frozen=True, slots=True)
class WorkflowEdgeSpec:
    tail: str
    head: str
    source_handle: str = "source"

    def as_edge_config(self) -> dict[str, str]:
        return {
            "source": self.tail,
            "target": self.head,
            "sourceHandle": self.source_handle,
        }


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    root_node_id: str
    nodes: tuple[WorkflowNodeSpec[BaseNodeData], ...]
    edges: tuple[WorkflowEdgeSpec, ...]

    @property
    def graph_config(self) -> dict[str, list[object]]:
        return {
            "nodes": [node.as_node_config() for node in self.nodes],
            "edges": [edge.as_edge_config() for edge in self.edges],
        }

    def materialize(self, runtime: WorkflowRuntime) -> Graph:
        return WorkflowMaterializer(runtime=runtime).materialize(self)


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
    def __init__(self) -> None:
        self._node_order: list[str] = []
        self._node_specs: dict[str, WorkflowNodeSpec[BaseNodeData]] = {}
        self._edges: list[WorkflowEdgeSpec] = []
        self._handles: dict[str, NodeHandle] = {}
        self._root_node_id: str | None = None

    def root(self, node_id: str, data: BaseNodeData) -> NodeHandle:
        if self._root_node_id is not None:
            msg = f"Root node has already been set to {self._root_node_id!r}."
            raise ValueError(msg)
        self._store_node(node_id=node_id, data=data)
        self._root_node_id = node_id
        return self._remember_handle(node_id)

    def add_node(
        self,
        *,
        node_id: str,
        data: BaseNodeData,
        from_node_id: str,
        source_handle: str = "source",
    ) -> NodeHandle:
        if from_node_id not in self._node_specs:
            msg = f"Predecessor node {from_node_id!r} is not registered."
            raise ValueError(msg)
        self._store_node(node_id=node_id, data=data)
        self._edges.append(
            WorkflowEdgeSpec(
                tail=from_node_id,
                head=node_id,
                source_handle=source_handle,
            ),
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
        self._edges.append(
            WorkflowEdgeSpec(
                tail=tail.node_id,
                head=head.node_id,
                source_handle=source_handle,
            ),
        )
        return head

    def handle(self, node_id: str) -> NodeHandle:
        try:
            return self._handles[node_id]
        except KeyError as error:
            msg = f"Unknown node id {node_id!r}."
            raise KeyError(msg) from error

    def build(self) -> WorkflowSpec:
        if self._root_node_id is None:
            msg = "WorkflowBuilder requires a root node before build()."
            raise ValueError(msg)
        return WorkflowSpec(
            root_node_id=self._root_node_id,
            nodes=tuple(self._node_specs[node_id] for node_id in self._node_order),
            edges=tuple(self._edges),
        )

    def materialize(self, runtime: WorkflowRuntime) -> Graph:
        return self.build().materialize(runtime)

    def _remember_handle(self, node_id: str) -> NodeHandle:
        handle = NodeHandle(_builder=self, node_id=node_id)
        self._handles[node_id] = handle
        return handle

    def _store_node(self, *, node_id: str, data: BaseNodeData) -> None:
        if node_id in self._node_specs:
            msg = f"Node id {node_id!r} is already registered."
            raise ValueError(msg)
        self._node_order.append(node_id)
        self._node_specs[node_id] = WorkflowNodeSpec(node_id=node_id, data=data)

    def _ensure_owned_handle(self, handle: NodeHandle) -> None:
        if handle._builder is not self:
            msg = "NodeHandle belongs to a different WorkflowBuilder instance."
            raise ValueError(msg)


@final
class _WorkflowNodeFactory:
    def __init__(
        self,
        *,
        runtime: WorkflowRuntime,
        graph_init_params: GraphInitParams,
        registrations_by_type: Mapping[
            type[BaseNodeData],
            _RegisteredNodeMaterializer[BaseNodeData],
        ],
        registrations_by_identity: Mapping[
            tuple[NodeType, str],
            _RegisteredNodeMaterializer[BaseNodeData],
        ],
    ) -> None:
        self._runtime = runtime
        self._graph_init_params = graph_init_params
        self._registrations_by_type = registrations_by_type
        self._registrations_by_identity = registrations_by_identity

    def create_node(self, node_config: NodeConfigDict) -> Node:
        node_id = node_config["id"]
        node_data = node_config["data"]
        registration = self._resolve_registration(node_data)
        typed_node_data = self._coerce_node_data(node_data, registration.data_type)
        context = NodeMaterializationContext(
            node_id=node_id,
            data=typed_node_data,
            runtime=self._runtime,
            graph_init_params=self._graph_init_params,
            graph_runtime_state=self._runtime.graph_runtime_state,
        )
        materializer = cast(
            "NodeMaterializer[BaseNodeData]",
            registration.materializer,
        )
        return materializer(context)

    def _resolve_registration(
        self,
        node_data: BaseNodeData,
    ) -> _RegisteredNodeMaterializer[BaseNodeData]:
        for data_type in type(node_data).__mro__:
            if not issubclass(data_type, BaseNodeData):
                continue
            registration = self._registrations_by_type.get(data_type)
            if registration is not None:
                return registration

        registration = self._registrations_by_identity.get(
            (node_data.type, node_data.version),
        )
        if registration is not None:
            return registration

        msg = (
            "No node materializer registered for "
            f"{type(node_data).__name__} "
            f"(type={node_data.type!r}, version={node_data.version!r}). "
            "Use `WorkflowMaterializer.register_node_materializer()` "
            "or `WorkflowMaterializer.register_node_class()`."
        )
        raise ValueError(msg)

    @staticmethod
    def _coerce_node_data[NodeDataT: BaseNodeData](
        node_data: BaseNodeData,
        data_type: type[NodeDataT],
    ) -> NodeDataT:
        if isinstance(node_data, data_type):
            return node_data
        return data_type.model_validate(node_data.model_dump(mode="python"))


class WorkflowMaterializer:
    def __init__(self, *, runtime: WorkflowRuntime) -> None:
        self._runtime = runtime
        self._llm_file_saver = runtime.llm_file_saver or _TextOnlyFileSaver()
        self._prompt_message_serializer = (
            runtime.prompt_message_serializer or _PassthroughPromptMessageSerializer()
        )
        self._registrations_by_type: dict[
            type[BaseNodeData],
            _RegisteredNodeMaterializer[BaseNodeData],
        ] = {}
        self._registrations_by_identity: dict[
            tuple[NodeType, str],
            _RegisteredNodeMaterializer[BaseNodeData],
        ] = {}
        self.register_node_class(StartNode)
        self.register_node_class(AnswerNode)
        self.register_node_class(EndNode)
        self.register_node_materializer(LLMNodeData, self._materialize_llm_node)

    @property
    def runtime(self) -> WorkflowRuntime:
        return self._runtime

    def register_node_materializer[NodeDataT: BaseNodeData](
        self,
        data_type: type[NodeDataT],
        materializer: NodeMaterializer[NodeDataT],
    ) -> None:
        registration = _RegisteredNodeMaterializer(
            data_type=data_type,
            materializer=materializer,
        )
        self._registrations_by_type[data_type] = cast(
            "_RegisteredNodeMaterializer[BaseNodeData]",
            registration,
        )

        identity = self._extract_node_identity(data_type)
        if identity is not None:
            self._registrations_by_identity[identity] = cast(
                "_RegisteredNodeMaterializer[BaseNodeData]",
                registration,
            )

    def register_node_class[NodeDataT: BaseNodeData](
        self,
        node_cls: type[Node[NodeDataT]],
        *,
        extra_kwargs_factory: (
            Callable[[NodeMaterializationContext[NodeDataT]], Mapping[str, object]]
            | None
        ) = None,
    ) -> None:
        data_type = cast("type[NodeDataT]", node_cls._node_data_type)

        def _materializer(context: NodeMaterializationContext[NodeDataT]) -> Node:
            extra_kwargs = (
                dict(extra_kwargs_factory(context))
                if extra_kwargs_factory is not None
                else {}
            )
            return node_cls(
                **self._base_node_kwargs(context),
                **extra_kwargs,
            )

        self.register_node_materializer(data_type, _materializer)

    def materialize(self, workflow: WorkflowSpec) -> Graph:
        graph_config = workflow.graph_config
        graph_init_params = self._runtime.create_graph_init_params(
            graph_config=graph_config,
        )
        node_factory = _WorkflowNodeFactory(
            runtime=self._runtime,
            graph_init_params=graph_init_params,
            registrations_by_type=self._registrations_by_type,
            registrations_by_identity=self._registrations_by_identity,
        )
        return Graph.init(
            graph_config=graph_config,
            node_factory=node_factory,
            root_node_id=workflow.root_node_id,
        )

    def _materialize_llm_node(
        self,
        context: NodeMaterializationContext[LLMNodeData],
    ) -> LLMNode:
        if context.runtime.prepared_llm is None:
            msg = "LLM nodes require `prepared_llm` when materializing a workflow."
            raise ValueError(msg)
        return LLMNode(
            **self._base_node_kwargs(context),
            model_instance=context.runtime.prepared_llm,
            llm_file_saver=self._llm_file_saver,
            prompt_message_serializer=self._prompt_message_serializer,
        )

    @staticmethod
    def _base_node_kwargs[NodeDataT: BaseNodeData](
        context: NodeMaterializationContext[NodeDataT],
    ) -> dict[str, object]:
        return {
            "node_id": context.node_id,
            "config": {"id": context.node_id, "data": context.data},
            "graph_init_params": context.graph_init_params,
            "graph_runtime_state": context.graph_runtime_state,
        }

    @staticmethod
    def _extract_node_identity(
        data_type: type[BaseNodeData],
    ) -> tuple[NodeType, str] | None:
        type_field = data_type.model_fields.get("type")
        version_field = data_type.model_fields.get("version")
        if type_field is None or version_field is None:
            return None

        node_type = type_field.default
        version = version_field.default
        if not isinstance(node_type, str) or not isinstance(version, str):
            return None

        return (cast("NodeType", node_type), version)


def template(*parts: str | NodeOutputRef | TemplateExpr) -> TemplateExpr:
    return TemplateExpr.from_parts(*parts)


def chat_message(
    role: PromptMessageRole,
    *parts: str | NodeOutputRef | TemplateExpr,
) -> LLMNodeChatModelMessage:
    return LLMNodeChatModelMessage(role=role, text=template(*parts).render())


def system(*parts: str | NodeOutputRef | TemplateExpr) -> LLMNodeChatModelMessage:
    return chat_message(PromptMessageRole.SYSTEM, *parts)


def user(*parts: str | NodeOutputRef | TemplateExpr) -> LLMNodeChatModelMessage:
    return chat_message(PromptMessageRole.USER, *parts)


def assistant(
    *parts: str | NodeOutputRef | TemplateExpr,
) -> LLMNodeChatModelMessage:
    return chat_message(PromptMessageRole.ASSISTANT, *parts)


def completion_prompt(
    *parts: str | NodeOutputRef | TemplateExpr,
) -> LLMNodeCompletionModelPromptTemplate:
    return LLMNodeCompletionModelPromptTemplate(text=template(*parts).render())


def input_variable(
    variable: str,
    *,
    variable_type: VariableEntityType,
    label: str | None = None,
    description: str = "",
    required: bool = False,
    hide: bool = False,
    default: object | None = None,
    max_length: int | None = None,
    options: Sequence[str] = (),
    allowed_file_types: Sequence[FileType] | None = (),
    allowed_file_extensions: Sequence[str] | None = (),
    allowed_file_upload_methods: Sequence[FileTransferMethod] | None = (),
    json_schema: Mapping[str, Any] | None = None,
) -> VariableEntity:
    return VariableEntity(
        variable=variable,
        label=label or variable.replace("_", " ").title(),
        description=description,
        type=variable_type,
        required=required,
        hide=hide,
        default=default,
        max_length=max_length,
        options=list(options),
        allowed_file_types=list(allowed_file_types or []),
        allowed_file_extensions=list(allowed_file_extensions or []),
        allowed_file_upload_methods=list(allowed_file_upload_methods or []),
        json_schema=dict(json_schema) if json_schema is not None else None,
    )


def text_input(
    variable: str,
    *,
    label: str | None = None,
    description: str = "",
    required: bool = False,
    hide: bool = False,
    default: str | None = None,
    max_length: int | None = None,
) -> VariableEntity:
    return input_variable(
        variable,
        variable_type=VariableEntityType.TEXT_INPUT,
        label=label,
        description=description,
        required=required,
        hide=hide,
        default=default,
        max_length=max_length,
    )


def paragraph_input(
    variable: str,
    *,
    label: str | None = None,
    description: str = "",
    required: bool = False,
    hide: bool = False,
    default: str | None = None,
    max_length: int | None = None,
) -> VariableEntity:
    return input_variable(
        variable,
        variable_type=VariableEntityType.PARAGRAPH,
        label=label,
        description=description,
        required=required,
        hide=hide,
        default=default,
        max_length=max_length,
    )


def number_input(
    variable: str,
    *,
    label: str | None = None,
    description: str = "",
    required: bool = False,
    hide: bool = False,
    default: float | None = None,
) -> VariableEntity:
    return input_variable(
        variable,
        variable_type=VariableEntityType.NUMBER,
        label=label,
        description=description,
        required=required,
        hide=hide,
        default=default,
    )


def select_input(
    variable: str,
    *,
    options: Sequence[str],
    label: str | None = None,
    description: str = "",
    required: bool = False,
    hide: bool = False,
    default: str | None = None,
) -> VariableEntity:
    return input_variable(
        variable,
        variable_type=VariableEntityType.SELECT,
        label=label,
        description=description,
        required=required,
        hide=hide,
        default=default,
        options=options,
    )


def checkbox_input(
    variable: str,
    *,
    label: str | None = None,
    description: str = "",
    required: bool = False,
    hide: bool = False,
    default: bool | None = None,
) -> VariableEntity:
    return input_variable(
        variable,
        variable_type=VariableEntityType.CHECKBOX,
        label=label,
        description=description,
        required=required,
        hide=hide,
        default=default,
    )


def json_input(
    variable: str,
    *,
    label: str | None = None,
    description: str = "",
    required: bool = False,
    hide: bool = False,
    default: object | None = None,
    json_schema: Mapping[str, Any] | None = None,
) -> VariableEntity:
    return input_variable(
        variable,
        variable_type=VariableEntityType.JSON_OBJECT,
        label=label,
        description=description,
        required=required,
        hide=hide,
        default=default,
        json_schema=json_schema,
    )


def file_input(
    variable: str,
    *,
    label: str | None = None,
    description: str = "",
    required: bool = False,
    hide: bool = False,
    allowed_file_types: Sequence[FileType] | None = (),
    allowed_file_extensions: Sequence[str] | None = (),
    allowed_file_upload_methods: Sequence[FileTransferMethod] | None = (),
) -> VariableEntity:
    return input_variable(
        variable,
        variable_type=VariableEntityType.FILE,
        label=label,
        description=description,
        required=required,
        hide=hide,
        allowed_file_types=allowed_file_types,
        allowed_file_extensions=allowed_file_extensions,
        allowed_file_upload_methods=allowed_file_upload_methods,
    )


def file_list_input(
    variable: str,
    *,
    label: str | None = None,
    description: str = "",
    required: bool = False,
    hide: bool = False,
    allowed_file_types: Sequence[FileType] | None = (),
    allowed_file_extensions: Sequence[str] | None = (),
    allowed_file_upload_methods: Sequence[FileTransferMethod] | None = (),
) -> VariableEntity:
    return input_variable(
        variable,
        variable_type=VariableEntityType.FILE_LIST,
        label=label,
        description=description,
        required=required,
        hide=hide,
        allowed_file_types=allowed_file_types,
        allowed_file_extensions=allowed_file_extensions,
        allowed_file_upload_methods=allowed_file_upload_methods,
    )


__all__ = [
    "NodeHandle",
    "NodeMaterializationContext",
    "NodeOutputRef",
    "OutputBinding",
    "TemplateExpr",
    "WorkflowBuilder",
    "WorkflowEdgeSpec",
    "WorkflowMaterializer",
    "WorkflowNodeSpec",
    "WorkflowRuntime",
    "WorkflowSpec",
    "assistant",
    "chat_message",
    "checkbox_input",
    "completion_prompt",
    "file_input",
    "file_list_input",
    "input_variable",
    "json_input",
    "number_input",
    "paragraph_input",
    "select_input",
    "system",
    "template",
    "text_input",
    "user",
]
