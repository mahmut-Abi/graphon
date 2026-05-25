from __future__ import annotations

import json
import logging
from collections.abc import Generator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, assert_never, override

from graphon.entities.graph_init_params import GraphInitParams
from graphon.entities.pause_reason import HumanInputRequired
from graphon.enums import (
    BuiltinNodeTypes,
    NodeExecutionType,
    WorkflowNodeExecutionStatus,
)
from graphon.node_events.base import NodeEventBase, NodeRunResult
from graphon.node_events.node import (
    HumanInputFormFilledEvent,
    HumanInputFormTimeoutEvent,
    PauseRequestedEvent,
    StreamCompletedEvent,
)
from graphon.nodes.base.node import Node
from graphon.nodes.protocols import FileReferenceFactoryProtocol
from graphon.nodes.runtime import (
    HumanInputFormStateProtocol,
    HumanInputNodeRuntimeProtocol,
    _HumanInputRuntimeLike,
    _normalize_human_input_runtime,
)
from graphon.runtime.graph_runtime_state import GraphRuntimeState
from graphon.variables.factory import build_segment
from graphon.variables.segments import Segment
from graphon.workflow_type_encoder import WorkflowRuntimeTypeConverter

from . import _exc as exc
from .entities import (
    FileInputConfig,
    FileListInputConfig,
    FormInputConfig,
    HumanInputNodeData,
    ParagraphInputConfig,
    SelectInputConfig,
)
from .enums import HumanInputFormStatus

_SELECTED_BRANCH_KEY = "selected_branch"


logger = logging.getLogger(__name__)


class HumanInputNode(Node[HumanInputNodeData]):
    node_type = BuiltinNodeTypes.HUMAN_INPUT
    execution_type = NodeExecutionType.BRANCH

    _BRANCH_SELECTION_KEYS: tuple[str, ...] = (
        "edge_source_handle",
        "edgeSourceHandle",
        "source_handle",
        _SELECTED_BRANCH_KEY,
        "selectedBranch",
        "branch",
        "branch_id",
        "branchId",
        "handle",
    )

    _node_data: HumanInputNodeData
    _OUTPUT_FIELD_ACTION_ID = "__action_id"
    _OUTPUT_FIELD_ACTION_VALUE = "__action_value"
    _OUTPUT_FIELD_RENDERED_CONTENT = "__rendered_content"
    _TIMEOUT_HANDLE = _TIMEOUT_ACTION_ID = "__timeout"

    @override
    def __init__(
        self,
        node_id: str,
        data: HumanInputNodeData,
        *,
        graph_init_params: GraphInitParams,
        graph_runtime_state: GraphRuntimeState,
        # TODO @-LAN: See https://github.com/langgenius/graphon/issues/new/choose.  # noqa: FIX002
        # Make `runtime` optional once Graphon provides a default human-input
        # runtime adapter instead of requiring an embedding-specific implementation.
        runtime: _HumanInputRuntimeLike,
        file_reference_factory: FileReferenceFactoryProtocol,
        form_repository: object | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            data=data,
            graph_init_params=graph_init_params,
            graph_runtime_state=graph_runtime_state,
        )
        self._runtime: HumanInputNodeRuntimeProtocol = _normalize_human_input_runtime(
            runtime,
            form_repository=form_repository,
        )
        self._file_reference_factory = file_reference_factory

    @classmethod
    @override
    def version(cls) -> str:
        return "1"

    def _resolve_branch_selection(self) -> str | None:
        """Determine the branch handle selected by human input if available."""
        variable_pool = self.graph_runtime_state.variable_pool

        for key in self._BRANCH_SELECTION_KEYS:
            handle = self._extract_branch_handle(variable_pool.get((self.id, key)))
            if handle:
                return handle

        default_values = self.node_data.default_value_dict
        for key in self._BRANCH_SELECTION_KEYS:
            handle = self._normalize_branch_value(default_values.get(key))
            if handle:
                return handle

        return None

    @staticmethod
    def _extract_branch_handle(segment: Any) -> str | None:
        if segment is None:
            return None

        candidate = getattr(segment, "to_object", None)
        raw_value = (
            candidate() if callable(candidate) else getattr(segment, "value", None)
        )
        if raw_value is None:
            return None

        return HumanInputNode._normalize_branch_value(raw_value)

    @staticmethod
    def _normalize_branch_value(value: Any) -> str | None:
        if value is None:
            return None

        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None

        if isinstance(value, Mapping):
            for key in (
                "handle",
                "edge_source_handle",
                "edgeSourceHandle",
                "branch",
                "id",
                "value",
            ):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate:
                    return candidate

        return None

    def _form_to_pause_event(
        self,
        form_entity: HumanInputFormStateProtocol,
    ) -> PauseRequestedEvent:
        required_event = self._human_input_required_event(form_entity)
        return PauseRequestedEvent(reason=required_event)

    def resolve_default_values(self) -> Mapping[str, Any]:
        variable_pool = self.graph_runtime_state.variable_pool
        resolved_defaults: dict[str, Any] = {}
        for form_input in self._node_data.inputs:
            resolved_default = form_input.resolve_default_value(variable_pool)
            if resolved_default is None:
                # Treat missing variable-backed defaults as absent defaults.
                continue
            resolved_defaults[form_input.output_variable_name] = (
                WorkflowRuntimeTypeConverter().value_to_json_encodable_recursive(
                    resolved_default.value,
                )
            )

        return resolved_defaults

    def _human_input_required_event(
        self,
        form_entity: HumanInputFormStateProtocol,
    ) -> HumanInputRequired:
        node_data = self._node_data
        resolved_default_values = self.resolve_default_values()
        return HumanInputRequired(
            form_id=form_entity.id,
            form_content=form_entity.rendered_content,
            inputs=node_data.inputs,
            actions=node_data.user_actions,
            node_id=self.id,
            node_title=node_data.title,
            resolved_default_values=resolved_default_values,
        )

    @override
    def _run(self) -> Generator[NodeEventBase, None, None]:
        """Execute the human input node.

        This method will:
        1. Generate a unique form ID
        2. Create form content with variable substitution
        3. Persist the form through the configured repository
        4. Send form via configured delivery methods
        5. Suspend workflow execution
        6. Wait for form submission to resume

        Yields:
            Node events describing form suspension, timeout, or submitted output.

        Raises:
            AssertionError: If a submitted form is missing its selected action id.

        """
        form = self._runtime.get_form(node_id=self.id)
        if form is None:
            form_entity = self._runtime.create_form(
                node_id=self.id,
                node_data=self._node_data,
                rendered_content=self.render_form_content_before_submission(),
                resolved_default_values=self.resolve_default_values(),
            )

            logger.info(
                "Human Input node suspended workflow for form. node_id=%s, form_id=%s",
                self.id,
                form_entity.id,
            )
            yield self._form_to_pause_event(form_entity)
            return

        if form.status in frozenset((
            HumanInputFormStatus.TIMEOUT,
            HumanInputFormStatus.EXPIRED,
        )) or form.expiration_time <= datetime.now(UTC).replace(tzinfo=None):
            yield HumanInputFormTimeoutEvent(
                node_title=self._node_data.title,
                expiration_time=form.expiration_time,
            )
            yield StreamCompletedEvent(
                node_run_result=NodeRunResult(
                    status=WorkflowNodeExecutionStatus.SUCCEEDED,
                    outputs=self._build_special_outputs(
                        action_id="",
                        action_value="",
                        rendered_content=form.rendered_content,
                    ),
                    edge_source_handle=self._TIMEOUT_HANDLE,
                ),
            )
            return

        if not form.submitted:
            yield self._form_to_pause_event(form)
            return

        selected_action_id = form.selected_action_id
        if selected_action_id is None:
            msg = (
                f"selected_action_id should not be None when form submitted, "
                f"form_id={form.id}"
            )
            raise AssertionError(msg)
        restored_submission_data = self._restore_submitted_data(
            submitted_data=form.submitted_data or {},
        )
        inputs_by_name = {
            form_input.output_variable_name: form_input
            for form_input in self._node_data.inputs
        }
        submitted_data: dict[str, Segment] = {}
        for name, value in restored_submission_data.items():
            if name not in inputs_by_name:
                logger.error("unexpected form data in submitted data, key=%s", name)
                continue
            submitted_data[name] = value
        selected_action_value = next(
            ua.title
            for ua in self._node_data.user_actions
            if ua.id == selected_action_id
        )
        rendered_content = self.render_form_content_with_outputs(
            form.rendered_content,
            submitted_data,
            self._node_data.outputs_field_names(),
            self._node_data.inputs,
        )
        outputs = dict(submitted_data) | self._build_special_outputs(
            action_id=selected_action_id,
            action_value=selected_action_value,
            rendered_content=rendered_content,
        )

        action_text = self._node_data.find_action_text(selected_action_id)

        yield HumanInputFormFilledEvent(
            node_title=self._node_data.title,
            rendered_content=rendered_content,
            action_id=selected_action_id,
            action_text=action_text,
            submitted_data=submitted_data,
        )

        yield StreamCompletedEvent(
            node_run_result=NodeRunResult(
                status=WorkflowNodeExecutionStatus.SUCCEEDED,
                inputs=submitted_data,
                outputs=outputs,
                edge_source_handle=selected_action_id,
            ),
        )

    def render_form_content_before_submission(self) -> str:
        """Process form content by substituting variables.

        This method should:
        1. Parse the form_content markdown
        2. Substitute {{#node_name.var_name#}} with actual values
        3. Keep {{#$output.field_name#}} placeholders for form inputs

        Returns:
            Rendered markdown with runtime variable references resolved.

        """
        rendered_form_content = self.graph_runtime_state.variable_pool.convert_template(
            self._node_data.form_content,
        )
        return rendered_form_content.markdown

    @staticmethod
    def render_form_content_with_outputs(
        form_content: str,
        outputs: Mapping[str, Any],
        field_names: Sequence[str],
        form_inputs: Sequence[FormInputConfig] | None = None,
    ) -> str:
        """Replace {{#$output.xxx#}} placeholders with submitted values.

        Text inputs render their submitted value directly. File inputs render as
        stable placeholders so the final content stays readable and does not
        inline transport metadata.

        Returns:
            the interplated form content
        """
        inputs_by_name = {}
        if form_inputs is not None:
            inputs_by_name = {
                form_input.output_variable_name: form_input
                for form_input in form_inputs
            }

        rendered_content = form_content
        for field_name in field_names:
            placeholder = "{{#$output." + field_name + "#}}"
            replacement = HumanInputNode._render_output_placeholder_value(
                value=outputs.get(field_name),
                form_input=inputs_by_name.get(field_name),
            )
            rendered_content = rendered_content.replace(placeholder, replacement)
        return rendered_content

    @staticmethod
    def _render_output_placeholder_value(
        *,
        value: Any,
        form_input: FormInputConfig | None,
    ) -> str:
        if isinstance(value, Segment):
            value = WorkflowRuntimeTypeConverter().value_to_json_encodable_recursive(
                value,
            )

        if value is None:
            return ""

        if isinstance(form_input, FileInputConfig):
            return "[file]"

        if isinstance(form_input, FileListInputConfig):
            file_count = 0
            if isinstance(value, Sequence) and not isinstance(value, str | bytes):
                file_count = len(value)
            return f"[{file_count} files]"

        if isinstance(form_input, ParagraphInputConfig | SelectInputConfig):
            return str(value)

        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)

        return str(value)

    def _restore_submitted_data(
        self,
        *,
        submitted_data: Mapping[str, Any],
    ) -> dict[str, Segment]:
        """_restore_submitted_data restruct python data types from
        **validated form data**.

        Returns:
            A mapping from input field names to their corresponding
            graphon runtime values.

        Raises:
            InvalidSubmittedDataError: if submission data are invalid.
        """
        # NOTE: ideally this logic shoule be integrated into
        # `HumanInputFormStateProtocol.submitted_data`.

        restored_data: dict[str, Segment] = {}
        inputs_by_name = {
            form_input.output_variable_name: form_input
            for form_input in self._node_data.inputs
        }

        for name, value in submitted_data.items():
            form_input = inputs_by_name.get(name)
            if form_input is None:
                restored_data[name] = build_segment(value)
                continue

            match form_input:
                case FileInputConfig():
                    if not isinstance(value, Mapping):
                        msg = (
                            "HumanInput file input expects a mapping payload, "
                            f"output_variable_name={name}, got={type(value).__name__}"
                        )
                        raise exc.InvalidSubmittedDataError(msg)
                    restored_data[name] = build_segment(
                        self._restore_file_value(
                            output_variable_name=name,
                            value=value,
                        )
                    )
                case FileListInputConfig():
                    if not isinstance(value, list):
                        msg = (
                            "HumanInput file list input expects a list payload, "
                            f"output_variable_name={name}, got={type(value).__name__}"
                        )
                        raise exc.InvalidSubmittedDataError(msg)
                    if not all(isinstance(item, Mapping) for item in value):
                        msg = (
                            "HumanInput file list input expects list items to be "
                            "mapping payloads, "
                            f"output_variable_name={name}"
                        )
                        raise exc.InvalidSubmittedDataError(msg)
                    restored_data[name] = build_segment([
                        self._restore_file_value(
                            output_variable_name=name,
                            value=item,
                        )
                        for item in value
                    ])
                case ParagraphInputConfig() | SelectInputConfig():
                    if not isinstance(value, str):
                        msg = (
                            "HumanInput file list input expects a string, "
                            f"output_variable_name={name}, got={type(value).__name__}"
                        )
                        raise exc.InvalidSubmittedDataError(msg)
                    restored_data[name] = build_segment(value)
                case _:
                    assert_never(form_input)

        return restored_data

    def _restore_file_value(
        self,
        *,
        output_variable_name: str,
        value: Any,
    ) -> Any:
        _ = output_variable_name
        return self._file_reference_factory.build_from_mapping(mapping=value)

    @classmethod
    def _build_special_outputs(
        cls,
        *,
        action_id: str,
        action_value: str,
        rendered_content: str,
    ) -> dict[str, Segment]:
        return {
            cls._OUTPUT_FIELD_ACTION_ID: build_segment(action_id),
            cls._OUTPUT_FIELD_RENDERED_CONTENT: build_segment(rendered_content),
            cls._OUTPUT_FIELD_ACTION_VALUE: build_segment(action_value),
        }

    @classmethod
    @override
    def _extract_variable_selector_to_variable_mapping(
        cls,
        *,
        graph_config: Mapping[str, Any],
        node_id: str,
        node_data: HumanInputNodeData,
    ) -> Mapping[str, Sequence[str]]:
        """Extract variable selectors referenced in form content
        and input default values.

        This method should parse:
        1. Variables referenced in form_content ({{#node_name.var_name#}})
        2. Variables referenced in input default values

        Returns:
            Mapping of local reference keys to the referenced variable selectors.

        """
        _ = graph_config
        return node_data.extract_variable_selector_to_variable_mapping(node_id)
