from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from typing import Any, Self

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from graphon.model_runtime.entities.message_entities import ImagePromptMessageContent

from . import helpers
from .constants import FILE_MODEL_IDENTITY
from .enums import FileTransferMethod, FileType

_FILE_REFERENCE_PREFIX = "dify-file-ref:"
_REFERENCE_TRANSFER_METHODS = frozenset((
    FileTransferMethod.LOCAL_FILE,
    FileTransferMethod.TOOL_FILE,
    FileTransferMethod.DATASOURCE_FILE,
))
_LEGACY_REFERENCE_FIELDS = (
    "related_id",
    "tool_file_id",
    "upload_file_id",
    "datasource_file_id",
)


def sign_tool_file(
    *,
    tool_file_id: str,
    extension: str,
    for_external: bool = True,
) -> str:
    """Return a signed tool-file URL shim for tests and legacy patches."""
    return helpers.get_signed_tool_file_url(
        tool_file_id=tool_file_id,
        extension=extension,
        for_external=for_external,
    )


class ImageConfig(BaseModel):
    """NOTE: This part of validation is deprecated, but still used in app
    features "Image Upload".
    """

    number_limits: int = 0
    transfer_methods: Sequence[FileTransferMethod] = Field(default_factory=list)
    detail: ImagePromptMessageContent.DETAIL | None = None


class FileUploadConfig(BaseModel):
    """File Upload Entity."""

    image_config: ImageConfig | None = None
    allowed_file_types: Sequence[FileType] = Field(default_factory=list)
    allowed_file_extensions: Sequence[str] = Field(default_factory=list)
    allowed_file_upload_methods: Sequence[FileTransferMethod] = Field(
        default_factory=list,
    )
    number_limits: int = 0


def _parse_reference(reference: str | None) -> tuple[str | None, str | None]:
    """Best-effort parser for record references and historical storage-key payloads."""
    if not reference:
        return None, None

    if not reference.startswith(_FILE_REFERENCE_PREFIX):
        return reference, None

    encoded_payload = reference.removeprefix(_FILE_REFERENCE_PREFIX)
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded_payload.encode()))
    except (ValueError, json.JSONDecodeError):
        return reference, None

    record_id = payload.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        return reference, None

    storage_key = payload.get("storage_key")
    if not isinstance(storage_key, str):
        storage_key = None

    return record_id, storage_key


def _pick_legacy_reference(data: Mapping[str, Any]) -> str | None:
    for field_name in _LEGACY_REFERENCE_FIELDS:
        value = data.get(field_name)
        if value is not None:
            return str(value)
    return None


def _normalize_file_input(data: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(data)

    file_id = normalized.pop("file_id", None)
    if "id" not in normalized and file_id is not None:
        normalized["id"] = file_id

    file_type = normalized.pop("file_type", None)
    if "type" not in normalized and file_type is not None:
        normalized["type"] = file_type

    normalized.pop("tenant_id", None)
    normalized.pop("url", None)

    reference = normalized.get("reference")
    if reference is None:
        legacy_reference = _pick_legacy_reference(normalized)
        if legacy_reference is not None:
            normalized["reference"] = legacy_reference

    for field_name in _LEGACY_REFERENCE_FIELDS:
        normalized.pop(field_name, None)

    if normalized.get("dify_model_identity") is None:
        normalized["dify_model_identity"] = FILE_MODEL_IDENTITY

    storage_key_hint = normalized.pop("storage_key", None)
    _, parsed_storage_key = _parse_reference(normalized.get("reference"))
    normalized["storage_key_hint"] = storage_key_hint or parsed_storage_key
    return normalized


class File(BaseModel):
    """Graph-owned file reference.

    The graph layer deliberately keeps only the metadata required to route,
    serialize, and render files. Application ownership concerns such as
    tenant/user/conversation identity stay in the workflow/storage layer.
    """

    # NOTE: dify_model_identity is a special identifier used to distinguish between
    # new and old data formats during serialization and deserialization.
    dify_model_identity: str = FILE_MODEL_IDENTITY

    id: str | None = None  # message file id
    type: FileType
    transfer_method: FileTransferMethod
    # If `transfer_method` is `FileTransferMethod.remote_url`, the
    # `remote_url` attribute must not be `None`.
    remote_url: str | None = None  # remote url
    # Opaque workflow-layer reference for files resolved outside ``graphon``.
    # New payloads only carry the backing record id; historical payloads may
    # still include storage_key and must remain readable.
    reference: str | None = None
    filename: str | None = None
    extension: str | None = Field(
        default=None,
        description="File extension, should contain dot",
    )
    mime_type: str | None = None
    size: int = -1
    storage_key_hint: str | None = Field(default=None, exclude=True, repr=False)
    _storage_key: str = PrivateAttr(default="")

    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)

    @classmethod
    def from_reference(
        cls,
        *,
        file_type: FileType,
        transfer_method: FileTransferMethod,
        reference: str | None = None,
        record_id: str | None = None,
        file_id: str | None = None,
        filename: str | None = None,
        extension: str | None = None,
        mime_type: str | None = None,
        size: int = -1,
        storage_key: str | None = None,
        dify_model_identity: str = FILE_MODEL_IDENTITY,
    ) -> Self:
        if transfer_method not in _REFERENCE_TRANSFER_METHODS:
            msg = (
                "File.from_reference() only supports storage-backed transfer "
                "methods; use File.from_remote_url() for remote files"
            )
            raise ValueError(msg)

        normalized_reference = cls._resolve_reference_input(
            reference=reference,
            record_id=record_id,
        )
        return cls.model_validate(
            {
                "id": file_id,
                "type": file_type,
                "transfer_method": transfer_method,
                "reference": normalized_reference,
                "filename": filename,
                "extension": extension,
                "mime_type": mime_type,
                "size": size,
                "storage_key": storage_key,
                "dify_model_identity": dify_model_identity,
            },
        )

    @classmethod
    def from_remote_url(
        cls,
        *,
        file_type: FileType,
        url: str,
        file_id: str | None = None,
        filename: str | None = None,
        extension: str | None = None,
        mime_type: str | None = None,
        size: int = -1,
        dify_model_identity: str = FILE_MODEL_IDENTITY,
    ) -> Self:
        return cls.model_validate(
            {
                "id": file_id,
                "type": file_type,
                "transfer_method": FileTransferMethod.REMOTE_URL,
                "remote_url": url,
                "filename": filename,
                "extension": extension,
                "mime_type": mime_type,
                "size": size,
                "dify_model_identity": dify_model_identity,
            },
        )

    @classmethod
    def from_legacy(
        cls,
        *,
        file_type: FileType,
        transfer_method: FileTransferMethod,
        file_id: str | None = None,
        tenant_id: str | None = None,
        remote_url: str | None = None,
        reference: str | None = None,
        related_id: str | None = None,
        filename: str | None = None,
        extension: str | None = None,
        mime_type: str | None = None,
        size: int = -1,
        storage_key: str | None = None,
        dify_model_identity: str | None = FILE_MODEL_IDENTITY,
        url: str | None = None,
        tool_file_id: str | None = None,
        upload_file_id: str | None = None,
        datasource_file_id: str | None = None,
    ) -> Self:
        return cls.model_validate(
            {
                "file_id": file_id,
                "tenant_id": tenant_id,
                "file_type": file_type,
                "transfer_method": transfer_method,
                "remote_url": remote_url,
                "reference": reference,
                "related_id": related_id,
                "filename": filename,
                "extension": extension,
                "mime_type": mime_type,
                "size": size,
                "storage_key": storage_key,
                "dify_model_identity": dify_model_identity,
                "url": url,
                "tool_file_id": tool_file_id,
                "upload_file_id": upload_file_id,
                "datasource_file_id": datasource_file_id,
            },
        )

    @staticmethod
    def _resolve_reference_input(
        *,
        reference: str | None,
        record_id: str | None,
    ) -> str | None:
        if reference is None:
            return record_id
        if record_id is None:
            return reference

        parsed_record_id, _ = _parse_reference(reference)
        if parsed_record_id != record_id:
            msg = "reference and record_id describe different files"
            raise ValueError(msg)
        return reference

    @model_validator(mode="before")
    @classmethod
    def normalize_constructor_input(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return _normalize_file_input(value)

    def model_post_init(self, __context: Any, /) -> None:
        self._storage_key = self.storage_key_hint or ""

    def to_dict(self) -> Mapping[str, str | int | None]:
        data = self.model_dump(mode="json")
        return {
            **data,
            "related_id": self.related_id,
            "url": self.generate_url(),
        }

    @property
    def markdown(self) -> str:
        url = self.generate_url()
        if self.type == FileType.IMAGE:
            text = f"![{self.filename or ''}]({url})"
        else:
            text = f"[{self.filename or url}]({url})"

        return text

    def generate_url(self, for_external: bool = True) -> str | None:
        return helpers.resolve_file_url(self, for_external=for_external)

    def to_plugin_parameter(self) -> dict[str, Any]:
        return {
            "dify_model_identity": FILE_MODEL_IDENTITY,
            "mime_type": self.mime_type,
            "filename": self.filename,
            "extension": self.extension,
            "size": self.size,
            "type": self.type,
            "url": self.generate_url(for_external=False),
        }

    @model_validator(mode="after")
    def validate_after(self) -> Self:
        match self.transfer_method:
            case FileTransferMethod.REMOTE_URL:
                if not self.remote_url:
                    msg = "Missing file url"
                    raise ValueError(msg)
                if not isinstance(
                    self.remote_url,
                    str,
                ) or not self.remote_url.startswith("http"):
                    msg = "Invalid file url"
                    raise ValueError(msg)
            case FileTransferMethod.LOCAL_FILE:
                if not self.reference:
                    msg = "Missing file reference"
                    raise ValueError(msg)
            case FileTransferMethod.TOOL_FILE:
                if not self.reference:
                    msg = "Missing file reference"
                    raise ValueError(msg)
            case FileTransferMethod.DATASOURCE_FILE:
                if not self.reference:
                    msg = "Missing file reference"
                    raise ValueError(msg)
        return self

    @property
    def related_id(self) -> str | None:
        record_id, _ = _parse_reference(self.reference)
        return record_id

    @related_id.setter
    def related_id(self, value: str | None) -> None:
        self.reference = value

    @property
    def storage_key(self) -> str:
        _, storage_key = _parse_reference(self.reference)
        return storage_key or self._storage_key

    @storage_key.setter
    def storage_key(self, value: str) -> None:
        self._storage_key = value
