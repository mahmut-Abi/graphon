import pytest

from graphon.file import helpers
from graphon.file.enums import (
    FileTransferMethod,
    FileType,
)
from graphon.file.models import File

from ..helpers import build_file_reference


def _build_local_file(*, reference: str, storage_key: str | None = None) -> File:
    return File.from_reference(
        file_type=FileType.DOCUMENT,
        transfer_method=FileTransferMethod.LOCAL_FILE,
        reference=reference,
        file_id="file-id",
        filename="report.pdf",
        extension=".pdf",
        mime_type="application/pdf",
        size=128,
        storage_key=storage_key,
    )


def test_file_exposes_legacy_aliases_from_opaque_reference() -> None:
    reference = build_file_reference(
        record_id="upload-file-id",
        storage_key="files/report.pdf",
    )

    file = _build_local_file(reference=reference)

    assert file.reference == reference
    assert file.related_id == "upload-file-id"
    assert file.storage_key == "files/report.pdf"


def test_file_falls_back_to_raw_reference_when_opaque_reference_is_invalid() -> None:
    file = _build_local_file(
        reference="dify-file-ref:not-base64",
        storage_key="fallback-key",
    )

    assert file.related_id == "dify-file-ref:not-base64"
    assert file.storage_key == "fallback-key"


def test_file_from_reference_supports_record_id_storage_key() -> None:
    file = File.from_reference(
        file_type=FileType.DOCUMENT,
        transfer_method=FileTransferMethod.LOCAL_FILE,
        record_id="upload-file-id",
        file_id="file-id",
        filename="report.pdf",
        extension=".pdf",
        mime_type="application/pdf",
        size=128,
        storage_key="files/report.pdf",
    )

    assert file.reference == "upload-file-id"
    assert file.related_id == "upload-file-id"
    assert file.storage_key == "files/report.pdf"


def test_file_to_dict_keeps_reference_and_legacy_related_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = build_file_reference(
        record_id="upload-file-id",
        storage_key="files/report.pdf",
    )
    file = _build_local_file(reference=reference)

    def fake_resolve_file_url(_file: File, *, for_external: bool = True) -> str:
        _ = for_external
        return "https://example.com/report.pdf"

    monkeypatch.setattr(
        helpers,
        "resolve_file_url",
        fake_resolve_file_url,
    )

    serialized = file.to_dict()

    assert serialized["reference"] == reference
    assert serialized["related_id"] == "upload-file-id"
    assert serialized["url"] == "https://example.com/report.pdf"


def test_file_related_id_setter_updates_reference_alias() -> None:
    file = _build_local_file(reference="upload-file-id", storage_key="files/report.pdf")

    file.related_id = "replacement-upload-id"

    assert file.reference == "replacement-upload-id"
    assert file.related_id == "replacement-upload-id"


def test_file_from_remote_url_builds_remote_file_shape() -> None:
    file = File.from_remote_url(
        file_type=FileType.IMAGE,
        url="https://example.com/image.png",
        file_id="file-id",
        filename="image.png",
        extension=".png",
        mime_type="image/png",
        size=64,
    )

    assert file.type == FileType.IMAGE
    assert file.transfer_method == FileTransferMethod.REMOTE_URL
    assert file.remote_url == "https://example.com/image.png"
    assert file.reference is None


def test_file_from_reference_rejects_remote_transfer_method() -> None:
    with pytest.raises(ValueError, match="from_remote_url"):
        File.from_reference(
            file_type=FileType.IMAGE,
            transfer_method=FileTransferMethod.REMOTE_URL,
            reference="https://example.com/image.png",
        )


def test_file_from_legacy_keeps_existing_aliases() -> None:
    file = File.from_legacy(
        file_id="file-id",
        file_type=FileType.DOCUMENT,
        transfer_method=FileTransferMethod.TOOL_FILE,
        tool_file_id="tool-file-id",
        filename="report.pdf",
        extension=".pdf",
        mime_type="application/pdf",
        size=128,
        storage_key="tool/report.pdf",
    )

    assert file.id == "file-id"
    assert file.reference == "tool-file-id"
    assert file.related_id == "tool-file-id"
    assert file.storage_key == "tool/report.pdf"


def test_file_direct_constructor_keeps_legacy_keyword_compatibility() -> None:
    file = File(
        file_id="file-id",
        file_type=FileType.DOCUMENT,
        transfer_method=FileTransferMethod.TOOL_FILE,
        tool_file_id="tool-file-id",
        filename="report.pdf",
        extension=".pdf",
        mime_type="application/pdf",
        size=128,
        storage_key="tool/report.pdf",
    )

    assert file.id == "file-id"
    assert file.reference == "tool-file-id"
    assert file.related_id == "tool-file-id"
    assert file.storage_key == "tool/report.pdf"


def test_file_model_validate_normalizes_legacy_mapping() -> None:
    file = File.model_validate(
        {
            "file_id": "file-id",
            "file_type": FileType.DOCUMENT,
            "transfer_method": FileTransferMethod.LOCAL_FILE,
            "upload_file_id": "upload-file-id",
            "filename": "report.pdf",
            "extension": ".pdf",
            "mime_type": "application/pdf",
            "size": 128,
            "storage_key": "files/report.pdf",
        },
    )

    assert file.id == "file-id"
    assert file.reference == "upload-file-id"
    assert file.related_id == "upload-file-id"
    assert file.storage_key == "files/report.pdf"
