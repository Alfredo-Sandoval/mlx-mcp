"""Validated, atomic persistence of passage records."""

import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO, BinaryIO, TypedDict, cast
from zipfile import BadZipFile

import numpy as np
from numpy.lib.npyio import NpzFile

from mlx_mcp.configuration import refresh_command
from mlx_mcp.packages import EvidenceKind, Package
from mlx_mcp.passages import Passage, validate_passages

# Bump whenever parsing or windowing change; older indexes are then rejected
# and rebuilt rather than migrated.
INDEX_VERSION = 1

type FloatScalar = np.float16 | np.float32 | np.float64 | np.longdouble


class IndexFileInfo(TypedDict):
    """Size and modification time captured from the open, loaded index file."""

    size_bytes: int
    modified_at_ns: int


@dataclass(frozen=True)
class LoadedIndex:
    """A loaded index: its passages and where they came from."""

    passages: list[Passage]
    # Package version that built the index; local indexes must match the release.
    built_by: str
    file_info: IndexFileInfo


@contextmanager
def atomic_output(path: Path) -> Generator[IO[bytes]]:
    """Yield a temporary file that replaces path only after a successful write.

    The file is synced before the rename and removed if the write fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_index(root: Path, passages: list[Passage]) -> None:
    """Validate passages, then atomically write root/index.npz."""
    from mlx_mcp import __version__

    validate_passages(passages)
    stored = json.dumps([_encode_passage(p) for p in passages])
    # Publish only records that load_index will decode.
    records: object = json.loads(stored)
    if not isinstance(records, list):
        raise TypeError("Serialized passages must be a list")
    for raw in cast(list[object], records):
        _decode_passage(raw)
    with atomic_output(root / "index.npz") as stream:
        np.savez_compressed(
            stream,
            version=INDEX_VERSION,
            built_by=__version__,
            passages=stored,
        )


def _open_archive(source: Path | BinaryIO) -> NpzFile:
    """Open an index archive without allowing pickled objects."""
    return np.load(source, allow_pickle=False)


def _scalar(archive: NpzFile, key: str, kind: str) -> object:
    """Return a 0-d archive field whose dtype kind is one of kind."""
    value = archive[key]
    if value.shape != () or value.dtype.kind not in kind:
        raise ValueError(f"Invalid index field: {key}")
    return value.item()


# Preserve the stored "section_text" key while using segment_text in memory.
_PASSAGE_FIELDS = frozenset(
    "section_text" if field.name == "segment_text" else field.name
    for field in fields(Passage)
)


def _encode_passage(passage: Passage) -> dict[str, object]:
    record = asdict(passage)
    record["section_text"] = record.pop("segment_text")
    return record


def _record_text(record: dict[object, object], name: str, default: str | None) -> str:
    value = record.get(name, default)
    if not isinstance(value, str):
        raise ValueError(f"Passage {name} must be a string")  # noqa: TRY004 - invalid stored data
    return value


def _record_integer(record: dict[object, object], name: str) -> int:
    value = record.get(name, -1)
    # bool is an int subclass; JSON true/false are not source offsets.
    if type(value) is not int:
        raise ValueError(f"Passage {name} must be an integer")
    return value


def _record_strings(record: dict[object, object], name: str) -> tuple[str, ...]:
    values = record.get(name, [])
    if not isinstance(values, list):
        raise ValueError(f"Passage {name} must be a JSON array")  # noqa: TRY004 - invalid stored data
    items = cast(list[object], values)
    strings = tuple(item for item in items if isinstance(item, str))
    if len(strings) != len(items):
        raise ValueError(f"Passage {name} must contain only strings")
    return strings


def _decode_passage(raw: object) -> Passage:
    """Build a Passage from one stored JSON record, checking every field's type."""
    if not isinstance(raw, dict):
        raise ValueError("Passage records must be JSON objects")  # noqa: TRY004 - invalid stored data
    record = cast(dict[object, object], raw)
    unknown = set(record) - _PASSAGE_FIELDS
    if unknown:
        raise ValueError(f"Unknown passage fields: {sorted(map(str, unknown))}")
    return Passage(
        title=_record_text(record, "title", None),
        url=_record_text(record, "url", None),
        text=_record_text(record, "text", None),
        segment_text=_record_text(record, "section_text", ""),
        anchors=_record_strings(record, "anchors"),
        documentation_version=_record_text(record, "documentation_version", ""),
        retrieved_at=_record_text(record, "retrieved_at", ""),
        section_path=_record_strings(record, "section_path"),
        section_urls=_record_strings(record, "section_urls"),
        segment_index=_record_integer(record, "segment_index"),
        chunk_start=_record_integer(record, "chunk_start"),
        chunk_end=_record_integer(record, "chunk_end"),
        package=Package(_record_text(record, "package", None)),
        evidence_kind=EvidenceKind(_record_text(record, "evidence_kind", None)),
        source_revision=_record_text(record, "source_revision", ""),
    )


def read_index_identity(root: Path) -> tuple[int | None, str | None]:
    """Read the format and builder release without decoding stored passages.

    Missing fields identify indexes that predate the corresponding metadata.
    """
    try:
        with _open_archive(root / "index.npz") as archive:
            version = (
                cast(int, _scalar(archive, "version", "iu"))
                if "version" in archive
                else None
            )
            built_by = (
                str(_scalar(archive, "built_by", "US"))
                if "built_by" in archive
                else None
            )
            return version, built_by
    # np.load reports a zero-byte or cut-off file as EOFError.
    except (ValueError, OSError, BadZipFile, EOFError) as exc:
        raise ValueError(
            f"Invalid MLX index at {root}: {exc}. Rebuild it with `{refresh_command(root)}`."
        ) from exc


def load_index(root: Path) -> LoadedIndex:
    """Load and validate root/index.npz, rejecting other formats."""
    # Read one complete file even if a refresh replaces the path meanwhile.
    with (root / "index.npz").open("rb") as stream:
        before = os.fstat(stream.fileno())
        data = stream.read()
        after = os.fstat(stream.fileno())
    file_info: IndexFileInfo = {
        "size_bytes": before.st_size,
        "modified_at_ns": before.st_mtime_ns,
    }
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ) or len(data) != before.st_size:
        raise ValueError("Index changed while loading; retry on a stable index file")
    try:
        with _open_archive(BytesIO(data)) as archive:
            if "version" not in archive or _scalar(archive, "version", "iu") != (
                INDEX_VERSION
            ):
                raise ValueError("Index format changed")
            built_by = str(_scalar(archive, "built_by", "US"))
            stored = _scalar(archive, "passages", "U")
            if not isinstance(stored, str):
                raise TypeError("Passages must be stored as JSON text")
            records: object = json.loads(stored)
            if not isinstance(records, list):
                raise TypeError("Passages must be a list")
            passages = [_decode_passage(raw) for raw in cast(list[object], records)]
        validate_passages(passages)
    except (
        ValueError,
        TypeError,
        KeyError,
        EOFError,
        OverflowError,
        BadZipFile,
        OSError,
    ) as exc:
        raise ValueError(
            f"Invalid MLX index: {exc}. Rebuild it with `{refresh_command(root)}`."
        ) from exc
    return LoadedIndex(passages, built_by, file_info)
