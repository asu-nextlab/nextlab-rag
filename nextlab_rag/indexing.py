from __future__ import annotations

import fcntl
import hashlib
import json
import os
import resource
import signal
import shutil
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from qdrant_client import QdrantClient, models

from .errors import IndexNotReadyError, RagError

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "nextlab_rag"
DENSE_VECTOR_NAME = "dense"
TEXT_FIELD = "passage_text"
SCHEMA_VERSION = 2
CHECKPOINT_TIMEOUT_SECONDS = 300.0
MAX_PASSAGE_TOKENS = 384
MAX_PASSAGE_CHARS = 1200
PASSAGE_OVERLAP_CHARS = 120
EMBED_BATCH_SIZE = 32
_REQUIRED_FILES = (
    "chunks.jsonl",
    "documents.jsonl",
    "provenance.jsonl",
    "review/correction-ledger.jsonl",
)


@dataclass(frozen=True)
class Passage:
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class Corpus:
    chunks: list[dict[str, Any]]
    chunks_by_id: dict[str, dict[str, Any]]
    documents: dict[str, dict[str, Any]]
    provenance: dict[str, dict[str, Any]]
    corrections: dict[str, dict[str, Any]]
    uncertainties: list[dict[str, Any]]
    uncertainties_by_doc: dict[str, list[dict[str, Any]]]
    file_hashes: dict[str, str]


class CheckpointTimeoutError(RagError):
    """Raised when indexing does not advance its durable checkpoint in time."""


def split_passages(
    text: str,
    *,
    token_count: Callable[[str], int],
    max_tokens: int = MAX_PASSAGE_TOKENS,
    max_chars: int = MAX_PASSAGE_CHARS,
    overlap_chars: int = PASSAGE_OVERLAP_CHARS,
) -> list[Passage]:
    """Split text into overlapping passages and verify each with the model tokenizer."""
    return list(_iter_passages(
        text, token_count=token_count, max_tokens=max_tokens,
        max_chars=max_chars, overlap_chars=overlap_chars,
    ))


def _iter_passages(
    text: str,
    *,
    token_count: Callable[[str], int],
    max_tokens: int = MAX_PASSAGE_TOKENS,
    max_chars: int = MAX_PASSAGE_CHARS,
    overlap_chars: int = PASSAGE_OVERLAP_CHARS,
) -> Iterator[Passage]:
    if max_tokens < 1 or max_chars < 1 or overlap_chars < 0:
        raise ValueError("passage limits must be positive, with nonnegative overlap")
    if not text:
        return

    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind(" ", start + 1, end), text.rfind("\n", start + 1, end))
            if boundary > start:
                end = boundary + 1

        while token_count(text[start:end]) > max_tokens:
            if end - start <= 1:
                raise RagError("a source span cannot fit within the BGE token limit")
            candidate = max(start + 1, start + (end - start) * 3 // 4)
            boundary = max(text.rfind(" ", start + 1, candidate), text.rfind("\n", start + 1, candidate))
            end = boundary + 1 if boundary > start else min(candidate, end - 1)
        if end <= start or token_count(text[start:end]) > max_tokens:
            raise RagError("a source span cannot fit within the BGE token limit")

        yield Passage(text[start:end], start, end)
        if end == len(text):
            break
        next_start = max(start + 1, end - overlap_chars)
        while next_start > start and next_start < len(text) and not text[next_start - 1].isspace() and not text[next_start].isspace():
            next_start -= 1
        start = next_start if next_start > start else end


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RagError(f"expected an object in {path} line {line_number}")
            rows.append(value)
    return rows


def _unique_by(rows: Iterable[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise RagError(f"{label} row is missing {key}")
        if value in result:
            raise RagError(f"duplicate {key} in {label}: {value}")
        result[value] = row
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_corpus(dataset_dir: Path) -> Corpus:
    paths: dict[str, Path] = {name: dataset_dir / name for name in _REQUIRED_FILES}
    optional_uncertainties = dataset_dir / "review/uncertainties.jsonl"
    if optional_uncertainties.is_file():
        paths["review/uncertainties.jsonl"] = optional_uncertainties
    for name, path in paths.items():
        if not path.is_file():
            raise RagError(f"required reviewed export file is missing: {name}")

    chunks = _jsonl(paths["chunks.jsonl"])
    documents = _unique_by(_jsonl(paths["documents.jsonl"]), "doc_id", "documents")
    provenance = _unique_by(_jsonl(paths["provenance.jsonl"]), "doc_id", "provenance")
    corrections = _unique_by(_jsonl(paths["review/correction-ledger.jsonl"]), "correction_id", "correction ledger")
    uncertainties = _jsonl(optional_uncertainties) if optional_uncertainties.is_file() else []
    chunks_by_id: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        doc_id = chunk.get("doc_id")
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id in chunks_by_id:
            raise RagError(f"chunk IDs must be present and unique: {chunk_id!r}")
        if doc_id not in documents or doc_id not in provenance:
            raise RagError(f"chunk {chunk_id} has no matching document and provenance row")
        if not isinstance(chunk.get("text"), str):
            raise RagError(f"chunk {chunk_id} has no text")
        chunks_by_id[chunk_id] = chunk
    if set(documents) != set(provenance):
        raise RagError("document and provenance doc_id sets do not match")
    uncertainties_by_doc: dict[str, list[dict[str, Any]]] = {}
    for row in uncertainties:
        doc_id = row.get("doc_id")
        if isinstance(doc_id, str):
            uncertainties_by_doc.setdefault(doc_id, []).append(row)
    return Corpus(
        chunks=chunks,
        chunks_by_id=chunks_by_id,
        documents=documents,
        provenance=provenance,
        corrections=corrections,
        uncertainties=uncertainties,
        uncertainties_by_doc=uncertainties_by_doc,
        file_hashes={name: _sha256(path) for name, path in sorted(paths.items())},
    )


def _corpus_fingerprint(file_hashes: dict[str, str]) -> str:
    encoded = json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _model_metadata(model_name: str) -> tuple[str, str]:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RagError("FastEmbed is missing; install requirements-rag.txt") from exc
    entries = TextEmbedding.list_supported_models()
    entry = next((row for row in entries if row.get("model", "").casefold() == model_name.casefold()), None)
    if entry is None:
        raise RagError(f"FastEmbed does not list the embedding model {model_name!r}")
    descriptor = json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str).encode()
    return version("fastembed"), hashlib.sha256(descriptor).hexdigest()


def _model_artifact_identity(model_name: str, embedder: Any) -> dict[str, Any]:
    """Hash the actual loaded model and tokenizer files used by FastEmbed."""
    model = getattr(embedder, "model", None)
    model_dir_value = getattr(model, "_model_dir", None)
    if model_dir_value is not None:
        model_dir = Path(model_dir_value).resolve()
        files = {
            path.relative_to(model_dir).as_posix(): _sha256(path)
            for path in sorted(model_dir.rglob("*"))
            if path.is_file()
        }
        if not files:
            raise RagError(f"the loaded embedding model has no files to identify: {model_name}")
        revision = model_dir.name if model_dir.parent.name == "snapshots" else None
        return {
            "model": model_name,
            "source": "loaded-model-files",
            "revision": revision,
            "files": files,
        }

    injected = getattr(embedder, "artifact_identity", None)
    if isinstance(injected, dict):
        try:
            identity = json.loads(json.dumps(injected, sort_keys=True))
        except (TypeError, ValueError) as exc:
            raise RagError("injected embedding artifact identity must be JSON serializable") from exc
        if identity.get("revision") and identity.get("files"):
            return {"model": model_name, **identity}
    raise RagError("the loaded embedding model does not expose an immutable artifact identity")


def _signature(
    file_hashes: dict[str, str],
    model_name: str,
    qdrant_version: str,
    fastembed_version: str,
    model_registry_hash: str,
    model_artifact_identity: dict[str, Any],
) -> str:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "corpus_files": file_hashes,
        "embedding_model": model_name,
        "model_registry_sha256": model_registry_hash,
        "model_artifact_identity": model_artifact_identity,
        "qdrant_client_version": qdrant_version,
        "fastembed_version": fastembed_version,
        "passage_limits": {
            "max_tokens": MAX_PASSAGE_TOKENS,
            "max_chars": MAX_PASSAGE_CHARS,
            "overlap_chars": PASSAGE_OVERLAP_CHARS,
        },
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise RagError(f"expected a JSON object in {path}")
    return value


def _current_rss_bytes() -> int | None:
    try:
        with Path("/proc/self/statm").open(encoding="ascii") as stream:
            resident_pages = int(stream.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _record_progress(
    path: Path,
    *,
    started: float,
    completed: int,
    total: int | None,
    model_stage: str,
) -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "pid": os.getpid(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "completed_passages": completed,
        "total_passages": total,
        "model_stage": model_stage,
        "rss_bytes": _current_rss_bytes(),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, separators=(",", ":")) + "\n")


@contextmanager
def _checkpoint_watchdog(timeout_seconds: float) -> Iterator[Callable[[], None]]:
    if timeout_seconds <= 0:
        raise ValueError("checkpoint timeout must be positive")
    if threading.current_thread() is not threading.main_thread():
        raise RagError("the checkpoint watchdog requires the main thread")
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    if old_timer[0] or old_timer[1] or old_handler not in (signal.SIG_DFL, signal.SIG_IGN):
        raise RagError("checkpoint watchdog cannot replace an existing SIGALRM handler or timer")

    def timeout_handler(signum: int, frame: Any) -> None:
        del signum, frame
        raise CheckpointTimeoutError(
            f"indexing made no durable checkpoint progress for {timeout_seconds:g} seconds; "
            "the manifest checkpoint was left unchanged"
        )

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)

    def reset_after_checkpoint() -> None:
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)

    try:
        yield reset_after_checkpoint
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _current(rag_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    pointer = _read_json(rag_dir / "current.json")
    if not pointer:
        return None
    generation = pointer.get("generation")
    if not isinstance(generation, str) or Path(generation).name != generation:
        raise IndexNotReadyError("the active index pointer is invalid")
    generation_dir = rag_dir / "indexes" / generation
    manifest = _read_json(generation_dir / "manifest.json")
    if manifest is None:
        raise IndexNotReadyError("the active index manifest is missing")
    return generation_dir, manifest


@contextmanager
def _index_lock(rag_dir: Path) -> Iterator[None]:
    rag_dir.mkdir(parents=True, exist_ok=True)
    lock_path = rag_dir / "index.lock"
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RagError("another RAG command is already using local Qdrant storage") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({"pid": os.getpid()}))
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _load_embedder(model_name: str, cache_dir: Path) -> Any:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("ORT_NUM_THREADS", "1")
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RagError("FastEmbed is missing; install requirements-rag.txt") from exc
    return TextEmbedding(
        model_name=model_name,
        cache_dir=str(cache_dir),
        threads=1,
        providers=["CPUExecutionProvider"],
    )


def _point_id(chunk_id: str, start: int, end: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"nextlab-rag:{chunk_id}:{start}:{end}"))


def _payloads(corpus: Corpus, token_count: Callable[[str], int]) -> Iterator[tuple[str, dict[str, Any]]]:
    for chunk in corpus.chunks:
        chunk_id = chunk["chunk_id"]
        doc_id = chunk["doc_id"]
        document = corpus.documents[doc_id]
        provenance = corpus.provenance[doc_id]
        correction_ids: list[str] = []
        for row in (chunk, document, provenance):
            value = row.get("review_correction_ids", [])
            if isinstance(value, list):
                correction_ids.extend(item for item in value if isinstance(item, str))
        correction_ids = list(dict.fromkeys(correction_ids))
        missing = [item for item in correction_ids if item not in corpus.corrections]
        if missing:
            raise RagError(f"chunk {chunk_id} references missing corrections: {', '.join(missing)}")
        text = chunk["text"]
        for passage in _iter_passages(text, token_count=token_count):
            yield (
                _point_id(chunk_id, passage.start_char, passage.end_char),
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    TEXT_FIELD: passage.text,
                    "passage_start_char": passage.start_char,
                    "passage_end_char": passage.end_char,
                    "review_correction_ids": correction_ids,
                },
            )


def _count_passages(corpus: Corpus, token_count: Callable[[str], int]) -> int:
    return sum(1 for _ in _payloads(corpus, token_count))


def _checkpoint_points_match(
    client: QdrantClient, corpus: Corpus, token_count: Callable[[str], int], completed: int
) -> bool:
    stored_count = _collection_count(client)
    if stored_count < completed or stored_count > completed + 128:
        return False
    rows = iter(_payloads(corpus, token_count))
    remaining = completed
    while remaining:
        expected_count = min(128, remaining)
        point_ids = [point_id for point_id, _ in islice(rows, expected_count)]
        if len(point_ids) != expected_count:
            return False
        found = client.retrieve(
            collection_name=COLLECTION_NAME, ids=point_ids, with_payload=False, with_vectors=False,
        )
        if len(found) != expected_count:
            return False
        remaining -= expected_count
    if stored_count == completed:
        return True
    # An interrupted upsert can persist part of a batch before its checkpoint.
    point_ids = [point_id for point_id, _ in islice(rows, 128)]
    found = client.retrieve(
        collection_name=COLLECTION_NAME, ids=point_ids, with_payload=False, with_vectors=False,
    ) if point_ids else []
    return len(found) == stored_count - completed


def _manifest_matches(
    manifest: dict[str, Any], *, signature: str, model_name: str, file_hashes: dict[str, str]
) -> bool:
    return (
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("status") == "complete"
        and manifest.get("signature") == signature
        and manifest.get("embedding_model") == model_name
        and manifest.get("corpus_files") == file_hashes
    )


def _collection_count(client: QdrantClient) -> int:
    return int(client.count(collection_name=COLLECTION_NAME, exact=True).count)


def _ensure_text_index(client: QdrantClient) -> None:
    info = client.get_collection(COLLECTION_NAME)
    schema = info.payload_schema or {}
    if TEXT_FIELD in schema:
        return
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=TEXT_FIELD,
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
            lowercase=True,
            scoring=models.TextScoringParams(type=models.TextScoringType.BM25),
        ),
        wait=True,
    )


def _sd_notify(message: str) -> None:
    notify_socket = os.environ.get("NOTIFY_SOCKET")
    if not notify_socket:
        return
    address = "\0" + notify_socket[1:] if notify_socket.startswith("@") else notify_socket
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as connection:
            connection.connect(address)
            connection.sendall(message.encode("utf-8"))
    except OSError as exc:
        raise RagError(f"could not notify the service manager through NOTIFY_SOCKET: {exc}") from exc


def index_corpus(
    repo_root: Path | str,
    *,
    dataset_dir: Path | str | None = None,
    rag_dir: Path | str | None = None,
    model_name: str = EMBEDDING_MODEL,
    embedder: Any | None = None,
    batch_size: int = EMBED_BATCH_SIZE,
    checkpoint_timeout_seconds: float = CHECKPOINT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Build or resume a staged local index and atomically publish its manifest."""
    if batch_size < 1 or batch_size > 128:
        raise ValueError("batch_size must be between 1 and 128")
    started = time.monotonic()
    with _checkpoint_watchdog(checkpoint_timeout_seconds) as reset_checkpoint_timer:
        _sd_notify("READY=1\nWATCHDOG=1")
        return _index_corpus(
            repo_root,
            dataset_dir=dataset_dir,
            rag_dir=rag_dir,
            model_name=model_name,
            embedder=embedder,
            batch_size=batch_size,
            started=started,
            reset_checkpoint_timer=reset_checkpoint_timer,
        )


def _index_corpus(
    repo_root: Path | str,
    *,
    dataset_dir: Path | str | None,
    rag_dir: Path | str | None,
    model_name: str,
    embedder: Any | None,
    batch_size: int,
    started: float,
    reset_checkpoint_timer: Callable[[], None],
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    dataset = Path(dataset_dir).resolve() if dataset_dir else root / "dataset"
    storage = Path(rag_dir).resolve() if rag_dir else root / ".rag"
    storage.mkdir(parents=True, exist_ok=True)
    progress_path = storage / "progress.jsonl"
    _record_progress(progress_path, started=started, completed=0, total=None, model_stage="startup")
    corpus = _load_corpus(dataset)
    _record_progress(progress_path, started=started, completed=0, total=None, model_stage="corpus_loaded")
    qdrant_version = version("qdrant-client")
    fastembed_version, registry_hash = _model_metadata(model_name)
    fingerprint = _corpus_fingerprint(corpus.file_hashes)

    with _index_lock(storage):
        if embedder is None:
            _record_progress(
                progress_path,
                started=started,
                completed=0,
                total=None,
                model_stage="model_loading",
            )
            embedder = _load_embedder(model_name, storage / "cache")
        model_artifact_identity = _model_artifact_identity(model_name, embedder)
        _record_progress(
            progress_path,
            started=started,
            completed=0,
            total=None,
            model_stage="model_ready",
        )
        signature = _signature(
            corpus.file_hashes,
            model_name,
            qdrant_version,
            fastembed_version,
            registry_hash,
            model_artifact_identity,
        )
        active = _current(storage)
        if active:
            active_dir, active_manifest = active
            if _manifest_matches(active_manifest, signature=signature, model_name=model_name, file_hashes=corpus.file_hashes):
                client = QdrantClient(path=str(active_dir / "qdrant"))
                try:
                    if _collection_count(client) == active_manifest.get("passage_count"):
                        _record_progress(
                            progress_path,
                            started=started,
                            completed=int(active_manifest["passage_count"]),
                            total=int(active_manifest["passage_count"]),
                            model_stage="already_complete",
                        )
                        return active_manifest
                finally:
                    client.close()

        stage = storage / "staging"
        stage_manifest_path = stage / "manifest.json"
        stage_manifest = _read_json(stage_manifest_path) if stage_manifest_path.exists() else None
        if (
            not stage_manifest
            or stage_manifest.get("schema_version") != SCHEMA_VERSION
            or stage_manifest.get("signature") != signature
            or stage_manifest.get("model_artifact_identity") != model_artifact_identity
            or stage_manifest.get("status") not in {"building", "complete"}
        ):
            if stage.exists():
                shutil.rmtree(stage)
            stage.mkdir(parents=True)
            stage_manifest = {
                "schema_version": SCHEMA_VERSION,
                "status": "building",
                "signature": signature,
                "corpus_fingerprint": fingerprint,
                "corpus_files": corpus.file_hashes,
                "qdrant_client_version": qdrant_version,
                "fastembed_version": fastembed_version,
                "model_registry_sha256": registry_hash,
                "model_artifact_identity": model_artifact_identity,
                "embedding_model": model_name,
                "dense_dimension": int(getattr(embedder, "embedding_size", 384)),
                "document_count": len(corpus.documents),
                "chunk_count": len(corpus.chunks),
                "passage_count": 0,
                "completed_passages": 0,
                "passage_limits": {
                    "max_tokens": MAX_PASSAGE_TOKENS,
                    "max_chars": MAX_PASSAGE_CHARS,
                    "overlap_chars": PASSAGE_OVERLAP_CHARS,
                },
            }
            _atomic_json(stage_manifest_path, stage_manifest)

        token_count = lambda value: int(embedder.token_count(value))
        passage_count = _count_passages(corpus, token_count)
        _record_progress(
            progress_path,
            started=started,
            completed=int(stage_manifest.get("completed_passages", 0)),
            total=passage_count,
            model_stage="passage_preparation_complete",
        )
        dimension = int(getattr(embedder, "embedding_size", 384))
        if stage_manifest.get("passage_count") not in (0, passage_count):
            raise IndexNotReadyError("the staged index passage count does not match this corpus")
        stage_manifest["passage_count"] = passage_count
        stage_manifest["dense_dimension"] = dimension
        _atomic_json(stage_manifest_path, stage_manifest)

        qdrant_path = stage / "qdrant"
        client = QdrantClient(path=str(qdrant_path))
        try:
            if not client.collection_exists(COLLECTION_NAME):
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config={
                        DENSE_VECTOR_NAME: models.VectorParams(size=dimension, distance=models.Distance.COSINE)
                    },
                )
            _ensure_text_index(client)

            completed = int(stage_manifest.get("completed_passages", 0))
            if completed < 0 or completed > passage_count:
                raise IndexNotReadyError("the staged index checkpoint is outside the corpus")
            if not _checkpoint_points_match(client, corpus, token_count, completed):
                client.delete_collection(COLLECTION_NAME)
                stage_manifest["completed_passages"] = 0
                _atomic_json(stage_manifest_path, stage_manifest)
                completed = 0
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config={
                        DENSE_VECTOR_NAME: models.VectorParams(size=dimension, distance=models.Distance.COSINE)
                    },
                )
                _ensure_text_index(client)
                _record_progress(
                    progress_path,
                    started=started,
                    completed=0,
                    total=passage_count,
                    model_stage="collection_reset",
                )
            # Replay only validated deterministic points beyond the checkpoint.

            rows = iter(_payloads(corpus, token_count))
            for _ in range(completed):
                try:
                    next(rows)
                except StopIteration as exc:
                    raise IndexNotReadyError("the staged checkpoint exceeds deterministic passage output") from exc

            start = completed
            while batch := list(islice(rows, batch_size)):
                texts = [payload[TEXT_FIELD] for _, payload in batch]
                vectors = list(embedder.embed(texts, batch_size=batch_size, parallel=None))
                if len(vectors) != len(batch):
                    raise RagError("the embedding model returned a different number of vectors than passages")
                points = [
                    models.PointStruct(
                        id=point_id,
                        vector={DENSE_VECTOR_NAME: [float(value) for value in vector]},
                        payload=payload,
                    )
                    for (point_id, payload), vector in zip(batch, vectors, strict=True)
                ]
                client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
                start += len(batch)
                stage_manifest["completed_passages"] = start
                _atomic_json(stage_manifest_path, stage_manifest)
                reset_checkpoint_timer()
                _record_progress(
                    progress_path,
                    started=started,
                    completed=start,
                    total=passage_count,
                    model_stage="embedding",
                )
                _sd_notify("WATCHDOG=1")
            if start != passage_count:
                raise IndexNotReadyError("the deterministic passage count changed during indexing")
            if _collection_count(client) != passage_count:
                raise IndexNotReadyError("staged Qdrant point count does not match the completed manifest")
            stage_manifest["status"] = "complete"
            stage_manifest["completed_passages"] = passage_count
            _atomic_json(stage_manifest_path, stage_manifest)
        finally:
            client.close()

        generation = f"{signature[:16]}-{uuid.uuid4().hex[:12]}"
        indexes = storage / "indexes"
        indexes.mkdir(parents=True, exist_ok=True)
        generation_dir = indexes / generation
        os.replace(stage, generation_dir)
        _atomic_json(storage / "current.json", {"generation": generation})
        for old_generation in indexes.iterdir():
            if old_generation.is_dir() and old_generation.name != generation:
                shutil.rmtree(old_generation)
        _record_progress(
            progress_path,
            started=started,
            completed=passage_count,
            total=passage_count,
            model_stage="published",
        )
        return stage_manifest


def _validate_current_index(
    repo_root: Path,
    storage: Path,
    model_name: str,
    model_artifact_identity: dict[str, Any],
) -> tuple[Path, dict[str, Any], Corpus]:
    active = _current(storage)
    if active is None:
        raise IndexNotReadyError("no completed index exists; run python -m nextlab_rag index")
    generation_dir, manifest = active
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise IndexNotReadyError("active index uses an old payload schema; run index")
    dataset = repo_root / "dataset"
    corpus = _load_corpus(dataset)
    qdrant_version = version("qdrant-client")
    fastembed_version, registry_hash = _model_metadata(model_name)
    if manifest.get("embedding_model") != model_name:
        raise IndexNotReadyError(
            f"index model mismatch: active index uses {manifest.get('embedding_model')!r}, requested {model_name!r}"
        )
    if manifest.get("model_artifact_identity") != model_artifact_identity:
        raise IndexNotReadyError("active index model artifact changed; run index")
    signature = _signature(
        corpus.file_hashes,
        model_name,
        qdrant_version,
        fastembed_version,
        registry_hash,
        model_artifact_identity,
    )
    if manifest.get("status") != "complete":
        raise IndexNotReadyError("the active index is partial; run python -m nextlab_rag index")
    if manifest.get("corpus_files") != corpus.file_hashes or manifest.get("signature") != signature:
        raise IndexNotReadyError("the active index is stale for the current corpus or package versions; run index")
    if manifest.get("qdrant_client_version") != qdrant_version or manifest.get("fastembed_version") != fastembed_version:
        raise IndexNotReadyError("the active index uses different library versions; re-run index")
    return generation_dir, manifest, corpus


def search_corpus(
    question: str,
    repo_root: Path | str,
    *,
    rag_dir: Path | str | None = None,
    model_name: str = EMBEDDING_MODEL,
    embedder: Any | None = None,
    limit: int = 5,
    text_query: str | None = None,
) -> list[dict[str, Any]]:
    """Search by dense vectors and BM25, then fuse ranks with RRF.

    ``text_query`` overrides the BM25 query only. The dense query stays
    ``question``. This lets callers expand acronyms for keyword search without
    changing the embedding.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if limit < 1 or limit > 50:
        raise ValueError("limit must be between 1 and 50")
    bm25_query = text_query.strip() if text_query and text_query.strip() else question
    root = Path(repo_root).resolve()
    storage = Path(rag_dir).resolve() if rag_dir else root / ".rag"
    with _index_lock(storage):
        if embedder is None:
            embedder = _load_embedder(model_name, storage / "cache")
        model_artifact_identity = _model_artifact_identity(model_name, embedder)
        generation_dir, manifest, corpus = _validate_current_index(
            root,
            storage,
            model_name,
            model_artifact_identity,
        )
        query_vector = next(iter(embedder.query_embed(question)))
        client = QdrantClient(path=str(generation_dir / "qdrant"))
        try:
            count = _collection_count(client)
            if count != manifest.get("passage_count"):
                raise IndexNotReadyError("the active collection is partial or contains stale points; run index")
            if count == 0:
                return []
            candidate_limit = min(max(20, limit * 4), count)
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    models.Prefetch(
                        query=[float(value) for value in query_vector],
                        using=DENSE_VECTOR_NAME,
                        limit=candidate_limit,
                    ),
                    models.Prefetch(
                        query=models.TextQuery(text=bm25_query),
                        using=TEXT_FIELD,
                        limit=candidate_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        finally:
            client.close()

        hits: list[dict[str, Any]] = []
        for point in response.points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            doc_id = payload.get("doc_id")
            if not isinstance(chunk_id, str) or not isinstance(doc_id, str):
                raise IndexNotReadyError("an active passage is missing its corpus join keys; run index")
            chunk = corpus.chunks_by_id.get(chunk_id)
            document = corpus.documents.get(doc_id)
            provenance = corpus.provenance.get(doc_id)
            if chunk is None or chunk.get("doc_id") != doc_id or document is None or provenance is None:
                raise IndexNotReadyError("an active passage has no matching corpus metadata; run index")
            raw_correction_ids = payload.get("review_correction_ids", [])
            if not isinstance(raw_correction_ids, list) or not all(
                isinstance(correction_id, str) for correction_id in raw_correction_ids
            ):
                raise IndexNotReadyError("an active passage has invalid correction references; run index")
            correction_rows: list[dict[str, Any]] = []
            for correction_id in raw_correction_ids:
                correction = corpus.corrections.get(correction_id)
                if correction is None:
                    raise IndexNotReadyError("an active passage references missing corrections; run index")
                correction_rows.append(correction)
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "passage_text": payload.get(TEXT_FIELD, ""),
                    "passage_start_char": payload.get("passage_start_char", 0),
                    "passage_end_char": payload.get("passage_end_char", 0),
                    "score": point.score,
                    "chunk": chunk,
                    "document": document,
                    "provenance": provenance,
                    "review_correction_ids": raw_correction_ids,
                    "review_corrections": correction_rows,
                    "review_uncertainties": corpus.uncertainties_by_doc.get(doc_id, []),
                }
            )
        return hits
