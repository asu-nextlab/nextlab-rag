from __future__ import annotations

import json
import hashlib
import re
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import QdrantClient

from nextlab_rag import indexing
from nextlab_rag import __main__ as rag_cli
from nextlab_rag.__main__ import _print_search
from nextlab_rag.answering import ABSTENTION_TEXT, RagConfigError, ask, chat_completion, validate_citations
from nextlab_rag.indexing import CheckpointTimeoutError, _index_lock, index_corpus, search_corpus, split_passages


class FakeEmbedder:
    embedding_size = 3

    def __init__(self, revision: str = "test-v1") -> None:
        self.artifact_identity = {
            "source": "test",
            "revision": revision,
            "files": {"fake-model.onnx": hashlib.sha256(revision.encode()).hexdigest()},
        }
        self.embedded: list[str] = []

    def token_count(self, text: str, batch_size: int = 1024) -> int:
        del batch_size
        return len(re.findall(r"\w+|[^\w\s]", text))

    def embed(self, texts: list[str], batch_size: int = 32, parallel: int | None = None):
        del batch_size, parallel
        for text in texts:
            self.embedded.append(text)
            yield self._vector(text)

    def query_embed(self, query: str):
        yield self._vector(query)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        if "indigo" in lowered or "cobalt-needle" in lowered:
            return [1.0, 0.0, 0.0]
        if "telescope" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def write_export(repo_root: Path, chunks: list[dict[str, Any]]) -> None:
    dataset = repo_root / "dataset"
    review = dataset / "review"
    review.mkdir(parents=True, exist_ok=True)
    documents: dict[str, dict[str, Any]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        doc_id = chunk["doc_id"]
        documents.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "canonical_url": f"https://example.test/{doc_id}",
                "title": f"Title for {doc_id}",
                "source_type": "linkedin_activity_feed_capture",
                "rights_status": "unreviewed-public",
                "license_status": "unknown",
                "text": chunk["text"],
                "source_link_summary": {
                    "original_post_url": "https://example.test/original",
                    "reshare_url": "https://example.test/reshare",
                    "match_status": "uncertain",
                },
                "capture_completeness": "profile-wide total remains uncertain",
                "completeness_note": "The export contains one reviewed post.",
            },
        )
        provenance.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "canonical_url": f"https://example.test/{doc_id}",
                "review_correction_ids": ["C001"],
                "source_records": [
                    {
                        "source_doc_id": doc_id,
                        "source_file": "history/capture/documents.jsonl",
                        "source_line": 1,
                        "raw_retained": False,
                    }
                ],
            },
        )
    for name, rows in (
        ("chunks.jsonl", chunks),
        ("documents.jsonl", list(documents.values())),
        ("provenance.jsonl", list(provenance.values())),
        (
            "review/correction-ledger.jsonl",
            [
                {
                    "correction_id": "C001",
                    "confidence": "high",
                    "replacement": "corrected term",
                    "evidence_url": "https://example.test/evidence",
                }
            ],
        ),
        (
            "review/uncertainties.jsonl",
            [
                {
                    "doc_id": "doc-indigo",
                    "uncertainty_id": "U001",
                    "note": "Original post versus reshare is uncertain.",
                }
            ],
        ),
    ):
        path = dataset / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture_chunks() -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": "doc-indigo-000",
            "doc_id": "doc-indigo",
            "chunk_index": 0,
            "canonical_url": "https://example.test/doc-indigo",
            "title": "Indigo River opening",
            "section_heading": "Announcement",
            "source_type": "linkedin_activity_feed_capture",
            "rights_status": "unreviewed-public",
            "license_status": "unknown",
            "custom_chunk_field": "preserve me",
            "text": "The Indigo River center opened in Tempe in 2022.",
        },
        {
            "chunk_id": "doc-stars-000",
            "doc_id": "doc-stars",
            "chunk_index": 0,
            "canonical_url": "https://example.test/doc-stars",
            "title": "Telescope notes",
            "section_heading": "Observations",
            "source_type": "external_seed_html",
            "rights_status": "unreviewed-public",
            "license_status": "unknown",
            "text": "The telescope guide describes star maps and lens calibration.",
        },
    ]


def test_index_reopens_persisted_hybrid_collection_and_keeps_source_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = fixture_chunks()
    write_export(tmp_path, chunks)
    rag_dir = tmp_path / ".rag"

    manifest = index_corpus(tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder())
    load_corpus = indexing._load_corpus
    corpus_loads = 0

    def count_corpus_load(path: Path):
        nonlocal corpus_loads
        corpus_loads += 1
        return load_corpus(path)

    monkeypatch.setattr(indexing, "_load_corpus", count_corpus_load)
    hits = search_corpus(
        "When did Indigo River open in Tempe?",
        tmp_path,
        rag_dir=rag_dir,
        embedder=FakeEmbedder(),
        limit=2,
    )

    assert manifest["status"] == "complete"
    assert manifest["chunk_count"] == 2
    assert hits[0]["chunk_id"] == "doc-indigo-000"
    assert corpus_loads == 1
    assert hits[0]["chunk"] == chunks[0]
    assert hits[0]["chunk"]["custom_chunk_field"] == "preserve me"
    assert hits[0]["document"]["source_link_summary"]["match_status"] == "uncertain"
    assert hits[0]["document"]["rights_status"] == "unreviewed-public"
    assert hits[0]["provenance"]["source_records"][0]["raw_retained"] is False
    assert hits[0]["review_corrections"][0]["correction_id"] == "C001"
    assert hits[0]["review_uncertainties"][0]["uncertainty_id"] == "U001"


def test_qdrant_stores_only_compact_passage_payload(tmp_path: Path) -> None:
    write_export(tmp_path, fixture_chunks())
    rag_dir = tmp_path / ".rag"
    manifest = index_corpus(tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder())
    generation = json.loads((rag_dir / "current.json").read_text(encoding="utf-8"))["generation"]
    client = QdrantClient(path=str(rag_dir / "indexes" / generation / "qdrant"))
    try:
        points, _ = client.scroll(collection_name="nextlab_rag", limit=1, with_payload=True)
    finally:
        client.close()

    assert len(points) == 1
    payload = points[0].payload or {}
    assert set(payload) == {
        "chunk_id",
        "doc_id",
        "passage_text",
        "passage_start_char",
        "passage_end_char",
        "review_correction_ids",
    }
    assert len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) <= (
        len(payload["passage_text"].encode("utf-8")) + 256
    )
    assert manifest["schema_version"] == 2


def test_old_schema_stage_and_active_generation_rebuild_without_opening_qdrant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_export(tmp_path, fixture_chunks())
    rag_dir = tmp_path / ".rag"
    old_generation = rag_dir / "indexes" / "legacy-v1"
    old_stage = rag_dir / "staging"
    for directory in (old_generation, old_stage):
        (directory / "qdrant").mkdir(parents=True)
        (directory / "qdrant" / "legacy-payloads.marker").write_text("bloated", encoding="utf-8")
        (directory / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "status": "complete", "passage_count": 2}),
            encoding="utf-8",
        )
    rag_dir.mkdir(parents=True, exist_ok=True)
    (rag_dir / "current.json").write_text(json.dumps({"generation": "legacy-v1"}), encoding="utf-8")
    original_client = indexing.QdrantClient
    opened_paths: list[str] = []

    def reject_legacy_open(*args: Any, **kwargs: Any):
        path = kwargs.get("path", args[0] if args else "")
        opened_paths.append(str(path))
        if "legacy-v1" in str(path) or "legacy-payloads.marker" in str(path):
            raise AssertionError("legacy Qdrant must not be opened")
        return original_client(*args, **kwargs)

    monkeypatch.setattr(indexing, "QdrantClient", reject_legacy_open)
    manifest = index_corpus(tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder())

    assert manifest["schema_version"] == 2
    assert manifest["status"] == "complete"
    assert not old_generation.exists()
    assert not old_stage.exists()
    assert all("legacy-v1" not in path for path in opened_paths)


def test_checkpoint_watchdog_covers_corpus_preparation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_export(tmp_path, fixture_chunks())

    def stalled_load(_: Path):
        time.sleep(1)
        raise AssertionError("watchdog should interrupt corpus preparation")

    monkeypatch.setattr(indexing, "_load_corpus", stalled_load)
    with pytest.raises(RuntimeError, match="no durable checkpoint progress"):
        index_corpus(
            tmp_path,
            rag_dir=tmp_path / ".rag",
            embedder=FakeEmbedder(),
            checkpoint_timeout_seconds=0.05,
        )

    progress = [json.loads(line) for line in (tmp_path / ".rag" / "progress.jsonl").read_text().splitlines()]
    assert progress[0]["model_stage"] == "startup"
    assert not (tmp_path / ".rag" / "staging" / "manifest.json").exists()


def test_checkpoint_watchdog_preserves_last_batch_and_does_not_publish(tmp_path: Path) -> None:
    write_export(tmp_path, fixture_chunks())
    rag_dir = tmp_path / ".rag"

    class StallsSecondBatch(FakeEmbedder):
        calls = 0

        def embed(self, texts: list[str], batch_size: int = 32, parallel: int | None = None):
            self.calls += 1
            if self.calls == 2:
                time.sleep(3)
            yield from super().embed(texts, batch_size=batch_size, parallel=parallel)

    with pytest.raises(CheckpointTimeoutError, match="no durable checkpoint progress"):
        index_corpus(
            tmp_path,
            rag_dir=rag_dir,
            embedder=StallsSecondBatch(),
            batch_size=1,
            checkpoint_timeout_seconds=2,
        )

    checkpoint = json.loads((rag_dir / "staging" / "manifest.json").read_text(encoding="utf-8"))
    client = QdrantClient(path=str(rag_dir / "staging" / "qdrant"))
    try:
        assert client.count(collection_name="nextlab_rag", exact=True).count == 1
    finally:
        client.close()
    assert checkpoint["status"] == "building"
    assert checkpoint["completed_passages"] == 1
    assert not (rag_dir / "current.json").exists()


def test_cli_reports_checkpoint_timeout_with_exit_code_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def timeout_index(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise CheckpointTimeoutError("indexing made no durable checkpoint progress")

    monkeypatch.setattr(rag_cli, "index_corpus", timeout_index)

    assert rag_cli.main(["index", "--repo-root", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: indexing made no durable checkpoint progress" in captured.err


@pytest.mark.parametrize("abstract", [False, True])
def test_sd_notify_supports_filesystem_and_abstract_sockets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, abstract: bool
) -> None:
    receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    address = f"@nextlab-rag-{tmp_path.name}" if abstract else str(tmp_path / "notify.sock")
    receiver.bind("\0" + address[1:] if abstract else address)
    receiver.settimeout(1)
    monkeypatch.setenv("NOTIFY_SOCKET", address)
    try:
        indexing._sd_notify("READY=1\nWATCHDOG=1")
        assert receiver.recv(64).decode("utf-8") == "READY=1\nWATCHDOG=1"
    finally:
        receiver.close()


def test_progress_telemetry_and_manager_heartbeat_track_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_export(tmp_path, fixture_chunks())
    messages: list[str] = []
    monkeypatch.setattr(indexing, "_sd_notify", messages.append)

    manifest = index_corpus(tmp_path, rag_dir=tmp_path / ".rag", embedder=FakeEmbedder(), batch_size=1)
    events = [json.loads(line) for line in (tmp_path / ".rag" / "progress.jsonl").read_text().splitlines()]
    checkpoints = [event for event in events if event["model_stage"] == "embedding"]

    assert messages == ["READY=1\nWATCHDOG=1", "WATCHDOG=1", "WATCHDOG=1"]
    assert [event["completed_passages"] for event in checkpoints] == [1, 2]
    assert all(event["total_passages"] == manifest["passage_count"] for event in checkpoints)
    assert all(event["rss_bytes"] is not None and event["peak_rss_bytes"] > 0 for event in checkpoints)


def test_search_text_displays_review_uncertainty_and_corrections(capsys: pytest.CaptureFixture[str]) -> None:
    _print_search(
        [
            {
                "chunk_id": "doc-indigo-000",
                "chunk": {"section_heading": "Announcement", "rights_status": "unreviewed-public"},
                "document": {"title": "Indigo River", "source_link_summary": {"match_status": "uncertain"}},
                "review_uncertainties": [{"note": "Original post versus reshare is uncertain."}],
                "review_correction_ids": ["C001"],
                "review_corrections": [{"correction_id": "C001", "replacement": "corrected term"}],
                "passage_text": "A reviewed excerpt.",
            }
        ]
    )

    output = capsys.readouterr().out
    assert "match_status" in output and "uncertain" in output
    assert "Original post versus reshare is uncertain." in output
    assert "corrected term" in output


def test_reindex_is_idempotent_and_removed_chunks_do_not_survive(tmp_path: Path) -> None:
    chunks = fixture_chunks()
    write_export(tmp_path, chunks)
    rag_dir = tmp_path / ".rag"
    first = index_corpus(tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder())
    repeated = index_corpus(tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder())
    assert repeated["passage_count"] == first["passage_count"]

    write_export(tmp_path, [chunks[1]])
    rebuilt = index_corpus(tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder())
    hits = search_corpus(
        "Indigo River Tempe",
        tmp_path,
        rag_dir=rag_dir,
        embedder=FakeEmbedder(),
        limit=4,
    )

    assert rebuilt["chunk_count"] == 1
    assert rebuilt["passage_count"] == 1
    assert all(hit["chunk_id"] != "doc-indigo-000" for hit in hits)
    assert len(list((rag_dir / "indexes").iterdir())) == 1


def test_interrupted_index_resumes_after_last_completed_batch(tmp_path: Path) -> None:
    chunks = fixture_chunks()
    write_export(tmp_path, chunks)
    rag_dir = tmp_path / ".rag"

    class FailSecondBatch(FakeEmbedder):
        calls = 0

        def embed(self, texts: list[str], batch_size: int = 32, parallel: int | None = None):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated interruption")
            yield from super().embed(texts, batch_size=batch_size, parallel=parallel)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        index_corpus(tmp_path, rag_dir=rag_dir, embedder=FailSecondBatch(), batch_size=1)

    checkpoint = json.loads((rag_dir / "staging" / "manifest.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "building"
    assert checkpoint["completed_passages"] == 1
    with pytest.raises(RuntimeError, match="no completed index"):
        search_corpus("Indigo River", tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder())

    staged_client = QdrantClient(path=str(rag_dir / "staging" / "qdrant"))
    staged_client.delete_payload_index(collection_name="nextlab_rag", field_name="passage_text")
    staged_client.close()

    completed = index_corpus(tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder(), batch_size=1)
    assert completed["status"] == "complete"
    assert completed["completed_passages"] == completed["passage_count"] == 2
    hits = search_corpus("Indigo River", tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder())
    assert hits[0]["chunk_id"] == "doc-indigo-000"


def test_resume_rebuilds_when_an_unknown_point_replaces_a_checkpointed_point(tmp_path: Path) -> None:
    write_export(tmp_path, fixture_chunks())
    rag_dir = tmp_path / ".rag"

    class FailSecondBatch(FakeEmbedder):
        calls = 0

        def embed(self, texts: list[str], batch_size: int = 32, parallel: int | None = None):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated interruption")
            yield from super().embed(texts, batch_size=batch_size, parallel=parallel)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        index_corpus(tmp_path, rag_dir=rag_dir, embedder=FailSecondBatch(), batch_size=1)

    client = QdrantClient(path=str(rag_dir / "staging" / "qdrant"))
    try:
        points, _ = client.scroll(collection_name="nextlab_rag", limit=1, with_payload=True, with_vectors=True)
        point = points[0]
        client.delete(collection_name="nextlab_rag", points_selector=[point.id], wait=True)
        client.upsert(
            collection_name="nextlab_rag",
            points=[indexing.models.PointStruct(
                id="00000000-0000-0000-0000-000000000001",
                vector=point.vector,
                payload={
                    "chunk_id": "foreign-chunk",
                    "doc_id": "foreign-document",
                    "passage_text": "Foreign passage.",
                    "passage_start_char": 0,
                    "passage_end_char": 16,
                    "review_correction_ids": [],
                },
            )],
            wait=True,
        )
    finally:
        client.close()

    manifest = index_corpus(tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder(), batch_size=1)
    assert manifest["status"] == "complete"
    hits = search_corpus("Indigo River", tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder())
    assert {hit["chunk_id"] for hit in hits} == {"doc-indigo-000", "doc-stars-000"}


def test_resume_replays_uncheckpointed_upsert_without_reembedding_completed_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = fixture_chunks()
    write_export(tmp_path, chunks)
    rag_dir = tmp_path / ".rag"
    original_client = indexing.QdrantClient

    class UpsertThenInterrupt:
        calls = 0

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.client = original_client(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.client, name)

        def upsert(self, *args: Any, **kwargs: Any) -> Any:
            result = self.client.upsert(*args, **kwargs)
            type(self).calls += 1
            if type(self).calls == 2:
                raise RuntimeError("simulated stop after persisted upsert")
            return result

    monkeypatch.setattr(indexing, "QdrantClient", UpsertThenInterrupt)
    with pytest.raises(RuntimeError, match="after persisted upsert"):
        index_corpus(tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder(), batch_size=1)

    checkpoint = json.loads((rag_dir / "staging" / "manifest.json").read_text(encoding="utf-8"))
    staged_client = original_client(path=str(rag_dir / "staging" / "qdrant"))
    try:
        stored_count = int(staged_client.count(collection_name="nextlab_rag", exact=True).count)
    finally:
        staged_client.close()
    assert checkpoint["completed_passages"] == 1
    assert stored_count == 2

    monkeypatch.setattr(indexing, "QdrantClient", original_client)
    resumed_embedder = FakeEmbedder()
    completed = index_corpus(tmp_path, rag_dir=rag_dir, embedder=resumed_embedder, batch_size=1)

    assert completed["status"] == "complete"
    assert completed["completed_passages"] == 2
    assert resumed_embedder.embedded == [chunks[1]["text"]]


def test_resume_rebuilds_when_model_artifact_revision_changes(tmp_path: Path) -> None:
    chunks = fixture_chunks()
    write_export(tmp_path, chunks)
    rag_dir = tmp_path / ".rag"

    class FailSecondBatch(FakeEmbedder):
        calls = 0

        def embed(self, texts: list[str], batch_size: int = 32, parallel: int | None = None):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated interruption")
            yield from super().embed(texts, batch_size=batch_size, parallel=parallel)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        index_corpus(tmp_path, rag_dir=rag_dir, embedder=FailSecondBatch(), batch_size=1)
    checkpoint = json.loads((rag_dir / "staging" / "manifest.json").read_text(encoding="utf-8"))
    assert checkpoint["completed_passages"] == 1

    changed_model = FakeEmbedder(revision="test-v2")
    completed = index_corpus(tmp_path, rag_dir=rag_dir, embedder=changed_model, batch_size=1)

    assert completed["status"] == "complete"
    assert len(changed_model.embedded) == completed["passage_count"]
    assert completed["model_artifact_identity"]["revision"] == "test-v2"


def test_resume_discards_stage_without_a_recorded_model_artifact_identity(tmp_path: Path) -> None:
    chunks = fixture_chunks()
    write_export(tmp_path, chunks)
    rag_dir = tmp_path / ".rag"

    class FailSecondBatch(FakeEmbedder):
        calls = 0

        def embed(self, texts: list[str], batch_size: int = 32, parallel: int | None = None):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated interruption")
            yield from super().embed(texts, batch_size=batch_size, parallel=parallel)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        index_corpus(tmp_path, rag_dir=rag_dir, embedder=FailSecondBatch(), batch_size=1)

    stage_manifest_path = rag_dir / "staging" / "manifest.json"
    stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    del stage_manifest["model_artifact_identity"]
    stage_manifest_path.write_text(json.dumps(stage_manifest), encoding="utf-8")
    resumed = FakeEmbedder()
    completed = index_corpus(tmp_path, rag_dir=rag_dir, embedder=resumed, batch_size=1)

    assert completed["status"] == "complete"
    assert len(resumed.embedded) == completed["passage_count"]


def test_search_rejects_changed_downloaded_model_artifact(tmp_path: Path) -> None:
    write_export(tmp_path, fixture_chunks())
    rag_dir = tmp_path / ".rag"

    model_dir = tmp_path / "models--test" / "snapshots" / "immutable-revision"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "model.onnx"
    tokenizer_file = model_dir / "tokenizer.json"
    model_file.write_bytes(b"model weights v1")
    tokenizer_file.write_bytes(b"tokenizer v1")

    class FilesystemEmbedder(FakeEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.model = SimpleNamespace(_model_dir=model_dir)

    first = FilesystemEmbedder()
    index_corpus(tmp_path, rag_dir=rag_dir, embedder=first)
    manifest = json.loads((rag_dir / "indexes" / json.loads((rag_dir / "current.json").read_text())["generation"] / "manifest.json").read_text())
    assert manifest["model_artifact_identity"]["revision"] == "immutable-revision"
    assert manifest["model_artifact_identity"]["files"]["model.onnx"] == hashlib.sha256(b"model weights v1").hexdigest()

    tokenizer_file.write_bytes(b"tokenizer v2")
    with pytest.raises(RuntimeError, match="model artifact"):
        search_corpus(
            "Indigo River",
            tmp_path,
            rag_dir=rag_dir,
            embedder=FilesystemEmbedder(),
        )


def test_search_serializes_local_qdrant_reads_with_index_publication(tmp_path: Path) -> None:
    write_export(tmp_path, fixture_chunks())
    rag_dir = tmp_path / ".rag"
    embedder = FakeEmbedder()
    index_corpus(tmp_path, rag_dir=rag_dir, embedder=embedder)

    with _index_lock(rag_dir):
        with pytest.raises(RuntimeError, match="another RAG command"):
            search_corpus("Indigo River", tmp_path, rag_dir=rag_dir, embedder=embedder)


def test_search_rejects_an_index_after_the_corpus_changes(tmp_path: Path) -> None:
    chunks = fixture_chunks()
    write_export(tmp_path, chunks)
    rag_dir = tmp_path / ".rag"
    index_corpus(tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder())
    write_export(tmp_path, [dict(chunks[0], text="Changed source text.")])

    with pytest.raises(RuntimeError, match="stale"):
        search_corpus("Indigo River", tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder())


def test_search_rejects_a_different_embedding_model(tmp_path: Path) -> None:
    write_export(tmp_path, fixture_chunks())
    rag_dir = tmp_path / ".rag"
    index_corpus(tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder())

    with pytest.raises(RuntimeError, match="index model mismatch"):
        search_corpus(
            "Indigo River",
            tmp_path,
            rag_dir=rag_dir,
            model_name="BAAI/bge-base-en-v1.5",
            embedder=FakeEmbedder(),
        )


def test_passage_splitter_covers_long_text_within_token_budget() -> None:
    text = "Opening. " + ("ordinary detail words " * 90) + " LATE_MARKER is at the end."
    count = lambda value: len(value.split())

    passages = split_passages(text, token_count=count, max_tokens=12, max_chars=90, overlap_chars=8)

    assert passages[0].start_char == 0
    assert passages[-1].end_char == len(text)
    assert all(count(passage.text) <= 12 for passage in passages)
    covered = 0
    for passage in passages:
        assert passage.start_char <= covered
        covered = max(covered, passage.end_char)
    assert covered == len(text)
    assert "LATE_MARKER" in passages[-1].text


def test_passage_splitter_rejects_a_single_character_over_the_token_budget() -> None:
    with pytest.raises(RuntimeError, match="cannot fit"):
        split_passages("x", token_count=lambda _: 2, max_tokens=1)


def test_late_chunk_content_is_retrievable_and_cited_by_parent_chunk(tmp_path: Path) -> None:
    late_text = "Introductory notes. " + ("ordinary background details " * 180)
    late_text += " The cobalt-needle marker appears only at the end."
    chunk = dict(fixture_chunks()[0], text=late_text)
    write_export(tmp_path, [chunk])
    rag_dir = tmp_path / ".rag"
    index_corpus(tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder())

    hits = search_corpus(
        "cobalt-needle marker",
        tmp_path,
        rag_dir=rag_dir,
        embedder=FakeEmbedder(),
        limit=3,
    )

    assert hits
    assert hits[0]["chunk_id"] == chunk["chunk_id"]
    assert hits[0]["passage_start_char"] > 0
    assert "cobalt-needle" in hits[0]["passage_text"].casefold()
    assert hits[0]["chunk"]["text"] == late_text


class CompletionHandler(BaseHTTPRequestHandler):
    answer = "The center opened in 2022. [S1]"
    requests: list[tuple[dict[str, Any], str | None]] = []
    redirect_to: str | None = None

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        type(self).requests.append((body, self.headers.get("Authorization")))
        if type(self).redirect_to:
            self.send_response(302)
            self.send_header("Location", type(self).redirect_to)
            self.end_headers()
            return
        payload = {"choices": [{"message": {"content": type(self).answer}}]}
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args: Any) -> None:
        del format, args


class RedirectThenTrickleHandler(BaseHTTPRequestHandler):
    request_paths: list[str] = []
    body = b'{"choices":[{"message":{"content":"slow answer"}}]}'

    def do_POST(self) -> None:
        type(self).request_paths.append(self.path)
        if self.path == "/v1/chat/completions":
            self.send_response(307)
            self.send_header("Location", "/v1/slow")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(type(self).body)))
        self.end_headers()
        try:
            for value in type(self).body:
                self.wfile.write(bytes([value]))
                self.wfile.flush()
                time.sleep(0.08)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: Any) -> None:
        del format, args


class CountingHandler(BaseHTTPRequestHandler):
    requests = 0

    def do_POST(self) -> None:
        type(self).requests += 1
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        del format, args


def serve(handler: type[BaseHTTPRequestHandler]):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}/v1"


def stop_server(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_ask_calls_configured_chat_endpoint_and_validates_citations(tmp_path: Path) -> None:
    write_export(tmp_path, fixture_chunks())
    rag_dir = tmp_path / ".rag"
    index_corpus(tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder())
    CompletionHandler.requests = []
    CompletionHandler.answer = "The center opened in 2022. [S1]"
    CompletionHandler.redirect_to = None
    server, thread, base_url = serve(CompletionHandler)
    try:
        result = ask(
            "When did Indigo River open in Tempe?",
            tmp_path,
            rag_dir=rag_dir,
            embedder=FakeEmbedder(),
            chat_base_url=base_url,
            chat_model="test-model",
            api_key="test-secret",
        )
    finally:
        stop_server(server, thread)

    assert result["abstained"] is False
    assert result["citation_validation"]["valid"] is True
    assert result["answer"] == "The center opened in 2022. [S1]"
    body, authorization = CompletionHandler.requests[0]
    assert body["model"] == "test-model"
    assert authorization == "Bearer test-secret"
    assert "untrusted" in body["messages"][0]["content"].casefold()
    assert "preserve corrections, uncertainty" in body["messages"][0]["content"].casefold()
    assert "Original post versus reshare is uncertain." in body["messages"][1]["content"]
    assert "corrected term" in body["messages"][1]["content"]
    assert "The export contains one reviewed post." in body["messages"][1]["content"]


def test_ask_uses_generator_decision_for_missing_fact_about_known_entity(tmp_path: Path) -> None:
    write_export(tmp_path, fixture_chunks())
    rag_dir = tmp_path / ".rag"
    index_corpus(tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder())
    CompletionHandler.requests = []
    CompletionHandler.answer = ABSTENTION_TEXT
    CompletionHandler.redirect_to = None
    server, thread, base_url = serve(CompletionHandler)

    try:
        result = ask(
            "Who founded Indigo River?",
            tmp_path,
            rag_dir=rag_dir,
            embedder=FakeEmbedder(),
            chat_base_url=base_url,
            chat_model="test-model",
        )
    finally:
        stop_server(server, thread)

    assert result["abstained"] is True
    assert result["answer"] == ABSTENTION_TEXT
    assert result["sources"]
    assert CompletionHandler.requests


def test_ask_abstains_without_config_when_index_has_no_passages(tmp_path: Path) -> None:
    write_export(tmp_path, [])
    rag_dir = tmp_path / ".rag"
    index_corpus(tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder())

    result = ask("What is in the export?", tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder())

    assert result["abstained"] is True
    assert result["answer"] == ABSTENTION_TEXT
    assert result["sources"] == []


def test_ask_allows_supported_paraphrase_without_literal_evidence_overlap(tmp_path: Path) -> None:
    class ParaphraseEmbedder(FakeEmbedder):
        def query_embed(self, query: str):
            del query
            yield [1.0, 0.0, 0.0]

    write_export(tmp_path, fixture_chunks())
    rag_dir = tmp_path / ".rag"
    index_corpus(tmp_path, rag_dir=rag_dir, embedder=ParaphraseEmbedder())
    CompletionHandler.requests = []
    CompletionHandler.answer = "The center began operations in 2022. [S1]"
    CompletionHandler.redirect_to = None
    server, thread, base_url = serve(CompletionHandler)

    try:
        result = ask(
            "In what calendar year did the facility first welcome visitors?",
            tmp_path,
            rag_dir=rag_dir,
            embedder=ParaphraseEmbedder(),
            chat_base_url=base_url,
            chat_model="test-model",
        )
    finally:
        stop_server(server, thread)

    assert result["abstained"] is False
    assert result["answer"] == "The center began operations in 2022. [S1]"
    assert result["sources"][0]["chunk_id"] == "doc-indigo-000"


def test_supported_ask_requires_an_explicit_endpoint(tmp_path: Path) -> None:
    write_export(tmp_path, fixture_chunks())
    rag_dir = tmp_path / ".rag"
    index_corpus(tmp_path, rag_dir=rag_dir, embedder=FakeEmbedder())

    with pytest.raises(RagConfigError, match="NEXTLAB_RAG_BASE_URL"):
        ask(
            "When did Indigo River open in Tempe?",
            tmp_path,
            rag_dir=rag_dir,
            embedder=FakeEmbedder(),
        )


def test_cross_origin_redirect_does_not_receive_api_key() -> None:
    CompletionHandler.requests = []
    CompletionHandler.redirect_to = None
    CountingHandler.requests = 0
    target, target_thread, target_url = serve(CountingHandler)
    source, source_thread, source_base_url = serve(CompletionHandler)
    CompletionHandler.redirect_to = target_url + "/chat/completions"
    try:
        with pytest.raises(RuntimeError, match="redirect"):
            chat_completion(
                [{"role": "user", "content": "hello"}],
                base_url=source_base_url,
                model="test-model",
                api_key="test-secret",
            )
    finally:
        stop_server(source, source_thread)
        stop_server(target, target_thread)

    assert CountingHandler.requests == 0


def test_chat_completion_enforces_total_deadline_across_redirect_and_trickled_body() -> None:
    RedirectThenTrickleHandler.request_paths = []
    server, thread, base_url = serve(RedirectThenTrickleHandler)
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="timed out"):
            chat_completion(
                [{"role": "user", "content": "hello"}],
                base_url=base_url,
                model="test-model",
                timeout=0.5,
            )
    finally:
        stop_server(server, thread)
    elapsed = time.monotonic() - started

    assert RedirectThenTrickleHandler.request_paths == ["/v1/chat/completions", "/v1/slow"]
    assert 0.35 <= elapsed < 1.5


def test_citation_validation_rejects_missing_and_unknown_sources() -> None:
    assert validate_citations("A supported claim. [S1]", {"S1"})["valid"] is True
    assert validate_citations("Supported claim. [S1] Unsupported claim.", {"S1"})["valid"] is False
    assert validate_citations("A claim without a citation.", {"S1"})["valid"] is False
    assert validate_citations("A claim cites [S2].", {"S1"})["valid"] is False
