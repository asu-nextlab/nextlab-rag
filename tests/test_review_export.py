import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from nextlab_scraper.review_export import apply_corrections, build_export, load_corrections, read_jsonl, verify_export


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_export_is_timestamp_scoped_deduplicated_and_audits_raw_bytes(tmp_path: Path) -> None:
    run_id = "fixture"
    run = tmp_path / "runs" / run_id
    normalized = run / "normalized"
    transcript_url = "https://www.youtube.com/watch?v=fixture"
    other_url = "https://example.test/article"
    media = b"<html>fixture</html>"
    media_hash = hashlib.sha256(media).hexdigest()
    raw = run / "raw" / "external" / f"{media_hash}.html"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(media)

    docs = [
        {
            "doc_id": "metadata-doc",
            "canonical_url": transcript_url,
            "source_type": "youtube_public_metadata",
            "title": "Fixture video",
            "caption_tracks": [{"language_code": "en"}],
            "raw_path": "",
            "raw_retained": False,
            "rights_status": "unreviewed-public",
            "license_status": "unknown",
            "sections": [],
        },
        {
            "doc_id": "transcript-doc",
            "canonical_url": transcript_url,
            "video_id": "fixture",
            "source_type": "youtube_audio_asr_transcript",
            "title": "Fixture video",
            "raw_path": "",
            "raw_retained": False,
            "rights_status": "unreviewed-public",
            "license_status": "unknown",
            "text": "[00:05.00–00:10.00] wrong.\n[00:11.00–00:12.00] wrong.",
            "sections": [
                {"heading": "00:05.00–00:10.00", "level": 1, "text": "[00:05.00–00:10.00] wrong."},
                {"heading": "00:11.00–00:12.00", "level": 1, "text": "[00:11.00–00:12.00] wrong."},
            ],
        },
        {
            "doc_id": "article-doc",
            "canonical_url": other_url,
            "source_type": "external_asu_article",
            "title": "Fixture article",
            "raw_path": f"raw/external/{media_hash}.html",
            "raw_retained": True,
            "sha256": media_hash,
            "rights_status": "unreviewed-public",
            "license_status": "unknown",
            "sections": [{"heading": "Body", "level": 1, "text": "fixture body"}],
        },
    ]
    chunks = [
        {"chunk_id": "metadata-chunk", "doc_id": "metadata-doc", "canonical_url": transcript_url,
         "chunk_index": 0, "source_type": "youtube_public_metadata", "section_heading": "Public video metadata",
         "text": "Caption track metadata", "rights_status": "unreviewed-public"},
        {"chunk_id": "transcript-chunk-1", "doc_id": "transcript-doc", "canonical_url": transcript_url,
         "chunk_index": 0, "source_type": "youtube_audio_asr_transcript", "section_heading": "00:05.00–00:10.00",
         "text": "[00:05.00–00:10.00] wrong.", "rights_status": "unreviewed-public"},
        {"chunk_id": "transcript-chunk-2", "doc_id": "transcript-doc", "canonical_url": transcript_url,
         "chunk_index": 1, "source_type": "youtube_audio_asr_transcript", "section_heading": "00:11.00–00:12.00",
         "text": "[00:11.00–00:12.00] wrong.", "rights_status": "unreviewed-public"},
        {"chunk_id": "article-chunk", "doc_id": "article-doc", "canonical_url": other_url,
         "chunk_index": 0, "source_type": "external_asu_article", "section_heading": "Body",
         "text": "fixture body", "rights_status": "unreviewed-public"},
    ]
    _write_jsonl(normalized / "documents.jsonl", docs)
    _write_jsonl(normalized / "chunks.jsonl", chunks)
    _write_jsonl(run / "transcripts" / "asr.jsonl", [{
        "video_id": "fixture",
        "segments": [
            {"start": 5.0, "end": 10.0, "text": "wrong."},
            {"start": 11.0, "end": 12.0, "text": "wrong."},
        ],
    }])
    review = run / "review"
    _write_jsonl(review / "correction-ledger.jsonl", [{
        "correction_id": "T1",
        "video_id": "fixture",
        "canonical_url": transcript_url,
        "time_ranges": [[5.0, 10.0]],
        "scoped_segment_count": 1,
        "confidence": "high",
        "substitutions": [{"from": "wrong", "to": "right", "expected_raw_occurrences": 1}],
    }])
    _write_jsonl(review / "uncertainties.jsonl", [])

    report = build_export(tmp_path, run_id, expected_duplicate_urls={transcript_url})
    export = run / "export"
    output_docs = read_jsonl(export / "documents.jsonl")
    output_chunks = read_jsonl(export / "chunks.jsonl")
    by_url = {doc["canonical_url"]: doc for doc in output_docs}
    transcript = by_url[transcript_url]
    assert len(output_docs) == 2
    assert len(output_chunks) == 3
    assert transcript["sections"][0]["text"].endswith("right.")
    assert transcript["sections"][1]["text"].endswith("wrong.")
    assert transcript["merged_metadata"][0]["caption_tracks"] == [{"language_code": "en"}]
    assert len(next(row for row in read_jsonl(export / "provenance.jsonl") if row["doc_id"] == "transcript-doc")["source_records"]) == 2
    assert report["verification"]["raw_file_hashes_verified"] == 1
    assert report["verification"]["correction_overlay"]["matches_source"] is True
    assert {chunk["chunk_id"]: chunk["text"] for chunk in output_chunks}["transcript-chunk-1"].endswith("right.")
    assert {chunk["chunk_id"]: chunk["text"] for chunk in output_chunks}["transcript-chunk-2"].endswith("wrong.")

    # Select a newer normalized snapshot for the same doc_id. The older copy
    # remains in the tree with conflicting chunks and an extra chunk, but is
    # not the source named by primary provenance.
    later_doc = copy.deepcopy(next(row for row in docs if row["doc_id"] == "transcript-doc"))
    later_doc["text"] = "[00:05.00–00:10.00] newer wrong.\n[00:11.00–00:12.00] still wrong."
    later_doc["sections"] = [
        {"heading": "00:05.00–00:10.00", "level": 1, "text": "[00:05.00–00:10.00] newer wrong."},
        {"heading": "00:11.00–00:12.00", "level": 1, "text": "[00:11.00–00:12.00] still wrong."},
    ]
    later_chunks = [copy.deepcopy(chunk) for chunk in chunks if chunk["doc_id"] == "transcript-doc"]
    later_chunks[0]["text"] = "[00:05.00–00:10.00] newer wrong."
    later_chunks[1]["text"] = "[00:11.00–00:12.00] still wrong."
    later_docs_path = tmp_path / "runs" / "later-snapshot" / "normalized" / "documents.jsonl"
    later_chunks_path = later_docs_path.parent / "chunks.jsonl"
    _write_jsonl(later_docs_path, [later_doc])
    _write_jsonl(later_chunks_path, later_chunks)

    rules = load_corrections(review / "correction-ledger.jsonl")
    corrected_later_doc = copy.deepcopy(later_doc)
    apply_corrections(corrected_later_doc, rules)
    original_export_doc = next(row for row in output_docs if row["doc_id"] == "transcript-doc")
    corrected_later_doc["merged_metadata"] = original_export_doc["merged_metadata"]
    output_docs = [row for row in output_docs if row["doc_id"] != "transcript-doc"] + [corrected_later_doc]
    output_chunks = [row for row in output_chunks if row["doc_id"] != "transcript-doc"]
    for chunk in later_chunks:
        corrected_chunk = copy.deepcopy(chunk)
        apply_corrections(corrected_chunk, rules)
        output_chunks.append(corrected_chunk)
    _write_jsonl(export / "documents.jsonl", output_docs)
    _write_jsonl(export / "chunks.jsonl", output_chunks)

    provenance_rows = read_jsonl(export / "provenance.jsonl")
    transcript_provenance = next(row for row in provenance_rows if row["doc_id"] == "transcript-doc")
    transcript_provenance["source_records"][0]["source_file"] = "runs/later-snapshot/normalized/documents.jsonl"
    transcript_provenance["source_records"][0]["source_line"] = 1
    _write_jsonl(export / "provenance.jsonl", provenance_rows)

    old_snapshot_chunks = read_jsonl(normalized / "chunks.jsonl")
    stale_chunks = [dict(chunk, text="stale old snapshot text") if chunk["doc_id"] == "transcript-doc" else chunk
                    for chunk in old_snapshot_chunks]
    stale_chunks.append({
        "chunk_id": "stale-extra-chunk", "doc_id": "transcript-doc", "canonical_url": transcript_url,
        "chunk_index": 2, "source_type": "youtube_audio_asr_transcript", "section_heading": "stale",
        "text": "stale extra chunk", "rights_status": "unreviewed-public",
    })
    _write_jsonl(normalized / "chunks.jsonl", stale_chunks)

    selected_verification = verify_export(tmp_path, export)
    assert selected_verification["correction_overlay"]["checked_documents"] == 2
    assert selected_verification["correction_overlay"]["checked_chunks"] == 3
    assert selected_verification["correction_overlay"]["matches_source"] is True
    selected_export_chunks = {chunk["chunk_id"]: chunk["text"] for chunk in read_jsonl(export / "chunks.jsonl")}
    assert selected_export_chunks["transcript-chunk-1"].endswith("newer right.")
    assert selected_export_chunks["transcript-chunk-2"].endswith("still wrong.")

    # A change to a chunk in the source selected by primary provenance must fail.
    changed_selected_chunks = read_jsonl(later_chunks_path)
    changed_selected_chunks[0]["text"] = "[00:05.00–00:10.00] changed selected chunk."
    _write_jsonl(later_chunks_path, changed_selected_chunks)
    with pytest.raises(ValueError, match="export chunk text differs from timestamp-scoped source"):
        verify_export(tmp_path, export)

    raw.write_bytes(b"altered")
    with pytest.raises(ValueError, match="raw SHA-256 mismatch"):
        verify_export(tmp_path, export)


def test_relocated_dataset_resolves_source_indexes_and_checks_text_and_hashes(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "fixture"
    normalized = run / "normalized"
    review = run / "review"
    raw_bytes = b"original retained HTML"
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    raw_path = run / "raw" / "external" / f"{raw_hash}.html"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(raw_bytes)
    url = "https://www.youtube.com/watch?v=relocated-fixture"
    document = {
        "doc_id": "relocated-doc",
        "canonical_url": url,
        "video_id": "relocated-fixture",
        "source_type": "youtube_audio_asr_transcript",
        "title": "Relocated source",
        "raw_path": f"raw/external/{raw_hash}.html",
        "raw_retained": True,
        "sha256": raw_hash,
        "rights_status": "unreviewed-public",
        "license_status": "unknown",
        "text": "[00:05.00–00:10.00] wrong.",
        "sampled_frame_paths": ["runs/fixture/evidence/frame.png"],
        "raw_source_artifacts": [{"path": f"runs/fixture/raw/external/{raw_hash}.html"}],
        "sections": [{"heading": "00:05.00–00:10.00", "level": 1, "text": "[00:05.00–00:10.00] wrong.",
                      "frame_path": "runs/fixture/evidence/frame.png"}],
    }
    chunk = {
        "chunk_id": "relocated-chunk",
        "doc_id": document["doc_id"],
        "canonical_url": url,
        "source_type": document["source_type"],
        "section_heading": "00:05.00–00:10.00",
        "text": "[00:05.00–00:10.00] wrong.",
        "rights_status": "unreviewed-public",
        "license_status": "unknown",
    }
    _write_jsonl(normalized / "documents.jsonl", [document])
    _write_jsonl(normalized / "chunks.jsonl", [chunk])
    frame = run / "evidence" / "frame.png"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"derived frame")
    asr_path = run / "transcripts" / "asr.jsonl"
    _write_jsonl(asr_path, [{
        "video_id": "relocated-fixture",
        "segments": [{"start": 5.0, "end": 10.0, "text": "wrong."}],
    }])
    _write_jsonl(review / "correction-ledger.jsonl", [{
        "correction_id": "relocated-correction",
        "video_id": "relocated-fixture",
        "canonical_url": url,
        "time_ranges": [[5.0, 10.0]],
        "scoped_segment_count": 1,
        "confidence": "high",
        "substitutions": [{"from": "wrong", "to": "right", "expected_raw_occurrences": 1}],
    }])
    _write_jsonl(review / "uncertainties.jsonl", [])

    build_export(tmp_path, "fixture")
    dataset = tmp_path / "dataset"
    shutil.move(run / "export", dataset)
    dataset_review = dataset / "review"
    dataset_review.mkdir()
    shutil.copy2(review / "correction-ledger.jsonl", dataset_review / "correction-ledger.jsonl")

    source_objects = tmp_path / "sources" / "files"
    source_objects.mkdir(parents=True)
    history_normalized = tmp_path / "history" / "collections" / "fixture" / "normalized"
    history_normalized.parent.mkdir(parents=True)
    shutil.move(normalized, history_normalized)
    history_frame = tmp_path / "history" / "collections" / "fixture" / "evidence" / "frame.png"
    history_frame.parent.mkdir(parents=True)
    shutil.move(frame, history_frame)
    asr_object_hash = hashlib.sha256(asr_path.read_bytes()).hexdigest()
    asr_object = source_objects / asr_object_hash
    shutil.move(asr_path, asr_object)
    raw_object = source_objects / raw_hash
    shutil.move(raw_path, raw_object)

    def source_index_row(old_path: str, path: Path, classification: str) -> dict:
        return {
            "old_path": old_path,
            "original_filename": Path(old_path).name,
            "classification": classification,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
            "stored_file": f"sources/files/{path.name}",
            "source_urls": [url] if old_path.endswith(".html") else [],
            "source_url": url if old_path.endswith(".html") else None,
        }

    raw_old_path = f"runs/fixture/raw/external/{raw_hash}.html"
    asr_old_path = "runs/fixture/transcripts/asr.jsonl"
    source_rows = [
        source_index_row(raw_old_path, raw_object, "original_raw_capture"),
        source_index_row(asr_old_path, asr_object, "original_transcript"),
    ]
    _write_jsonl(tmp_path / "sources" / "index.jsonl", source_rows)
    path_rows = [
        {"old_path": row["old_path"], "new_path": row["stored_file"], "stored_file": row["stored_file"],
         "sha256": row["sha256"], "size_bytes": row["size_bytes"]}
        for row in source_rows
    ]
    for name in ("documents.jsonl", "chunks.jsonl"):
        path_rows.append({
            "old_path": f"runs/fixture/normalized/{name}",
            "new_path": f"history/collections/fixture/normalized/{name}",
        })
    path_rows.append({
        "old_path": "runs/fixture/evidence/frame.png",
        "new_path": "history/collections/fixture/evidence/frame.png",
    })
    _write_jsonl(tmp_path / "history" / "path-map.jsonl", path_rows)

    docs_path = dataset / "documents.jsonl"
    docs = read_jsonl(docs_path)
    docs[0]["raw_path"] = f"sources/files/{raw_hash}"
    docs[0]["sampled_frame_paths"] = ["history/collections/fixture/evidence/frame.png"]
    docs[0]["raw_source_artifacts"][0]["path"] = f"sources/files/{raw_hash}"
    docs[0]["sections"][0]["frame_path"] = "history/collections/fixture/evidence/frame.png"
    _write_jsonl(docs_path, docs)
    provenance_path = dataset / "provenance.jsonl"
    provenance = read_jsonl(provenance_path)
    provenance[0]["source_replacement"] = "runs/fixture/normalized/documents.jsonl"
    source = provenance[0]["source_records"][0]
    source["source_file"] = "history/collections/fixture/normalized/documents.jsonl"
    source["raw_path"] = f"sources/files/{raw_hash}"
    _write_jsonl(provenance_path, provenance)

    moved_verification = verify_export(tmp_path, dataset)
    assert moved_verification["correction_overlay"] == {
        "checked_documents": 1,
        "checked_chunks": 1,
        "matches_source": True,
    }
    assert moved_verification["raw_file_resolutions"][0]["path"] == f"sources/files/{raw_hash}"

    original_frame_path = docs[0]["sampled_frame_paths"][0]
    docs[0]["sampled_frame_paths"] = ["runs/fixture/evidence/missing.png"]
    _write_jsonl(docs_path, docs)
    with pytest.raises(ValueError, match="export document text/provenance differs"):
        verify_export(tmp_path, dataset)
    docs[0]["sampled_frame_paths"] = [original_frame_path]
    _write_jsonl(docs_path, docs)

    original_doc = docs[0]["text"]
    docs[0]["text"] = "altered selected document text"
    _write_jsonl(docs_path, docs)
    with pytest.raises(ValueError, match="export document text/provenance differs"):
        verify_export(tmp_path, dataset)
    docs[0]["text"] = original_doc
    _write_jsonl(docs_path, docs)

    chunks_path = dataset / "chunks.jsonl"
    chunks = read_jsonl(chunks_path)
    chunks[0]["text"] = "altered selected chunk text"
    _write_jsonl(chunks_path, chunks)
    with pytest.raises(ValueError, match="export chunk text differs from timestamp-scoped source"):
        verify_export(tmp_path, dataset)

    raw_object.write_bytes(b"altered retained bytes")
    with pytest.raises(ValueError, match="raw SHA-256 mismatch"):
        verify_export(tmp_path, dataset)
