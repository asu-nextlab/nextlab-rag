"""Build a reviewed, recursively deduplicated export without editing source runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


TRANSCRIPT_TYPES = {"youtube_audio_asr_transcript", "public_audio_download_local_asr_transcript"}
METADATA_TYPE = "youtube_public_metadata"
EXPECTED_DUPLICATE_URLS = {
    "https://www.youtube.com/watch?v=f-2JyjJPZpE",
    "https://www.youtube.com/watch?v=LQH0eHHEAlU",
    "https://www.youtube.com/watch?v=DoRXNK-XOJs",
    "https://www.youtube.com/watch?v=AvgZscNZzYw",
}
HASH_FIELDS = ("sha256", "source_sha256", "source_audio_sha256", "audio_sha256", "media_sha256")
PATH_METADATA_FIELDS = {
    "raw_path", "raw_paths", "html_path", "source_audio_path", "metadata_source_path",
    "source_file", "source_mcp_file", "source_preview_file", "source_replacement",
    "path", "audio_path", "frame_path", "sampled_frame_paths",
    "checkpoint_source_file", "initial_capture_file", "previous_collapsed_capture_file",
}
METADATA_FIELDS = (
    "doc_id", "source_type", "title", "channel", "video_id", "duration_seconds",
    "caption_tracks", "caption_text_available", "caption_body_fetched", "fetched_at",
    "response_bytes", "source_response_bytes", "source_response_sha256", "metadata_source_path",
    "metadata_source_file_sha256", "raw_path", "raw_retained", "rights_status", "license_status",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _load_path_map(repo_root_text: str, index_mtime_ns: int) -> dict[str, dict[str, Any]]:
    del index_mtime_ns
    path = Path(repo_root_text) / "history" / "path-map.jsonl"
    mapping: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for row in read_jsonl(path):
            old_path = row.get("old_path")
            if isinstance(old_path, str) and old_path:
                mapping[old_path] = row
    return mapping


@lru_cache(maxsize=8)
def _load_source_map(repo_root_text: str, index_mtime_ns: int) -> dict[str, dict[str, Any]]:
    del index_mtime_ns
    path = Path(repo_root_text) / "sources" / "index.jsonl"
    mapping: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for row in read_jsonl(path):
            old_path = row.get("old_path")
            if isinstance(old_path, str) and old_path:
                mapping[old_path] = row
    return mapping


def _cached_index(path: Path, loader: Any) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return loader(str(path.parents[1].resolve()), path.stat().st_mtime_ns)


def _path_maps(repo_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    root = repo_root.resolve()
    return (
        _cached_index(root / "history" / "path-map.jsonl", _load_path_map),
        _cached_index(root / "sources" / "index.jsonl", _load_source_map),
    )


def _source_old_path(repo_root: Path, source_file: str) -> str:
    normalized = Path(source_file).as_posix()
    parts = Path(normalized).parts
    if len(parts) >= 2 and parts[0] == "runs":
        return normalized
    if len(parts) >= 4 and parts[:2] == ("history", "collections"):
        return (Path("runs") / parts[2] / Path(*parts[3:])).as_posix()
    path_map, _source_map = _path_maps(repo_root)
    for old_path, row in path_map.items():
        if row.get("new_path") == normalized:
            return old_path
    return normalized


def _old_path_candidates(repo_root: Path, source_file: str, path_value: str) -> list[str]:
    value = Path(path_value)
    if value.is_absolute():
        try:
            return [value.resolve().relative_to(repo_root.resolve()).as_posix()]
        except ValueError:
            return []
    raw = value.as_posix()
    candidates = [raw]
    if raw.startswith(("runs/", "archive/", "sources/", "history/")):
        return candidates
    old_source = Path(_source_old_path(repo_root, source_file))
    parts = old_source.parts
    if len(parts) >= 2 and parts[0] == "runs":
        candidates.append((Path("runs") / parts[1] / value).as_posix())
    elif len(parts) >= 3 and parts[:2] == ("sources", "collections"):
        candidates.append((Path(*parts[:3]) / value).as_posix())
    source_parent = Path(source_file).parent
    candidates.extend(((source_parent / value).as_posix(), (source_parent.parent / value).as_posix()))
    return list(dict.fromkeys(candidates))


def _mapped_file(repo_root: Path, old_path: str) -> Path | None:
    path_map, source_map = _path_maps(repo_root)
    row = source_map.get(old_path) or path_map.get(old_path)
    if row is None:
        return None
    for field in ("stored_file", "new_path"):
        value = row.get(field)
        if not isinstance(value, str) or not value:
            continue
        candidate = repo_root / value
        if candidate.is_file():
            return candidate.resolve()
    return None


def _resolve_metadata_path(repo_root: Path, source_file: str, path_value: str) -> str | None:
    if not path_value:
        return None
    value = Path(path_value)
    if value.is_absolute():
        try:
            return value.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return path_value if value.is_file() else None
    for candidate in _old_path_candidates(repo_root, source_file, path_value):
        mapped = _mapped_file(repo_root, candidate)
        if mapped is not None:
            try:
                return mapped.relative_to(repo_root.resolve()).as_posix()
            except ValueError:
                return None
        direct = repo_root / candidate
        if direct.is_file():
            try:
                return direct.resolve().relative_to(repo_root.resolve()).as_posix()
            except ValueError:
                return None
    return None


def _normalize_path_metadata(repo_root: Path, record: dict[str, Any], source_file: str) -> dict[str, Any]:
    normalized = copy.deepcopy(record)

    def resolve(value: Any) -> Any:
        if isinstance(value, str) and value:
            return _resolve_metadata_path(repo_root, source_file, value) or value
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, dict):
            walk(value)
        return value

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in list(value.items()):
                if key in PATH_METADATA_FIELDS:
                    value[key] = resolve(item)
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(normalized)
    return normalized


def _normalized_directories(repo_root: Path, input_dirs: list[Path] | None) -> list[Path]:
    if input_dirs:
        directories = []
        for supplied in input_dirs:
            path = supplied if supplied.is_absolute() else repo_root / supplied
            normalized = path if path.name == "normalized" else path / "normalized"
            if not (normalized / "documents.jsonl").is_file() or not (normalized / "chunks.jsonl").is_file():
                raise ValueError(f"input directory needs normalized/documents.jsonl and chunks.jsonl: {normalized}")
            directories.append(normalized.resolve())
        return sorted(set(directories))

    directories: set[Path] = set()
    base = repo_root / "history" / "collections" / "20261021T041222Z" / "normalized"
    if (base / "documents.jsonl").is_file() and (base / "chunks.jsonl").is_file():
        directories.add(base.resolve())
    collection_root = repo_root / "sources" / "collections"
    if collection_root.is_dir():
        for path in collection_root.glob("**/normalized/documents.jsonl"):
            if (path.parent / "chunks.jsonl").is_file():
                directories.add(path.parent.resolve())
    if not directories:
        # Compatibility for temporary fixtures and old collection layouts.
        for path in repo_root.glob("runs/**/normalized/documents.jsonl"):
            if (path.parent / "chunks.jsonl").is_file():
                directories.add(path.parent.resolve())
    return sorted(directories)


def discover_inputs(repo_root: Path, input_dirs: list[Path] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read selected normalized pairs from current collections or an explicit input list."""
    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    root = repo_root.resolve()
    for directory in _normalized_directories(root, input_dirs):
        doc_path = directory / "documents.jsonl"
        for line_number, row in enumerate(read_jsonl(doc_path), 1):
            row = copy.deepcopy(row)
            row["_export_source_file"] = doc_path.relative_to(root).as_posix()
            row["_export_source_line"] = line_number
            documents.append(row)
        chunk_path = directory / "chunks.jsonl"
        for line_number, row in enumerate(read_jsonl(chunk_path), 1):
            row = copy.deepcopy(row)
            row["_export_source_file"] = chunk_path.relative_to(root).as_posix()
            row["_export_source_line"] = line_number
            chunks.append(row)
    return documents, chunks


def load_corrections(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    ids = [row.get("correction_id") for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("correction ledger IDs must be present and unique")
    for row in rows:
        if row.get("confidence") != "high":
            raise ValueError(f"only high-confidence corrections may be exported: {row.get('correction_id')}")
        if not row.get("canonical_url") or not row.get("substitutions") or not row.get("time_ranges"):
            raise ValueError(f"incomplete or unscoped correction row: {row.get('correction_id')}")
        for substitution in row["substitutions"]:
            if not substitution.get("from") or not substitution.get("to"):
                raise ValueError(f"invalid substitution in {row.get('correction_id')}")
    return rows


def parse_timestamp_range(value: str) -> tuple[float, float] | None:
    """Parse either MM:SS.ss–MM:SS.ss headings or seconds --> seconds text."""
    clock = re.search(r"(\d+):(\d+(?:\.\d+)?)\s*[–-]\s*(\d+):(\d+(?:\.\d+)?)", value)
    if clock:
        start = int(clock.group(1)) * 60 + float(clock.group(2))
        end = int(clock.group(3)) * 60 + float(clock.group(4))
        return start, end
    seconds = re.search(r"\[?\s*(\d+(?:\.\d+)?)\s*--?>\s*(\d+(?:\.\d+)?)\s*\]?", value)
    if seconds:
        return float(seconds.group(1)), float(seconds.group(2))
    return None


def _overlaps(time_range: tuple[float, float] | None, allowed: list[list[float]]) -> bool:
    if time_range is None:
        return False
    start, end = time_range
    return any(start <= float(rule_end) and end >= float(rule_start) for rule_start, rule_end in allowed)


def _substitution_regex(value: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)")


def validate_source_occurrence_counts(repo_root: Path, rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check ledger counts against the immutable raw ASR segments and their times."""
    source_rows: dict[str, dict[str, Any]] = {}
    transcript_paths: dict[str, Path] = {}
    _path_map, source_map = _path_maps(repo_root)
    for old_path, mapping in source_map.items():
        if old_path.endswith("/transcripts/asr.jsonl"):
            stored_file = mapping.get("stored_file")
            if isinstance(stored_file, str):
                path = repo_root / stored_file
                if path.is_file():
                    transcript_paths[old_path] = path.resolve()
    for pattern in ("runs/**/transcripts/asr.jsonl", "sources/collections/**/transcripts/asr.jsonl"):
        for path in repo_root.glob(pattern):
            if path.is_file():
                transcript_paths[path.relative_to(repo_root).as_posix()] = path.resolve()
    for old_path, path in sorted(transcript_paths.items()):
        for row in read_jsonl(path):
            if row.get("video_id"):
                source_rows[row["video_id"]] = row
    audit: list[dict[str, Any]] = []
    for rule in rules:
        source = source_rows.get(rule.get("video_id"))
        if source is None:
            raise ValueError(f"missing raw ASR source for {rule.get('correction_id')}")
        segments = []
        for segment in source.get("segments", []):
            try:
                segment_range = (float(segment["start"]), float(segment["end"]))
            except (KeyError, TypeError, ValueError):
                continue
            if _overlaps(segment_range, rule["time_ranges"]):
                segments.append(segment)
        if len(segments) != rule.get("scoped_segment_count"):
            raise ValueError(f"timestamp segment count mismatch for {rule['correction_id']}")
        counts: Counter[str] = Counter()
        for segment in segments:
            text = str(segment.get("text", ""))
            ordered = sorted(rule["substitutions"], key=lambda item: -len(item["from"]))
            for substitution in ordered:
                text, count = _substitution_regex(substitution["from"]).subn(substitution["to"], text)
                counts[substitution["from"]] += count
        for substitution in rule["substitutions"]:
            expected = substitution.get("expected_raw_occurrences")
            actual = counts[substitution["from"]]
            if expected is None or expected != actual or actual == 0:
                raise ValueError(
                    f"raw ASR occurrence mismatch for {rule['correction_id']} / {substitution['from']}: "
                    f"ledger={expected} actual={actual}"
                )
        audit.append({
            "correction_id": rule["correction_id"],
            "video_id": rule["video_id"],
            "scoped_segment_count": len(segments),
            "substitutions": [
                {"from": sub["from"], "to": sub["to"], "raw_asr_occurrences": counts[sub["from"]]}
                for sub in rule["substitutions"]
            ],
        })
    return audit


def _apply_scoped_text(
    text: str, rules: list[dict[str, Any]], canonical_url: str, time_range: tuple[float, float] | None,
) -> tuple[str, list[str], Counter[str]]:
    eligible = [rule for rule in rules if rule["canonical_url"] == canonical_url and _overlaps(time_range, rule["time_ranges"])]
    if not eligible or time_range is None:
        return text, [], Counter()
    substitutions: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for rule in eligible:
        for substitution in rule["substitutions"]:
            substitutions.append((len(substitution["from"]), rule, substitution))
    substitutions.sort(key=lambda item: (-item[0], item[1]["correction_id"], item[2]["from"]))
    applied: set[str] = set()
    occurrence_counts: Counter[str] = Counter()
    for _length, rule, substitution in substitutions:
        pattern = _substitution_regex(substitution["from"])
        text, count = pattern.subn(lambda _match, value=substitution["to"]: value, text)
        if count:
            applied.add(rule["correction_id"])
            occurrence_counts[rule["correction_id"]] += count
    return text, sorted(applied), occurrence_counts


def apply_corrections(record: dict[str, Any], rules: list[dict[str, Any]]) -> tuple[list[str], Counter[str]]:
    """Apply only in timestamped text spans covered by each ledger row."""
    canonical_url = record.get("canonical_url", "")
    applied: set[str] = set()
    counts: Counter[str] = Counter()

    def update_field(container: dict[str, Any], time_range: tuple[float, float] | None) -> None:
        value = container.get("text")
        if not isinstance(value, str) or time_range is None:
            return
        updated, ids, found = _apply_scoped_text(value, rules, canonical_url, time_range)
        container["text"] = updated
        applied.update(ids)
        counts.update(found)

    # Chunk text has a single section heading, as do normalized sections.
    heading = record.get("section_heading") or record.get("heading")
    if isinstance(heading, str):
        update_field(record, parse_timestamp_range(heading))
    else:
        # Whole-document transcript text is line-oriented with a timestamp prefix.
        value = record.get("text")
        if isinstance(value, str):
            output_lines = []
            for line in value.splitlines(keepends=True):
                time_range = parse_timestamp_range(line)
                updated, ids, found = _apply_scoped_text(line, rules, canonical_url, time_range)
                output_lines.append(updated)
                applied.update(ids)
                counts.update(found)
            record["text"] = "".join(output_lines)
        for section in record.get("sections", []):
            if isinstance(section, dict):
                section_range = parse_timestamp_range(str(section.get("heading", "")))
                update_field(section, section_range)
        for segment in record.get("transcript_segments", []):
            if isinstance(segment, dict):
                try:
                    segment_range = (float(segment["start"]), float(segment["end"]))
                except (KeyError, TypeError, ValueError):
                    segment_range = None
                update_field(segment, segment_range)
    ids = sorted(applied)
    if ids:
        record["review_correction_ids"] = ids
    return ids, counts


def _preferred_document(group: list[dict[str, Any]]) -> dict[str, Any]:
    if len(group) == 1:
        return group[0]
    transcripts = [row for row in group if row.get("source_type") in TRANSCRIPT_TYPES]
    if len(transcripts) != 1:
        raise ValueError(f"duplicate canonical URL must have exactly one transcript: {group[0].get('canonical_url')}")
    return transcripts[0]


def _active_source_file(repo_root: Path, source_file: str) -> str:
    path_map, _source_map = _path_maps(repo_root)
    mapped = path_map.get(source_file, {}).get("new_path")
    return mapped if isinstance(mapped, str) and mapped else source_file


def _active_path_record(repo_root: Path, row: dict[str, Any], source_file: str) -> dict[str, Any]:
    return _normalize_path_metadata(repo_root, row, source_file)


def _source_record(row: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    source_file = row["_export_source_file"]
    active = _active_path_record(repo_root, row, source_file)
    return {
        "source_file": _active_source_file(repo_root, source_file),
        "source_line": row["_export_source_line"],
        "source_doc_id": active.get("doc_id"),
        "source_chunk_id": active.get("chunk_id"),
        "canonical_url": active.get("canonical_url"),
        "source_type": active.get("source_type"),
        "raw_path": active.get("raw_path"),
        "raw_retained": active.get("raw_retained"),
        "sha256_fields": {key: active[key] for key in HASH_FIELDS if active.get(key)},
        "source_audio_path": active.get("source_audio_path"),
        "source_audio_sha256": active.get("source_audio_sha256"),
        "metadata_source_path": active.get("metadata_source_path"),
        "metadata_source_file_sha256": active.get("metadata_source_file_sha256"),
        "source_response_sha256": active.get("source_response_sha256"),
    }


def _metadata_overlay(row: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    active = _active_path_record(repo_root, row, row["_export_source_file"])
    return {key: active[key] for key in METADATA_FIELDS if key in active}


def _resolve_raw_path(repo_root: Path, source_file: str, raw_path: str) -> Path | None:
    relative = Path(raw_path)
    if relative.is_absolute():
        return relative if relative.is_file() else None
    root = repo_root.resolve()
    for old_path in _old_path_candidates(root, source_file, raw_path):
        mapped = _mapped_file(root, old_path)
        if mapped is not None:
            try:
                mapped.relative_to(root)
            except ValueError:
                continue
            return mapped
        direct = root / old_path
        if direct.is_file():
            resolved = direct.resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            return resolved
    return None


def _raw_identity_path(repo_root: Path, source_file: str, raw_path: str, resolved: Path) -> str:
    """Keep the verifier's old resolved-path grouping after content-store migration."""
    path_map, source_map = _path_maps(repo_root)
    for candidate in _old_path_candidates(repo_root, source_file, raw_path):
        mapping = path_map.get(candidate) or source_map.get(candidate)
        if mapping is not None:
            if mapping.get("kind") == "symlink" and mapping.get("resolved_target"):
                return str(mapping["resolved_target"])
            return candidate
        direct = repo_root / candidate
        if direct.is_file():
            target = direct.resolve()
            try:
                return target.relative_to(repo_root.resolve()).as_posix()
            except ValueError:
                return str(target)
    try:
        return resolved.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved.resolve())


def _expected_hash(row: dict[str, Any], raw_path: str = "") -> tuple[str | None, str | None]:
    stem = Path(raw_path).stem
    if re.fullmatch(r"[0-9a-fA-F]{64}", stem):
        return stem.lower(), "SHA-256 encoded in content-addressed raw filename"
    for key in ("raw_sha256", "raw_file_sha256", "source_audio_sha256", "audio_sha256", "media_sha256"):
        value = row.get(key)
        if isinstance(value, str) and len(value) == 64:
            return value.lower(), key
    # For LinkedIn MCP rows this field hashes the normalized per-item text, not
    # the shared saved batch JSON, so it must not be compared to the raw file.
    if row.get("source_type") != "linkedin_mcp":
        value = row.get("sha256")
        if isinstance(value, str) and len(value) == 64:
            return value.lower(), "sha256"
    return None, None


def _source_lookup(repo_root: Path, source: dict[str, Any]) -> dict[str, Any]:
    path = repo_root / source["source_file"]
    if not path.is_file():
        raise ValueError(f"missing provenance source JSONL: {source['source_file']}")
    rows = read_jsonl(path)
    index = int(source["source_line"]) - 1
    if index < 0 or index >= len(rows):
        raise ValueError(f"invalid provenance line {source['source_file']}:{source['source_line']}")
    row = rows[index]
    expected_id = source.get("source_doc_id") or source.get("source_chunk_id")
    actual_id = row.get("doc_id") or row.get("chunk_id")
    if (
        expected_id != actual_id
        or row.get("canonical_url") != source.get("canonical_url")
        or (source.get("source_type") and row.get("source_type") != source.get("source_type"))
    ):
        raise ValueError(f"provenance row mismatch: {source['source_file']}:{source['source_line']}")
    return row


def verify_export(repo_root: Path, export_dir: Path, review_dir: Path | None = None) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    export_dir = export_dir if export_dir.is_absolute() else repo_root / export_dir
    export_dir = export_dir.resolve()
    if review_dir is None:
        review_candidate = export_dir / "review"
        if not review_candidate.is_dir():
            review_candidate = export_dir.parent / "review"
        review_dir = review_candidate
    elif not review_dir.is_absolute():
        review_dir = repo_root / review_dir
    review_dir = review_dir.resolve()
    docs = read_jsonl(export_dir / "documents.jsonl")
    chunks = read_jsonl(export_dir / "chunks.jsonl")
    provenance = read_jsonl(export_dir / "provenance.jsonl")
    doc_ids = [row.get("doc_id") for row in docs]
    chunk_ids = [row.get("chunk_id") for row in chunks]
    urls = [row.get("canonical_url") for row in docs]
    if any(not isinstance(url, str) or not url for url in urls) or len(set(urls)) != len(urls):
        raise ValueError("export canonical URLs must be present and unique")
    if any(not value for value in doc_ids) or len(set(doc_ids)) != len(doc_ids):
        raise ValueError("export doc_ids must be present and unique")
    if any(not value for value in chunk_ids) or len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("export chunk_ids must be present and unique")
    doc_id_set = set(doc_ids)
    orphan_ids = sorted({row.get("doc_id") for row in chunks} - doc_id_set)
    if orphan_ids:
        raise ValueError(f"orphan chunk doc_ids: {orphan_ids[:8]}")
    doc_by_id = {row["doc_id"]: row for row in docs}
    for doc in docs:
        if not (doc.get("rights_status") or doc.get("license_status")):
            raise ValueError(f"missing rights/license status on {doc.get('doc_id')}")
    for chunk in chunks:
        if not (chunk.get("rights_status") or chunk.get("license_status")):
            raise ValueError(f"missing rights/license status on {chunk.get('chunk_id')}")
    if len(provenance) != len(docs):
        raise ValueError("provenance record count does not match export document count")
    provenance_by_id = {row.get("doc_id"): row for row in provenance}
    if set(provenance_by_id) != doc_id_set:
        raise ValueError("provenance does not cover export doc_ids exactly")

    source_rows_checked = 0
    raw_files: dict[str, dict[str, Any]] = {}
    no_retained_bytes = 0
    for doc_id, mapping in provenance_by_id.items():
        if mapping.get("canonical_url") != doc_by_id[doc_id].get("canonical_url"):
            raise ValueError(f"provenance URL mismatch for {doc_id}")
        for source in mapping.get("source_records", []):
            original = _source_lookup(repo_root, source)
            source_rows_checked += 1
            raw_path = original.get("raw_path")
            if not raw_path:
                if original.get("raw_retained") is False:
                    no_retained_bytes += 1
                continue
            resolved = _resolve_raw_path(repo_root, source["source_file"], raw_path)
            if resolved is None:
                raise ValueError(f"unresolved raw_path for {doc_id}: {raw_path}")
            expected, basis = _expected_hash(original, raw_path)
            filename = str(resolved)
            identity = _raw_identity_path(repo_root, source["source_file"], str(raw_path), resolved)
            if identity in raw_files:
                if raw_files[identity]["expected"] != expected or raw_files[identity]["filename"] != filename:
                    raise ValueError(f"conflicting raw identity mapping for {identity}")
            raw_files.setdefault(identity, {
                "filename": filename,
                "expected": expected,
                "bases": set(),
                "original_paths": set(),
            })["bases"].add(basis or "no reliable raw-byte hash field")
            raw_files[identity]["original_paths"].update(_old_path_candidates(repo_root, source["source_file"], str(raw_path)))
    raw_hashes_verified = 0
    raw_without_hash = 0
    hash_cache: dict[str, str] = {}
    unverified_raw_paths = []
    raw_resolution_rows = []
    for identity, record in raw_files.items():
        filename = record["filename"]
        expected = record["expected"]
        bases = record["bases"]
        try:
            resolved_relative = Path(filename).resolve().relative_to(repo_root).as_posix()
        except ValueError:
            resolved_relative = filename
        actual = hash_cache.get(filename)
        if actual is None:
            actual = file_sha256(Path(filename))
            hash_cache[filename] = actual
        if expected is None:
            raw_without_hash += 1
            unresolved_hash = {
                "path": resolved_relative,
                "identity": identity,
                "original_paths": sorted(record["original_paths"]),
                "reason": sorted(bases),
            }
            unverified_raw_paths.append(unresolved_hash)
            raw_resolution_rows.append({**unresolved_hash, "sha256": actual, "hash_status": "no_reliable_source_hash"})
        elif expected != actual:
            raise ValueError(f"raw SHA-256 mismatch: {filename}")
        else:
            raw_hashes_verified += 1
            raw_resolution_rows.append({
                "path": resolved_relative,
                "identity": identity,
                "original_paths": sorted(record["original_paths"]),
                "sha256": actual,
                "hash_status": "verified",
            })
    ledger_path = review_dir / "correction-ledger.jsonl"
    correction_overlay = {
        "checked_documents": 0,
        "checked_chunks": 0,
        "matches_source": ledger_path.is_file(),
    }
    if export_dir.name == "dataset" and not ledger_path.is_file():
        raise ValueError(f"current dataset requires a correction ledger: {ledger_path}")
    if ledger_path.exists():
        rules = load_corrections(ledger_path)
        validate_source_occurrence_counts(repo_root, rules)
        for doc_id, mapping in provenance_by_id.items():
            primary_source = mapping.get("source_records", [None])[0]
            if primary_source is None:
                raise ValueError(f"missing primary source provenance for {doc_id}")
            expected_doc = _normalize_path_metadata(
                repo_root, _source_lookup(repo_root, primary_source), primary_source["source_file"]
            )
            apply_corrections(expected_doc, rules)
            actual_doc = _normalize_path_metadata(repo_root, doc_by_id[doc_id], primary_source["source_file"])
            actual_doc.pop("merged_metadata", None)
            if expected_doc != actual_doc:
                raise ValueError(f"export document text/provenance differs from reviewed source for {doc_id}")
            correction_overlay["checked_documents"] += 1
        # Limit correction verification to the normalized source selected by
        # each export document's primary provenance row. Other snapshots can
        # contain an older or newer row with the same doc_id.
        selected_chunk_sources: dict[Path, set[str]] = defaultdict(set)
        source_file_by_doc: dict[str, str] = {}
        for mapping in provenance_by_id.values():
            primary_source = mapping.get("source_records", [None])[0]
            if primary_source is None:
                raise ValueError("missing primary source provenance for chunk verification")
            source_file = repo_root / primary_source["source_file"]
            chunk_file = source_file.parent / "chunks.jsonl"
            source_doc_id = primary_source.get("source_doc_id")
            if not source_doc_id:
                raise ValueError(f"missing primary source_doc_id for {mapping.get('doc_id')}")
            selected_chunk_sources[chunk_file].add(source_doc_id)
            source_file_by_doc[source_doc_id] = primary_source["source_file"]
        source_chunks: dict[str, dict[str, Any]] = {}
        for path, selected_ids in sorted(selected_chunk_sources.items()):
            if not path.is_file():
                raise ValueError(f"missing normalized chunks for selected provenance source: {path}")
            for chunk in read_jsonl(path):
                if chunk.get("doc_id") in selected_ids:
                    source_chunks[chunk.get("chunk_id")] = chunk
        export_by_chunk = {row["chunk_id"]: row for row in chunks}
        if set(source_chunks) != set(export_by_chunk):
            raise ValueError("export chunk IDs differ from selected source chunk IDs")
        for chunk_id, source_chunk in source_chunks.items():
            expected_chunk = copy.deepcopy(source_chunk)
            source_file = source_file_by_doc.get(source_chunk.get("doc_id"))
            if source_file is None:
                raise ValueError(f"missing selected normalized source for chunk {chunk_id}")
            expected_chunk = _normalize_path_metadata(repo_root, expected_chunk, source_file)
            actual_chunk = _normalize_path_metadata(
                repo_root,
                export_by_chunk[chunk_id],
                source_file,
            )
            apply_corrections(expected_chunk, rules)
            if expected_chunk != actual_chunk:
                raise ValueError(f"export chunk text differs from timestamp-scoped source for {chunk_id}")
            correction_overlay["checked_chunks"] += 1
    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "unique_canonical_urls": len(set(urls)),
        "unique_document_ids": len(set(doc_ids)),
        "unique_chunk_ids": len(set(chunk_ids)),
        "orphan_chunks": 0,
        "provenance_records": len(provenance),
        "source_jsonl_rows_checked": source_rows_checked,
        "raw_file_paths_resolved": len(raw_files),
        "unique_raw_byte_objects": len({row["filename"] for row in raw_files.values()}),
        "raw_file_hashes_verified": raw_hashes_verified,
        "raw_file_paths_without_sha256_field": raw_without_hash,
        "raw_file_paths_without_reliable_hash": unverified_raw_paths,
        "raw_file_resolutions": raw_resolution_rows,
        "source_records_without_retained_raw_bytes": no_retained_bytes,
        "rights_status_present": True,
        "correction_overlay": correction_overlay,
    }


def _scan_terms(repo_root: Path) -> dict[str, Any]:
    patterns = {
        "Dreamscape Learn": r"dreamscape\s+learn(?:ed)?",
        "EDge AI": r"ed?ge\s*ai|edgeai",
        "SolarSPELL": r"solar\s*spell",
        "DGX Spark": r"d[jg]x\s+spark",
        "ASUniverse": r"as.?universe",
        "Huddle": r"\bhuddle\b",
        "FutureScape": r"futurescape",
    }
    repo_root = repo_root.resolve()
    # These are historical source-path identifiers resolved through the migration indexes.
    old_transcript_paths = (
        "runs/20260923T173642Z/transcripts/asr.jsonl",
        "runs/20260923T162311Z/transcripts/asr.jsonl",
    )
    transcript_paths = []
    for old_path in old_transcript_paths:
        mapped = _mapped_file(repo_root, old_path)
        candidate = mapped or repo_root / old_path
        if candidate.is_file():
            transcript_paths.append(candidate.resolve())
    reviewed_video_ids = {
        "54MP_dFg_JU", "f-2JyjJPZpE", "05dTrtAVuxA", "ZXA7K4115ko", "vNAh3_-4il8",
        "LQH0eHHEAlU", "DoRXNK-XOJs", "AvgZscNZzYw", "UDKge13zbAg",
    }
    videos = []
    for path in transcript_paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if row.get("video_id") not in reviewed_video_ids:
                continue
            segments = row.get("segments", [])
            text = " ".join(str(segment.get("text", "")) for segment in segments)
            term_hits = {}
            for term, pattern in patterns.items():
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                term_hits[term] = {
                    "count": len(matches),
                    "examples": [text[max(0, hit.start() - 45):hit.end() + 55] for hit in matches[:2]],
                }
            videos.append({
                "video_id": row.get("video_id"),
                "title": row.get("title"),
                "duration_seconds": row.get("duration_seconds"),
                "terms": term_hits,
            })
    return {
        "source_files": [path.relative_to(repo_root).as_posix() for path in transcript_paths if path.exists()],
        "scan_scope": "original ASR segment text in the five duration-deferred videos and four speech drafts (no-speech records excluded)",
        "term_patterns": list(patterns),
        "videos": videos,
    }


def _markdown_report(report: dict[str, Any]) -> str:
    duplicates = report["deduplication"]["dropped_metadata_sources"]
    lines = [
        "# Reviewed Next Lab corpus export",
        "",
        f"Run: `{report['run_id']}`",
        "",
        "## Counts",
        "",
        f"- Input: {report['input']['documents']} documents / {report['input']['chunks']} chunks across {report['input']['unique_canonical_urls']} canonical URLs.",
        f"- Export: {report['output']['documents']} documents / {report['output']['chunks']} chunks / {report['output']['unique_canonical_urls']} unique canonical URLs.",
        f"- Deduplication: {report['deduplication']['dropped_metadata_documents']} metadata-only documents and their {report['deduplication']['dropped_chunks']} chunks removed; transcript records retained with metadata and both provenance records.",
        f"- Review overlay: {report['review']['ledger_entries']} high-confidence corrections; {report['review']['uncertainties']} unresolved items retained verbatim.",
        "",
        "## Verification",
        "",
        f"- {report['verification']['source_jsonl_rows_checked']} source JSONL rows and {report['verification']['provenance_records']} provenance maps checked.",
        f"- {report['verification']['raw_file_paths_resolved']} original raw-path identities resolved to {report['verification']['unique_raw_byte_objects']} unique byte objects; {report['verification']['raw_file_hashes_verified']} path identities passed SHA-256 checks.",
        f"- Timestamp-scoped overlay matched {report['verification']['correction_overlay']['checked_documents']} source documents and {report['verification']['correction_overlay']['checked_chunks']} chunks; no text outside ledger spans was changed.",
        f"- {report['verification']['orphan_chunks']} orphan chunks; export IDs and canonical URLs are unique; every document and chunk retains a rights/license status.",
        "",
        "## Deduplicated metadata sources",
        "",
    ]
    for item in duplicates:
        lines.append(f"- `{item['canonical_url']}` — metadata-only `{item['metadata_doc_id']}` merged into transcript `{item['transcript_doc_id']}`.")
    lines.extend([
        "",
        "## Rights and quality",
        "",
        f"Source rights/license flags are carried through unchanged. Public accessibility does not grant reuse permission; `unreviewed-public`, `unknown`, and `NOASSERTION` remain unresolved. ASR text remains a machine draft. High-confidence replacements are export-only and timestamp-scoped; uncertain phrases remain unchanged in `{report['review'].get('uncertainties_path') or 'the historical review record'}.",
        "",
        "## Files",
        "",
        "- `documents.jsonl`, `chunks.jsonl`, `provenance.jsonl`",
        "- `report.json`, `report.md`",
        f"- `{report['review']['correction_ledger_path']}`, `{report['review']['term_scan_path']}`",
        "",
    ])
    return "\n".join(lines)


def build_export(
    repo_root: Path,
    run_id: str | None = None,
    *,
    expected_duplicate_urls: set[str] | None = None,
    export_dir: Path | None = None,
    review_dir: Path | None = None,
    input_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    if export_dir is None:
        export_dir = repo_root / "dataset" if run_id is None else repo_root / "runs" / run_id / "export"
    elif not export_dir.is_absolute():
        export_dir = repo_root / export_dir
    export_dir = export_dir.resolve()
    if review_dir is None:
        review_dir = export_dir / "review" if export_dir.name == "dataset" else export_dir.parent / "review"
    elif not review_dir.is_absolute():
        review_dir = repo_root / review_dir
    review_dir = review_dir.resolve()
    run_id = run_id or export_dir.name
    docs, chunks = discover_inputs(repo_root, input_dirs)
    corrections = load_corrections(review_dir / "correction-ledger.jsonl")
    source_occurrence_audit = validate_source_occurrence_counts(repo_root, corrections)
    uncertainties = read_jsonl(review_dir / "uncertainties.jsonl") if (review_dir / "uncertainties.jsonl").exists() else []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        url = doc.get("canonical_url")
        if not isinstance(url, str) or not url:
            raise ValueError(f"source document has no canonical URL: {doc.get('doc_id')}")
        groups[url].append(doc)
    duplicate_urls = {url for url, rows in groups.items() if len(rows) > 1}
    if expected_duplicate_urls is not None and duplicate_urls != expected_duplicate_urls:
        raise ValueError(f"unexpected duplicate URL set: {sorted(duplicate_urls)}")

    primary_by_url: dict[str, dict[str, Any]] = {}
    merged_by_url: dict[str, list[dict[str, Any]]] = {}
    dropped_metadata: list[dict[str, Any]] = []
    for url, group in groups.items():
        primary = _preferred_document(group)
        merged = [row for row in group if row is not primary]
        if any(row.get("source_type") != METADATA_TYPE for row in merged):
            raise ValueError(f"refusing to merge non-metadata duplicate for {url}")
        primary_by_url[url] = primary
        merged_by_url[url] = merged
        dropped_metadata.extend({
            "canonical_url": url,
            "metadata_doc_id": row.get("doc_id"),
            "transcript_doc_id": primary.get("doc_id"),
        } for row in merged)
    if expected_duplicate_urls is not None and len(dropped_metadata) != len(expected_duplicate_urls):
        raise ValueError("expected one metadata-only document per duplicate canonical URL")

    export_docs: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    applied_doc_counts: Counter[str] = Counter()
    for url in sorted(primary_by_url):
        source = primary_by_url[url]
        doc = {key: copy.deepcopy(value) for key, value in source.items() if not key.startswith("_export_")}
        doc = _active_path_record(repo_root, doc, source["_export_source_file"])
        merged = merged_by_url[url]
        if merged:
            doc["merged_metadata"] = [_metadata_overlay(row, repo_root) for row in merged]
        applied_ids, counts = apply_corrections(doc, corrections)
        applied_doc_counts.update(counts)
        export_docs.append(doc)
        provenance_rows.append({
            "doc_id": doc["doc_id"],
            "canonical_url": url,
            "source_records": [_source_record(row, repo_root) for row in [source, *merged]],
            "review_correction_ids": applied_ids,
        })
    selected_ids = {row.get("doc_id") for row in primary_by_url.values()}
    export_chunks = []
    applied_chunk_counts: Counter[str] = Counter()
    for source in chunks:
        if source.get("doc_id") not in selected_ids:
            continue
        chunk = {key: copy.deepcopy(value) for key, value in source.items() if not key.startswith("_export_")}
        chunk = _active_path_record(repo_root, chunk, source["_export_source_file"])
        applied_ids, counts = apply_corrections(chunk, corrections)
        applied_chunk_counts.update(counts)
        export_chunks.append(chunk)
    export_chunks.sort(key=lambda row: (row.get("canonical_url", ""), row.get("chunk_index", 0), row.get("chunk_id", "")))

    input_urls = {doc.get("canonical_url") for doc in docs}
    input_counts = {"documents": len(docs), "chunks": len(chunks), "unique_canonical_urls": len(input_urls)}
    dropped_chunks = len(chunks) - len(export_chunks)
    if expected_duplicate_urls is not None and dropped_chunks != len(expected_duplicate_urls):
        raise ValueError(f"expected one chunk per duplicate metadata doc, dropped {dropped_chunks}")
    export_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(export_dir / "documents.jsonl", export_docs)
    write_jsonl(export_dir / "chunks.jsonl", export_chunks)
    write_jsonl(export_dir / "provenance.jsonl", provenance_rows)

    term_scan = _scan_terms(repo_root)
    term_scan_path = review_dir / "term-scan.json"
    term_scan_path.write_text(json.dumps(term_scan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    verification = verify_export(repo_root, export_dir, review_dir)
    source_types = Counter(str(doc.get("source_type", "unknown")) for doc in export_docs)
    rights = Counter(str(doc.get("rights_status", doc.get("license_status", "missing"))) for doc in export_docs)
    applied_ids = sorted({value for row in export_docs for value in row.get("review_correction_ids", [])})
    ledger_ids = {row["correction_id"] for row in corrections}
    if set(applied_ids) != ledger_ids:
        raise ValueError(f"correction ledger/export mismatch; unapplied={sorted(ledger_ids - set(applied_ids))}")
    report: dict[str, Any] = {
        "run_id": run_id,
        "scope": "recursive export of normalized corpus, timestamp-scoped transcript review overlay, and metadata/transcript deduplication",
        "input": input_counts,
        "deduplication": {
            "duplicate_url_pairs": len(dropped_metadata),
            "dropped_metadata_documents": len(dropped_metadata),
            "dropped_chunks": dropped_chunks,
            "dropped_metadata_sources": sorted(dropped_metadata, key=lambda row: row["canonical_url"]),
            "expected_duplicate_urls_verified": expected_duplicate_urls is None or duplicate_urls == expected_duplicate_urls,
        },
        "output": {
            "documents": len(export_docs),
            "chunks": len(export_chunks),
            "unique_canonical_urls": len({doc.get("canonical_url") for doc in export_docs}),
            "source_type_counts": dict(sorted(source_types.items())),
        },
        "review": {
            "ledger_entries": len(corrections),
            "applied_correction_ids": applied_ids,
            "raw_asr_occurrence_audit": source_occurrence_audit,
            "uncertainties": len(uncertainties),
            "raw_asr_modified": False,
            "term_scan_path": term_scan_path.relative_to(repo_root).as_posix(),
            "correction_ledger_path": (review_dir / "correction-ledger.jsonl").relative_to(repo_root).as_posix(),
            "uncertainties_path": (review_dir / "uncertainties.jsonl").relative_to(repo_root).as_posix() if (review_dir / "uncertainties.jsonl").is_file() else None,
            "document_text_occurrences_by_correction": dict(sorted(applied_doc_counts.items())),
            "chunk_text_occurrences_by_correction": dict(sorted(applied_chunk_counts.items())),
        },
        "rights_status_counts": dict(sorted(rights.items())),
        "rights_note": "Public accessibility does not grant reuse permission. Source license and rights flags are carried without blanket reuse claims.",
        "verification": verification,
    }
    artifact_paths = [
        export_dir / "documents.jsonl",
        export_dir / "chunks.jsonl",
        export_dir / "provenance.jsonl",
        review_dir / "correction-ledger.jsonl",
        term_scan_path,
        review_dir / "uncertainties.jsonl",
    ]
    report["artifact_hashes"] = {}
    for path in artifact_paths:
        path = path.resolve()
        if path.exists():
            report["artifact_hashes"][path.relative_to(repo_root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    (export_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (export_dir / "report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", help="legacy run ID; defaults to the active dataset")
    parser.add_argument("--export-dir", type=Path, help="export directory to verify or build; defaults to dataset/")
    parser.add_argument("--review-dir", type=Path, help="review directory containing the correction ledger")
    parser.add_argument("--input-dir", type=Path, action="append", help="run or normalized directory to include; may be repeated")
    parser.add_argument("--expected-duplicate-url", action="append", help="expected duplicate URL; may be repeated")
    parser.add_argument("--verify-only", action="store_true", help="verify an existing export without rebuilding it")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    if args.export_dir is not None:
        export_dir = args.export_dir if args.export_dir.is_absolute() else root / args.export_dir
    elif args.run_id:
        export_dir = root / "runs" / args.run_id / "export"
    else:
        export_dir = root / "dataset"
    review_dir = args.review_dir
    if args.verify_only:
        print(json.dumps(verify_export(root, export_dir, review_dir), sort_keys=True))
        return 0
    expected_urls = set(args.expected_duplicate_url) if args.expected_duplicate_url else None
    result = build_export(
        root,
        args.run_id,
        expected_duplicate_urls=expected_urls,
        export_dir=export_dir,
        review_dir=review_dir,
        input_dirs=args.input_dir,
    )
    print(json.dumps({"output": result["output"], "verification": result["verification"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
