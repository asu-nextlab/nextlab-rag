"""Chunk normalized documents with the shared collector text splitter."""

from __future__ import annotations

from typing import Any, Mapping

from .collector import chunk_text


def make_document_chunks(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for section_index, section in enumerate(document.get("sections", [])):
        for chunk_index, text in enumerate(chunk_text(str(section.get("text", "")))):
            chunks.append(
                {
                    "chunk_id": f"{document['doc_id']}-{section_index:03d}-{chunk_index:03d}",
                    "doc_id": document["doc_id"],
                    "canonical_url": document["canonical_url"],
                    "title": document.get("title", ""),
                    "section_heading": section.get("heading", ""),
                    "section_level": section.get("level", 0),
                    "chunk_index": chunk_index,
                    "text": text,
                    "source_type": document.get("source_type", ""),
                    "rights_status": document.get("rights_status", "unreviewed-public"),
                }
            )
    return chunks
