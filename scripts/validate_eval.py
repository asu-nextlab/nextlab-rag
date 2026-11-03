#!/usr/bin/env python3
"""Validate the Next Lab evaluation set.

Checks schema, quotas, gold chunk consistency, quote support, leakage, and
duplicates. It does not judge whether an answer is semantically correct.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "eval" / "candidates.jsonl"
CHUNKS = ROOT / "dataset" / "chunks.jsonl"

TYPES = {
    "single_hop",
    "multi_hop",
    "transcript",
    "social",
    "timeline",
    "unanswerable",
    "conflict",
    "incomplete_capture",
}

PROFILES: dict[str, dict[str, int]] = {
    "a": {"single_hop": 30, "multi_hop": 20},
    "b": {
        "transcript": 10,
        "social": 5,
        "timeline": 15,
        "unanswerable": 10,
        "conflict": 5,
        "incomplete_capture": 5,
    },
    "final": {
        "single_hop": 30,
        "multi_hop": 24,
        "transcript": 10,
        "social": 5,
        "timeline": 16,
        "unanswerable": 10,
        "incomplete_capture": 5,
    },
    "none": {},
}

QID_RANGES = {"a": (1, 50), "b": (51, 100), "final": (1, 100), "none": None}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise SystemExit(f"{path}:{number}: expected a JSON object")
            rows.append(value)
    return rows


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
    )
    return re.sub(r"\s+", " ", text).strip().casefold()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize(text))


def jaccard(left: str, right: str) -> float:
    a, b = set(words(left)), set(words(right))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="final")
    parser.add_argument("questions", type=Path)
    args = parser.parse_args()

    candidates = {row["chunk_id"]: row for row in load_jsonl(CANDIDATES)}
    full_chunks = {row["chunk_id"]: row for row in load_jsonl(CHUNKS)}
    questions = load_jsonl(args.questions)

    errors: list[str] = []
    warnings: list[str] = []

    def error(qid: str, message: str) -> None:
        errors.append(f"{qid}: {message}")

    def warn(qid: str, message: str) -> None:
        warnings.append(f"{qid}: {message}")

    seen_qids: set[str] = set()
    counts: Counter[str] = Counter()

    for index, row in enumerate(questions, 1):
        qid = str(row.get("qid") or f"row-{index}")
        tag = qid
        if qid in seen_qids:
            error(tag, "duplicate qid")
        seen_qids.add(qid)

        if not re.fullmatch(r"q\d{3}", qid):
            error(tag, "qid must look like q001")

        required = [
            "qid",
            "question",
            "question_type",
            "answer",
            "answer_aliases",
            "gold_chunk_ids",
            "gold_doc_ids",
            "support_quotes",
            "source_type",
            "difficulty",
            "answerable",
            "absence_check",
            "notes",
        ]
        for key in required:
            if key not in row:
                error(tag, f"missing field {key}")

        question_type = row.get("question_type")
        if question_type not in TYPES:
            error(tag, f"unknown question_type {question_type!r}")
            continue
        counts[question_type] += 1

        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            error(tag, "question must be a non-empty string")
        else:
            if not question.rstrip().endswith("?"):
                error(tag, "question must end with ?")
            length = len(question.split())
            if length < 8 or length > 30:
                warn(tag, f"question length {length} words, target 8-30")

        answer = row.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            error(tag, "answer must be a non-empty string")

        aliases = row.get("answer_aliases")
        if not isinstance(aliases, list) or not all(isinstance(item, str) for item in aliases):
            error(tag, "answer_aliases must be a list of strings")

        gold_ids = row.get("gold_chunk_ids")
        if not isinstance(gold_ids, list) or not all(isinstance(item, str) for item in gold_ids):
            error(tag, "gold_chunk_ids must be a list of strings")
            gold_ids = []
        gold_ids = list(dict.fromkeys(gold_ids))

        gold_docs = row.get("gold_doc_ids")
        if not isinstance(gold_docs, list) or not all(isinstance(item, str) for item in gold_docs):
            error(tag, "gold_doc_ids must be a list of strings")
            gold_docs = []

        for chunk_id in gold_ids:
            if chunk_id not in candidates:
                error(tag, f"gold chunk not in candidate pool: {chunk_id}")
        expected_docs = sorted({candidates[c]["doc_id"] for c in gold_ids if c in candidates})
        if sorted(gold_docs) != expected_docs:
            error(tag, f"gold_doc_ids {sorted(gold_docs)} do not match chunk docs {expected_docs}")

        source_type = row.get("source_type")
        if gold_ids and gold_ids[0] in candidates:
            expected_source = candidates[gold_ids[0]]["source_type"]
            if source_type != expected_source:
                warn(tag, f"source_type {source_type!r} differs from candidate {expected_source!r}")

        if row.get("difficulty") not in {"easy", "medium", "hard"}:
            error(tag, "difficulty must be easy, medium, or hard")

        answerable = row.get("answerable")
        if not isinstance(answerable, bool):
            error(tag, "answerable must be a boolean")

        quotes = row.get("support_quotes")
        if not isinstance(quotes, list):
            error(tag, "support_quotes must be a list")
            quotes = []
        quote_chunks: set[str] = set()
        for quote_row in quotes:
            if not isinstance(quote_row, dict):
                error(tag, "support_quotes entries must be objects")
                continue
            chunk_id = quote_row.get("chunk_id")
            quote = quote_row.get("quote")
            if not isinstance(chunk_id, str) or chunk_id not in candidates:
                error(tag, f"quote chunk not in candidate pool: {chunk_id}")
                continue
            quote_chunks.add(chunk_id)
            if chunk_id not in gold_ids:
                error(tag, f"quote chunk {chunk_id} not in gold_chunk_ids")
            if not isinstance(quote, str) or len(quote.split()) < 4:
                error(tag, f"quote for {chunk_id} is too short")
                continue
            haystack = full_chunks[chunk_id]["text"] if chunk_id in full_chunks else candidates[chunk_id]["text"]
            if normalize(quote) not in normalize(haystack):
                error(tag, f"quote is not an exact substring of {chunk_id}")

        if question_type == "unanswerable":
            if answerable is not False:
                error(tag, "unanswerable questions must set answerable to false")
            if gold_ids or quotes:
                error(tag, "unanswerable questions must have no gold chunks or quotes")
            absence = row.get("absence_check")
            if not isinstance(absence, dict):
                error(tag, "unanswerable questions need an absence_check object")
            else:
                if not isinstance(absence.get("query"), str) or not absence["query"].strip():
                    error(tag, "absence_check.query must be a non-empty string")
                if absence.get("hits") != 0:
                    error(tag, "absence_check.hits must be 0")
        elif question_type == "incomplete_capture":
            if answerable is not False:
                error(tag, "incomplete_capture questions must set answerable to false")
            if not gold_ids or not quotes:
                error(tag, "incomplete_capture needs a gold chunk and a quote")
            if not re.search(r"truncat|incomplete|does not establish", (answer or ""), re.IGNORECASE):
                error(tag, "answer must state that the capture is truncated or does not establish the fact")
        else:
            if answerable is not True:
                error(tag, "answerable questions must set answerable to true")
            if not gold_ids:
                error(tag, "answerable questions need at least one gold chunk")
            if not quotes:
                error(tag, "answerable questions need at least one quote")

        if question_type == "multi_hop":
            if len(gold_ids) < 2:
                error(tag, "multi_hop needs at least two gold chunks")
            if len(set(gold_docs)) < 2:
                error(tag, "multi_hop needs chunks from at least two documents")
            if len(quote_chunks) < 2:
                error(tag, "multi_hop needs quotes from both documents")

        if question_type == "conflict":
            if len(gold_ids) < 2 or len(set(gold_docs)) < 2:
                error(tag, "conflict needs two chunks from two documents")
            if len(quote_chunks) < 2:
                error(tag, "conflict needs two quotes")

        if question_type == "transcript" and gold_ids and gold_ids[0] in candidates:
            group = candidates[gold_ids[0]]["group"]
            if group != "transcripts":
                warn(tag, f"transcript question uses group {group!r}")

        if question_type == "social" and gold_ids and gold_ids[0] in candidates:
            group = candidates[gold_ids[0]]["group"]
            if group != "social":
                warn(tag, f"social question uses group {group!r}")

        # Leakage checks.
        answer_words = words(answer or "")
        question_norm = normalize(question or "")
        if 0 < len(answer_words) <= 6 and " ".join(answer_words) in question_norm:
            warn(tag, "question may contain the answer verbatim")
        for chunk_id in gold_ids[:1]:
            title = candidates.get(chunk_id, {}).get("title") or ""
            title_words = words(title)
            if len(title_words) >= 3 and " ".join(title_words) in question_norm:
                warn(tag, "question repeats the source title")

    # Duplicate questions.
    for left in range(len(questions)):
        for right in range(left + 1, len(questions)):
            a = questions[left].get("question", "")
            b = questions[right].get("question", "")
            if jaccard(a, b) >= 0.85:
                error(
                    str(questions[right].get("qid")),
                    f"near-duplicate of {questions[left].get('qid')}: {jaccard(a, b):.2f}",
                )

    # Quotas and QID range.
    expected = PROFILES[args.profile]
    for question_type, count in expected.items():
        if counts[question_type] != count:
            errors.append(f"quota: {question_type} has {counts[question_type]}, expected {count}")
    unexpected = set(counts) - set(expected)
    for question_type in sorted(unexpected):
        if args.profile == "none":
            continue
        errors.append(f"quota: unexpected type {question_type} ({counts[question_type]})")

    low, high = QID_RANGES[args.profile] if QID_RANGES[args.profile] is not None else (None, None)
    if low is not None and high is not None:
        expected_qids = {f"q{number:03d}" for number in range(low, high + 1)}
        missing = expected_qids - seen_qids
        extra = seen_qids - expected_qids
        if missing:
            errors.append(f"qids missing: {', '.join(sorted(missing))}")
        if extra:
            errors.append(f"qids outside profile: {', '.join(sorted(extra))}")

    print(f"questions: {len(questions)}")
    print("counts:", ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    if warnings:
        print(f"\nwarnings ({len(warnings)}):")
        for message in warnings:
            print(f"  {message}")
    if errors:
        print(f"\nerrors ({len(errors)}):")
        for message in errors:
            print(f"  {message}")
        sys.exit(1)
    print("\nok")


if __name__ == "__main__":
    main()
