#!/usr/bin/env python3
"""Measure retrieval hit@k for the evaluation set.

Run with the project environment:
  .venv-pdf/bin/python scripts/eval_retrieval.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nextlab_rag.indexing import EMBEDDING_MODEL, _load_embedder, search_corpus

QUESTIONS = ROOT / "eval" / "questions.jsonl"
REPORT = ROOT / "eval" / "retrieval_report.jsonl"
SUMMARY = ROOT / "eval" / "retrieval_summary.md"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    rows = load(QUESTIONS)
    embedder = _load_embedder(EMBEDDING_MODEL, ROOT / ".rag" / "cache")

    results: list[dict] = []
    for row in rows:
        gold = row.get("gold_chunk_ids") or []
        if not gold:
            results.append(
                {
                    "qid": row["qid"],
                    "question_type": row["question_type"],
                    "split": row["split"],
                    "gold": [],
                    "hits": [],
                    "first_rank": None,
                    "hit5": None,
                    "hit10": None,
                    "coverage10": None,
                }
            )
            continue
        hits = search_corpus(
            row["question"],
            ROOT,
            embedder=embedder,
            limit=args.top_k,
        )
        hit_ids = [hit["chunk_id"] for hit in hits]
        gold_set = set(gold)
        first_rank = next((rank for rank, chunk_id in enumerate(hit_ids, 1) if chunk_id in gold_set), None)
        found10 = len(gold_set & set(hit_ids))
        results.append(
            {
                "qid": row["qid"],
                "question_type": row["question_type"],
                "split": row["split"],
                "gold": gold,
                "hits": hit_ids,
                "first_rank": first_rank,
                "hit5": any(chunk_id in gold_set for chunk_id in hit_ids[:5]),
                "hit10": any(chunk_id in gold_set for chunk_id in hit_ids[:10]),
                "coverage10": found10 / len(gold_set),
            }
        )

    with REPORT.open("w", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    answerable = [row for row in results if row["gold"]]
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in answerable:
        by_type[row["question_type"]].append(row)

    lines = [
        "# Retrieval hit@k",
        "",
        f"Questions: {len(results)} total, {len(answerable)} with gold chunks.",
        f"Embedding model: `{EMBEDDING_MODEL}`.",
        "",
        "| Type | n | hit@5 | hit@10 | mean gold coverage@10 | MRR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind in sorted(by_type):
        rows_for_type = by_type[kind]
        hit5 = mean(1.0 if row["hit5"] else 0.0 for row in rows_for_type)
        hit10 = mean(1.0 if row["hit10"] else 0.0 for row in rows_for_type)
        coverage = mean(row["coverage10"] for row in rows_for_type)
        mrr = mean(1.0 / row["first_rank"] if row["first_rank"] else 0.0 for row in rows_for_type)
        lines.append(
            f"| {kind} | {len(rows_for_type)} | {hit5:.1%} | {hit10:.1%} | {coverage:.1%} | {mrr:.2f} |"
        )

    overall_hit5 = mean(1.0 if row["hit5"] else 0.0 for row in answerable)
    overall_hit10 = mean(1.0 if row["hit10"] else 0.0 for row in answerable)
    overall_mrr = mean(1.0 / row["first_rank"] if row["first_rank"] else 0.0 for row in answerable)
    lines.extend(
        [
            f"| **all** | **{len(answerable)}** | **{overall_hit5:.1%}** | **{overall_hit10:.1%}** | "
            f"**{mean(row['coverage10'] for row in answerable):.1%}** | **{overall_mrr:.2f}** |",
            "",
        ]
    )

    multi = by_type.get("multi_hop", [])
    if multi:
        both = mean(1.0 if row["coverage10"] == 1.0 else 0.0 for row in multi)
        lines.append(f"Multi-hop questions with every gold chunk retrieved at k=10: {both:.1%} ({len(multi)} items).")
        lines.append("")

    misses = [row for row in answerable if not row["hit10"]]
    lines.append(f"Misses at k=10: {len(misses)}.")
    if misses:
        lines.append("")
        for row in misses:
            lines.append(f"- {row['qid']} ({row['question_type']}): gold {', '.join(row['gold'])}")
    lines.append("")

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
