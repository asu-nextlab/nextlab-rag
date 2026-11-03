#!/usr/bin/env python3
"""Compare retrieval variants on the Next Lab eval set.

Modes: baseline, rerank, expand, rerank_expand. The dense query is always the
original question. Expansion only changes the BM25 query. Reranking uses a
cross-encoder over the top 20 deduplicated chunks and keeps the top 10.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastembed.rerank.cross_encoder import TextCrossEncoder
from nextlab_rag.indexing import EMBEDDING_MODEL, _load_embedder, search_corpus

QUESTIONS = ROOT / "eval" / "questions.jsonl"
OUT_DIR = ROOT / "eval" / "experiments"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

EXPANSIONS = {
    "asuniverse": ["asu universe", "metaverse"],
    "asu universe": ["asuniverse"],
    "eoss": ["educational outreach and student services"],
    "mcbi": ["maricopa county broadband initiative"],
    "az-1": ["az1", "as one"],
    "az1": ["az-1", "as one"],
    "edge ai": ["edgeai", "offline ai"],
    "edgeai": ["edge ai", "offline ai"],
    "tamac": ["technical artistry modeling animation concept"],
    "gsc": ["games and spatial computing"],
    "uxr": ["user experience and research"],
    "aaii": ["applied ai infrastructure"],
    "qs impact": ["qs impact"],
    "westmarc": ["westmarc"],
    "solarspell": ["solar spell"],
    "n50": ["n50 ai for all"],
    "dgx spark": ["nvidia dgx spark"],
    "engagevr": ["engage vr"],
    "cesium": ["cesium ion"],
    "arcgis": ["esri"],
    "newswell": ["news well"],
    "newspack": ["newspack"],
}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def expand(question: str) -> str:
    lowered = question.casefold()
    additions: list[str] = []
    for key, values in EXPANSIONS.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", lowered):
            additions.extend(values)
    if not additions:
        return question
    return question + " " + " ".join(dict.fromkeys(additions))


def dedupe(hits: list[dict], limit: int) -> list[dict]:
    seen: set[str] = set()
    kept: list[dict] = []
    for hit in hits:
        chunk_id = hit["chunk_id"]
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        kept.append(hit)
        if len(kept) >= limit:
            break
    return kept


def rerank(rows: list[dict], question: str, model: TextCrossEncoder, limit: int) -> list[dict]:
    if not rows:
        return []
    texts = [row.get("passage_text") or row["chunk"].get("text", "") for row in rows]
    scores = list(model.rerank(question, texts, batch_size=16))
    ranked = sorted(zip(rows, scores), key=lambda item: item[1], reverse=True)
    return [row for row, _ in ranked[:limit]]


def score(rows: list[dict], gold: list[str]) -> dict:
    hit_ids = [row["chunk_id"] for row in rows]
    gold_set = set(gold)
    first = next((rank for rank, chunk_id in enumerate(hit_ids, 1) if chunk_id in gold_set), None)
    return {
        "hits": hit_ids,
        "first_rank": first,
        "hit5": any(chunk_id in gold_set for chunk_id in hit_ids[:5]),
        "hit10": any(chunk_id in gold_set for chunk_id in hit_ids[:10]),
        "coverage10": len(gold_set & set(hit_ids[:10])) / len(gold_set) if gold_set else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", type=Path)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--modes", default="baseline,rerank,expand,rerank_expand")
    args = parser.parse_args()

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    allowed = {"baseline", "rerank", "expand", "rerank_expand"}
    if not modes or set(modes) - allowed:
        raise SystemExit(f"modes must be a subset of {sorted(allowed)}")
    need_expand = bool({"expand", "rerank_expand"} & set(modes))
    need_rerank = bool({"rerank", "rerank_expand"} & set(modes))

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    allowed = {"baseline", "rerank", "expand", "rerank_expand"}
    if not modes or set(modes) - allowed:
        raise SystemExit(f"modes must be a subset of {sorted(allowed)}")
    need_expand = bool({"expand", "rerank_expand"} & set(modes))
    need_rerank = bool({"rerank", "rerank_expand"} & set(modes))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = load(QUESTIONS)
    embedder = _load_embedder(args.embedding_model, ROOT / ".rag" / "cache")
    reranker = (
        TextCrossEncoder(
            model_name=RERANK_MODEL,
            cache_dir=str(ROOT / ".rag" / "cache"),
            threads=1,
            providers=["CPUExecutionProvider"],
        )
        if need_rerank
        else None
    )

    results: dict[str, list[dict]] = {mode: [] for mode in modes}

    for number, row in enumerate(questions, 1):
        gold = row.get("gold_chunk_ids") or []
        if not gold:
            for mode in modes:
                results[mode].append(
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

        normal = dedupe(
            search_corpus(
                row["question"],
                ROOT,
                rag_dir=args.rag_dir,
                model_name=args.embedding_model,
                embedder=embedder,
                limit=args.candidates,
            ),
            args.candidates,
        )
        expanded_query = expand(row["question"])
        expanded: list[dict] = []
        if need_expand:
            expanded = dedupe(
                search_corpus(
                    row["question"],
                    ROOT,
                    rag_dir=args.rag_dir,
                    model_name=args.embedding_model,
                    embedder=embedder,
                    limit=args.candidates,
                    text_query=expanded_query,
                ),
                args.candidates,
            )

        variants: dict[str, list[dict]] = {}
        if "baseline" in modes:
            variants["baseline"] = normal[: args.limit]
        if "rerank" in modes:
            variants["rerank"] = rerank(normal, row["question"], reranker, args.limit)
        if "expand" in modes:
            variants["expand"] = expanded[: args.limit]
        if "rerank_expand" in modes:
            variants["rerank_expand"] = rerank(expanded, row["question"], reranker, args.limit)
        for mode, rows in variants.items():
            results[mode].append(
                {
                    "qid": row["qid"],
                    "question_type": row["question_type"],
                    "split": row["split"],
                    "gold": gold,
                    **score(rows, gold),
                }
            )
        if number % 10 == 0:
            print(f"  {number}/{len(questions)}", flush=True)

    summary: dict[str, dict] = {}
    for mode, rows in results.items():
        path = OUT_DIR / f"{args.name}-{mode}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for item in rows:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")

        answerable = [item for item in rows if item["gold"]]
        by_type: dict[str, list[dict]] = defaultdict(list)
        for item in answerable:
            by_type[item["question_type"]].append(item)
        summary[mode] = {
            "n": len(answerable),
            "hit5": mean(1.0 if item["hit5"] else 0.0 for item in answerable),
            "hit10": mean(1.0 if item["hit10"] else 0.0 for item in answerable),
            "mrr": mean(1.0 / item["first_rank"] if item["first_rank"] else 0.0 for item in answerable),
            "coverage10": mean(item["coverage10"] for item in answerable),
            "by_type": {
                kind: {
                    "n": len(items),
                    "hit5": mean(1.0 if item["hit5"] else 0.0 for item in items),
                    "hit10": mean(1.0 if item["hit10"] else 0.0 for item in items),
                    "mrr": mean(1.0 / item["first_rank"] if item["first_rank"] else 0.0 for item in items),
                    "both10": mean(1.0 if item["coverage10"] == 1.0 else 0.0 for item in items),
                }
                for kind, items in sorted(by_type.items())
            },
        }

    (OUT_DIR / f"{args.name}-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        f"# Retrieval experiments: {args.name}",
        "",
        f"Embedding: `{args.embedding_model}`",
        f"Reranker: `{RERANK_MODEL}`",
        "",
        "| Mode | n | hit@5 | hit@10 | MRR | coverage@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, values in summary.items():
        lines.append(
            f"| {mode} | {values['n']} | {values['hit5']:.1%} | {values['hit10']:.1%} | "
            f"{values['mrr']:.3f} | {values['coverage10']:.1%} |"
        )
    lines.extend(["", "## Multi-hop both chunks at k=10", "", "| Mode | n | both@10 |", "| --- | ---: | ---: |"])
    for mode, values in summary.items():
        multi = values["by_type"].get("multi_hop")
        if multi:
            lines.append(f"| {mode} | {multi['n']} | {multi['both10']:.1%} |")
    lines.append("")
    (OUT_DIR / f"{args.name}-summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
