from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .answering import ask
from .errors import RagError
from .indexing import EMBEDDING_MODEL, index_corpus, search_corpus


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _print_search(hits: list[dict[str, Any]]) -> None:
    if not hits:
        print("No retrieved passages.")
        return
    for number, hit in enumerate(hits, 1):
        document = hit["document"]
        chunk = hit["chunk"]
        print(f"[S{number}] {document.get('title', chunk.get('title', 'Untitled source'))}")
        print(f"  {document.get('canonical_url', chunk.get('canonical_url', ''))}")
        print(f"  chunk={hit['chunk_id']} section={chunk.get('section_heading') or '(none)'}")
        print(f"  rights={chunk.get('rights_status', document.get('rights_status'))}; "
              f"license={chunk.get('license_status', document.get('license_status'))}")
        for key in (
            "capture_completeness",
            "completeness_note",
            "source_link_summary",
            "reconciliation_metadata",
        ):
            value = document.get(key)
            if value:
                print(f"  {key}={json.dumps(value, ensure_ascii=False)}")
        for key in ("review_uncertainties", "review_correction_ids", "review_corrections"):
            value = hit.get(key)
            if value:
                print(f"  {key}={json.dumps(value, ensure_ascii=False)}")
        print(f"  {hit['passage_text']}")


def _print_answer(result: dict[str, Any]) -> None:
    print(result["answer"])
    for source in result["sources"]:
        print(f"\n[{source['id']}] {source['title']}")
        print(f"  {source.get('canonical_url') or ''}")
        print(f"  chunk={source.get('chunk_id')} section={source.get('section_heading') or '(none)'}")
        print(f"  rights={source.get('rights_status')}; license={source.get('license_status')}")
        for key in (
            "capture_completeness",
            "completeness_note",
            "source_link_summary",
            "reconciliation_metadata",
            "review_uncertainties",
            "review_correction_ids",
            "review_corrections",
        ):
            value = source.get(key)
            if value:
                print(f"  {key}={json.dumps(value, ensure_ascii=False)}")
        print(f"  {source['passage_text']}")
    if result.get("citation_validation", {}).get("valid") is False:
        print("\nCitation check failed. The answer was replaced with an abstention.", file=sys.stderr)
    print("\nCitation check is structural; it does not prove semantic support.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m nextlab_rag")
    commands = parser.add_subparsers(dest="command", required=True)
    index_parser = commands.add_parser("index", help="build or resume the local Qdrant index")
    search_parser = commands.add_parser("search", help="retrieve hybrid search excerpts")
    ask_parser = commands.add_parser("ask", help="answer from retrieved excerpts using a configured endpoint")
    for command in (index_parser, search_parser, ask_parser):
        command.add_argument("--repo-root", type=Path, default=Path.cwd())
        command.add_argument("--rag-dir", type=Path)
        command.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    index_parser.add_argument("--batch-size", type=int, default=32)
    for command in (search_parser, ask_parser):
        command.add_argument("question", nargs="+", help="natural-language question")
        command.add_argument("--top-k", type=int, default=5)
        command.add_argument("--json", action="store_true", help="write structured JSON")
    ask_parser.add_argument("--base-url", help="chat-completions API base URL, or NEXTLAB_RAG_BASE_URL")
    ask_parser.add_argument("--model", help="chat model ID, or NEXTLAB_RAG_MODEL")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    question = " ".join(getattr(args, "question", []))
    try:
        if args.command == "index":
            manifest = index_corpus(
                args.repo_root,
                rag_dir=args.rag_dir,
                model_name=args.embedding_model,
                batch_size=args.batch_size,
            )
            print(_json(manifest))
            return 0
        if args.command == "search":
            hits = search_corpus(
                question,
                args.repo_root,
                rag_dir=args.rag_dir,
                model_name=args.embedding_model,
                limit=args.top_k,
            )
            if args.json:
                print(_json(hits))
            else:
                _print_search(hits)
            return 0
        result = ask(
            question,
            args.repo_root,
            rag_dir=args.rag_dir,
            model_name=args.embedding_model,
            limit=args.top_k,
            chat_base_url=args.base_url,
            chat_model=args.model,
        )
        if args.json:
            print(_json(result))
        else:
            _print_answer(result)
        return 0
    except (RagError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
