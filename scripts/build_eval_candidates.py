#!/usr/bin/env python3
"""Build a stratified candidate pool for the Next Lab evaluation set.

The pool is deliberately small enough for one strong model to read in full.
It covers topic groups rather than a uniform chunk sample, because transcripts
are 40 percent of the corpus and would otherwise dominate.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
OUT = ROOT / "eval"

# Topic groups. Each group can supply single-hop, multi-hop, and timeline
# questions. Doc IDs were checked against dataset/documents.jsonl.
GROUPS: dict[str, list[str]] = {
    "asuniverse": [
        "41b3b04f85922613bf1a",  # Welcome to the ASUniverse
        "1962089e9b51a3144274",  # ASU News: ASUniverse launch
        "32e3685218d3de18e66b",  # ASUniverse immersive campus
    ],
    "zoom": [
        "60c4939185dbe6e52807",  # Inaugural Zoom Fellowship
        "0e334a673a25df592c79",  # Zoom Innovation Lab article
        "5e3530a5673db04f73ed",  # One year later
        "eb88a363bedf7e841214",  # Zoom Innovation Lab 1yr
    ],
    "surprise": [
        "e50ec06793148d622d31",  # City of Surprise Digital Twin
        "9c0f44e99d6c04c5aa22",  # Surprise slides
    ],
    "maricopa": [
        "ff4d75fa212d76321473",  # Maricopa County Digital Twins
        "1505a6fe490d79b488a9",  # Maricopa County Digital Storytelling
    ],
    "water": [
        "29badab0cc60920eb15a",  # Arizona Water Metaverse
        "79dcc550393afd8ba03b",  # ASU water scarcity story
    ],
    "alder_mayo": ["711083be2588f5da20ae"],
    "acacia": [
        "f19093351a2ca724b886",  # Acacia
        "2946648c167919c621ca",  # Acacia final deck
    ],
    "arolla": ["2b5cbee3a1ab73780d18"],
    "betel": ["0c38b485c723f39e7f46"],
    "global_launch": ["d5bcfd940d05f02f5cde"],
    "sensory": ["988a967a0381fdeaac3d"],
    "whatsapp": [
        "2202a20a709b98e67c5a",  # WhatsApp Learning Chatbot
        "27f37fda439da68b2e80",  # WhatsApp slides
    ],
    "pong": ["0d2c1c6065e3c760cd00"],
    "art_gallery": [
        "73a06f6bc2f351938625",  # VR Art Gallery
        "e2b6009f25b1d2032ff3",  # TAMAC
    ],
    "gsc": ["e053f3083aef48c75416"],
    "storytelling": ["97d2865f30507158406a"],
    "cloud": [
        "981e75f7a8464972d606",  # Cloud Infrastructure
        "47ce657a47cd9fc742b8",  # Applied AI Infrastructure
    ],
    "futurescape": ["083f988b670ca9847caf"],
    "edge_ai": [
        "eb28c5d2c8f7cfc51559",  # EDge AI page
        "b9012597a859c6573b1b",  # EDge AI article
        "0d723a33322557853d3e",  # AI Education at the Edge
        "e39365ace9ba94243247",  # We Got a Spark
    ],
    "solarspell": [
        "49a444111c49800a5070",  # SolarSPELL partner page
        "f8bc7aefff5e7832807f",  # Solar-powered libraries + edge AI
    ],
    "broadband": [
        "9100ee241b5d839b60e9",  # FCC broadband analysis
        "ee78186b794ecfafd229",  # Broadband StoryMap
        "fc3500ab0d3863e5023c",  # Broadband stories
    ],
    "argentina": [
        "778331e86825878854b1",  # ASU News Argentina
        "b0e40364fdbd2d298a48",  # Argentina 2,000 faculty
    ],
    "battle_bands": [
        "015516b5670c29597f4e",  # ASU Enterprise Technology
        "f316e8aeb7f23a4855de",  # Next Lab blog
    ],
    "dreamscape": [
        "46c53be8861d3a9c46be",  # Dreamscape Learn introduction
        "783e0f27e8e476d636de",  # Dreamscape Learn
        "2349f13f34715ed5b105",  # Dreamscape transcript
    ],
    "verizon_5g": [
        "cd982d6a9b5c41f28d52",  # ASU and Verizon 5G
        "6bb33e216c08e135835f",  # VR Innovation Lab
    ],
    "reports": [
        "c1716db8c770d5a105fb",  # 2023/24 annual report
        "e671b7412103bcaa4057",  # 2024/25 OCR
        "3b40e5b9ba7c58bfc71a",  # 2025/26 OCR
        "a0f0127c97d434734cb1",  # FY25 year end
        "a3c928105e3dbd6678c8",  # SolarSPELL impact report
    ],
    "transcripts": [
        "9ea55b80eaa0189a37dd",  # ASU advances AI for all
        "3821ff9723d7ed76f4ae",  # DGX Spark
        "fd3c09306515953c4e28",  # Connecting AZ-1 episode 02
        "05c8ba3216d9033e7679",  # Hollywood and higher ed
        "61d1b43b41340aae541a",  # About Next Lab
        "b3307b5fb53abcb1dc2c",  # Unity for Humanity
        "8dd67b1affdc0b2bc2b0",  # Peer-to-peer learning
        "ac79d43bf87dfd74d69a",  # Build Your Own Olympic Village
        "f94f13b1fffd99d9c6d0",  # Tomorrow's Cure
    ],
    "social": [
        "f581faa76b21ef9d9096",  # Kelly Chung Mukherjee feed
        "2196b3dca0e770e3582d",  # Dan Munnerley posts
        "27c0fa8d739cdca6a683",  # ASU Next Lab company posts
        "c7dd717dfbda72ddb8ca",  # Amanda Federico feed
    ],
}

# Dense groups get fewer chunks per document so the pool stays readable.
DOC_CAP = 4
KEY_DOCS = {
    "e50ec06793148d622d31": 6,  # Surprise digital twin
    "29badab0cc60920eb15a": 6,  # Water Metaverse
    "f19093351a2ca724b886": 6,  # Acacia
    "0c38b485c723f39e7f46": 6,  # Betel-NextACT
    "988a967a0381fdeaac3d": 6,  # Sensory-friendly
    "2202a20a709b98e67c5a": 6,  # WhatsApp
    "c1716db8c770d5a105fb": 8,  # Annual report
    "a3c928105e3dbd6678c8": 6,  # SolarSPELL report
    "f581faa76b21ef9d9096": 6,  # Kelly feed
    "2196b3dca0e770e3582d": 6,  # Dan posts
}

MIN_CHARS = 200
MAX_CHARS = 4000
DISPLAY_CHARS = 1500

NAV_PATTERNS = re.compile(
    r"skip to content|leave a reply|your email address will not be published|"
    r"sign up for|subscribe to|privacy policy|terms of use|cookie policy|"
    r"all rights reserved|powered by wordpress|share this|follow us on",
    re.IGNORECASE,
)


def is_nav(text: str) -> bool:
    if NAV_PATTERNS.search(text):
        return True
    words = text.split()
    if len(words) < 35:
        return True
    # Link farms and menus have many short lines and few sentence stops.
    stops = sum(text.count(ch) for ch in ".!?")
    return stops < 2 and len(words) > 40


def pick_evenly(items: list[dict], limit: int) -> list[dict]:
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (limit - 1)
    seen: set[str] = set()
    chosen: list[dict] = []
    for index in range(limit):
        item = items[round(index * step)]
        if item["chunk_id"] not in seen:
            seen.add(item["chunk_id"])
            chosen.append(item)
    return chosen


def main() -> None:
    chunks = [json.loads(line) for line in (DATASET / "chunks.jsonl").open()]
    documents = {row["doc_id"]: row for row in map(json.loads, (DATASET / "documents.jsonl").open())}
    by_doc: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk["doc_id"], []).append(chunk)
    for rows in by_doc.values():
        rows.sort(key=lambda row: (row.get("chunk_index", 0), row["chunk_id"]))

    OUT.mkdir(exist_ok=True)
    pool: list[dict] = []
    groups_out: dict[str, dict] = {}
    seen_text: set[str] = set()
    counter = 0

    for group, doc_ids in GROUPS.items():
        group_rows: list[dict] = []
        for doc_id in doc_ids:
            document = documents.get(doc_id)
            if document is None:
                raise SystemExit(f"missing document {doc_id} in group {group}")
            eligible = [
                chunk
                for chunk in by_doc.get(doc_id, [])
                if MIN_CHARS <= len(chunk["text"]) <= MAX_CHARS and not is_nav(chunk["text"])
            ]
            if not eligible:
                continue
            cap = KEY_DOCS.get(doc_id, DOC_CAP)
            for chunk in pick_evenly(eligible, cap):
                fingerprint = re.sub(r"\s+", " ", chunk["text"][:120]).strip().lower()
                if fingerprint in seen_text:
                    continue
                seen_text.add(fingerprint)
                counter += 1
                text = chunk["text"]
                group_rows.append(
                    {
                        "candidate_id": f"c{counter:03d}",
                        "group": group,
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": doc_id,
                        "title": document.get("title") or chunk.get("title"),
                        "source_type": document.get("source_type") or chunk.get("source_type"),
                        "canonical_url": document.get("canonical_url") or chunk.get("canonical_url"),
                        "published": document.get("published"),
                        "section_heading": chunk.get("section_heading") or "",
                        "text": text[:DISPLAY_CHARS],
                        "text_chars": len(text),
                        "display_truncated": len(text) > DISPLAY_CHARS,
                    }
                )
        pool.extend(group_rows)
        groups_out[group] = {
            "doc_ids": doc_ids,
            "candidate_ids": [row["candidate_id"] for row in group_rows],
            "count": len(group_rows),
        }

    with (OUT / "candidates.jsonl").open("w", encoding="utf-8") as stream:
        for row in pool:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "candidate_groups.json").write_text(
        json.dumps(groups_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    counts: dict[str, int] = {}
    for row in pool:
        counts[row["group"]] = counts.get(row["group"], 0) + 1
    print(f"wrote {len(pool)} candidates across {len(groups_out)} groups")
    for group, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {group:16} {count:3}")


if __name__ == "__main__":
    main()
