"""Collect the public ASU Next Lab WordPress corpus.

The collector uses Python's standard library and an optional ``pypdf``
dependency for local PDF text extraction. It writes dated captures below
``sources/collections`` by default. It is a bounded, polite crawl:
robots.txt is fetched and enforced before discovery, the WordPress API is
limited to public read-only collections, and all URLs are deduplicated before
a request is made.
"""

from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import html as html_lib
import json
import logging
import mimetypes
import os
import posixpath
import re
import shutil
import sys
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.message import Message
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from xml.etree import ElementTree


LOG = logging.getLogger("nextlab_scraper")
USER_AGENT = "nextlab-public-corpus/1.0 (+https://nextlab.asu.edu/)"
DEFAULT_OUTPUT_DIR = Path("sources/collections")
TRACKING_QUERY_KEYS = {
    "_ga",
    "_gl",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
}
SENSITIVE_QUERY_PARTS = (
    "auth",
    "key",
    "nonce",
    "pass",
    "secret",
    "sig",
    "signature",
    "token",
)
LINKEDIN_INPUT_GLOB = "/tmp/linkedin-search-posts-*.json"
PRIVATE_PATH_PARTS = (
    "/wp-admin",
    "/wp-login.php",
    "/login",
    "/logout",
    "/account",
    "/members",
    "/private",
)
ASSET_SUFFIXES = {
    ".7z",
    ".avi",
    ".bmp",
    ".css",
    ".csv",
    ".doc",
    ".docx",
    ".eot",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".m4v",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".ogg",
    ".otf",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".svg",
    ".tar",
    ".tif",
    ".tiff",
    ".ttf",
    ".txt",
    ".wav",
    ".webm",
    ".webp",
    ".woff",
    ".woff2",
    ".xls",
    ".xlsx",
    ".xml",
    ".zip",
}
MEDIA_SUFFIXES = {
    ".avi",
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".m4v",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".ogg",
    ".pdf",
    ".png",
    ".svg",
    ".tif",
    ".tiff",
    ".wav",
    ".webm",
    ".webp",
}
SKIP_HTML_TAGS = {"script", "style", "noscript", "template", "svg", "nav", "header", "footer", "aside"}
BLOCK_HTML_TAGS = {"address", "article", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption", "figure", "li", "main", "p", "pre", "section", "table", "td", "th", "tr", "ul", "ol"}

# These public references are intentionally captured as bounded seed artifacts,
# not crawled recursively.  Export endpoints are used for Slides so the run
# retains a useful public PDF when anonymous access permits it; no edit/auth
# pages, video bytes, or private/authenticated content are requested.
CURATED_EXTERNAL_SEEDS: tuple[dict[str, str], ...] = (
    {
        "label": "impact_water_slides",
        "url": "https://docs.google.com/presentation/d/1CFTjw50YYzzEvLn9K0WxvU23V2B2xjJpfMR6p3-BgJo/edit",
        "capture_url": "https://docs.google.com/presentation/d/1CFTjw50YYzzEvLn9K0WxvU23V2B2xjJpfMR6p3-BgJo/export/pdf",
        "capture_kind": "pdf",
    },
    {
        "label": "surprise_digital_twin_slides",
        "url": "https://docs.google.com/presentation/d/1wKvw94183a5POEDTC1arvgn4qzNydHMKvzLjwLUvHJo/edit",
        "capture_url": "https://docs.google.com/presentation/d/1wKvw94183a5POEDTC1arvgn4qzNydHMKvzLjwLUvHJo/export/pdf",
        "capture_kind": "pdf",
    },
    {
        "label": "whatsapp_learning_slides",
        "url": "https://docs.google.com/presentation/d/1xPHw7VJ3NM_tMyjWKFcRQei5gYSk8RTiFG0pTdriJWc/edit",
        "capture_url": "https://docs.google.com/presentation/d/1xPHw7VJ3NM_tMyjWKFcRQei5gYSk8RTiFG0pTdriJWc/export/pdf",
        "capture_kind": "pdf",
    },
    {
        "label": "next_lab_published_deck",
        "url": "https://docs.google.com/presentation/d/e/2PACX-1vTWEhNCT0hn9SIGYxwxXKsanJwoEMuyb7ChfOMmiD_g1T90SD9YZv199vjdCt2q6sAyLwomazaamcKI/pubembed?delayms=5000&loop=false&start=false",
        "capture_url": "https://docs.google.com/presentation/d/e/2PACX-1vTWEhNCT0hn9SIGYxwxXKsanJwoEMuyb7ChfOMmiD_g1T90SD9YZv199vjdCt2q6sAyLwomazaamcKI/pubembed?delayms=5000&loop=false&start=false",
        "capture_kind": "html",
    },
    {
        "label": "impact_water_drive",
        "url": "https://drive.google.com/file/d/1_merCSboRXOmqNFNqGYQiFdikNjztLKT/view",
        "capture_url": "https://drive.google.com/file/d/1_merCSboRXOmqNFNqGYQiFdikNjztLKT/view",
        "capture_kind": "html",
    },
    {
        "label": "next_lab_eoy_report_issuu",
        "url": "https://issuu.com/asu_uto/docs/next_lab_eoy_report_2024-25",
        "capture_url": "https://issuu.com/asu_uto/docs/next_lab_eoy_report_2024-25",
        "capture_kind": "html",
    },
    {
        "label": "literary_magazine_canva",
        "url": "https://www.canva.com/design/DAGuxHyYalM/0Q3jTYiMHZs092rAZA94pg/view",
        "capture_url": "https://www.canva.com/design/DAGuxHyYalM/0Q3jTYiMHZs092rAZA94pg/view",
        "capture_kind": "html",
    },
    {
        "label": "betel_nextact_canva",
        "url": "https://www.canva.com/design/DAGre4Oeu0Y/clzdWLQsfaE457He-IJsJA/view",
        "capture_url": "https://www.canva.com/design/DAGre4Oeu0Y/clzdWLQsfaE457He-IJsJA/view",
        "capture_kind": "html",
    },
    {
        "label": "global_culture_explorer_canva",
        "url": "https://www.canva.com/design/DAGv5FdtOqs/6T_81UuIQ5mhNdaFYvbAdQ/view",
        "capture_url": "https://www.canva.com/design/DAGv5FdtOqs/6T_81UuIQ5mhNdaFYvbAdQ/view",
        "capture_kind": "html",
    },
    {
        "label": "alder_mayo_demo",
        "url": "https://alder-mayo-sept-8-2025.netlify.app/",
        "capture_url": "https://alder-mayo-sept-8-2025.netlify.app/",
        "capture_kind": "html",
    },
    {
        "label": "pong_demo",
        "url": "https://game-engine-pong-aug-26-2025.netlify.app/",
        "capture_url": "https://game-engine-pong-aug-26-2025.netlify.app/",
        "capture_kind": "html",
    },
    {
        "label": "nextact_demo",
        "url": "https://nextact-v-1-3.netlify.app/",
        "capture_url": "https://nextact-v-1-3.netlify.app/",
        "capture_kind": "html",
    },
    {
        "label": "matterport_acacia_demo",
        "url": "https://my.matterport.com/show/?m=KFbXU6ZREaj",
        "capture_url": "https://my.matterport.com/show/?m=KFbXU6ZREaj",
        "capture_kind": "html",
    },
    {
        "label": "zoom_fellowship_github_repo",
        "url": "https://github.com/shitijkarsolia/zoom-fellowship-website",
        "capture_url": "https://api.github.com/repos/shitijkarsolia/zoom-fellowship-website",
        "capture_kind": "github_repo",
        "license_status": "NOASSERTION",
    },
    {
        "label": "zoom_fellowship_github_readme",
        "url": "https://github.com/shitijkarsolia/zoom-fellowship-website#readme",
        "capture_url": "https://api.github.com/repos/shitijkarsolia/zoom-fellowship-website/readme",
        "capture_kind": "github_readme",
        "license_status": "NOASSERTION",
    },
    {
        "label": "asu_next_labs_unity_game_github_repo",
        "url": "https://github.com/hrushipandit/ASUNextLabs_UnityGame",
        "capture_url": "https://api.github.com/repos/hrushipandit/ASUNextLabs_UnityGame",
        "capture_kind": "github_repo",
        "license_status": "NOASSERTION",
    },
    {
        "label": "asu_next_labs_unity_game_github_readme",
        "url": "https://github.com/hrushipandit/ASUNextLabs_UnityGame#readme",
        "capture_url": "https://api.github.com/repos/hrushipandit/ASUNextLabs_UnityGame/readme",
        "capture_kind": "github_readme",
        "license_status": "NOASSERTION",
    },
    {
        "label": "spacewesus_game_of_life_github_repo",
        "url": "https://github.com/SpaceWesus/Game-Of-life",
        "capture_url": "https://api.github.com/repos/SpaceWesus/Game-Of-life",
        "capture_kind": "github_repo",
        "license_status": "NOASSERTION",
    },
    {
        "label": "spacewesus_pong_github_repo",
        "url": "https://github.com/SpaceWesus/Pong",
        "capture_url": "https://api.github.com/repos/SpaceWesus/Pong",
        "capture_kind": "github_repo",
        "license_status": "NOASSERTION",
    },
    {
        "label": "asu_nextlab_github_org",
        "url": "https://github.com/asu-nextlab",
        "capture_url": "https://api.github.com/orgs/asu-nextlab",
        "capture_kind": "json",
        "license_status": "NOASSERTION",
    },
    {
        "label": "asu_nextlab_github_dotgithub_repo",
        "url": "https://github.com/asu-nextlab/.github",
        "capture_url": "https://api.github.com/repos/asu-nextlab/.github",
        "capture_kind": "github_repo",
        "license_status": "NOASSERTION",
    },
    {
        "label": "asu_nextlab_digital_twin_website_repo",
        "url": "https://github.com/asu-nextlab/Digital-Twin-Website",
        "capture_url": "https://api.github.com/repos/asu-nextlab/Digital-Twin-Website",
        "capture_kind": "github_repo",
        "license_status": "NOASSERTION",
    },
    {
        "label": "asu_nextlab_mlapi_repo",
        "url": "https://github.com/asu-nextlab/ics-MLAPIWorkshop",
        "capture_url": "https://api.github.com/repos/asu-nextlab/ics-MLAPIWorkshop",
        "capture_kind": "github_repo",
        "license_status": "NOASSERTION",
    },
    {
        "label": "asu_nextlab_unity_ui_repo",
        "url": "https://github.com/asu-nextlab/ics-unity-ui",
        "capture_url": "https://api.github.com/repos/asu-nextlab/ics-unity-ui",
        "capture_kind": "github_repo",
        "license_status": "NOASSERTION",
    },
    {
        "label": "asu_nextlab_production_4k_yacht_repo",
        "url": "https://github.com/asu-nextlab/production-4k-yacht",
        "capture_url": "https://api.github.com/repos/asu-nextlab/production-4k-yacht",
        "capture_kind": "github_repo",
        "license_status": "NOASSERTION",
    },
    {
        "label": "asu_nextlab_restful_api_tutorial_repo",
        "url": "https://github.com/asu-nextlab/RestfulAPITutorial",
        "capture_url": "https://api.github.com/repos/asu-nextlab/RestfulAPITutorial",
        "capture_kind": "github_repo",
        "license_status": "NOASSERTION",
    },
    {
        "label": "asu_nextlab_local_slm_repo",
        "url": "https://github.com/asu-nextlab/self-serving-local-slm",
        "capture_url": "https://api.github.com/repos/asu-nextlab/self-serving-local-slm",
        "capture_kind": "github_repo",
        "license_status": "NOASSERTION",
    },
    {
        "label": "asu_nextlab_tilt_brush_repo",
        "url": "https://github.com/asu-nextlab/tilt-brush",
        "capture_url": "https://api.github.com/repos/asu-nextlab/tilt-brush",
        "capture_kind": "github_repo",
        "license_status": "Apache-2.0",
    },
    {
        "label": "asu_nextlab_zoom_metaverse_repo",
        "url": "https://github.com/asu-nextlab/Zoom-in-the-Metaverse-hackathon",
        "capture_url": "https://api.github.com/repos/asu-nextlab/Zoom-in-the-Metaverse-hackathon",
        "capture_kind": "github_repo",
        "license_status": "NOASSERTION",
    },
    {
        "label": "asu_engage_2025_event",
        "url": "https://tech.asu.edu/events/engage2025",
        "capture_url": "https://tech.asu.edu/events/engage2025",
        "capture_kind": "html",
        "related_asu": "true",
    },
    {
        "label": "asu_games_community_practice_event",
        "url": "https://asuevents.asu.edu/event/asu-games-community-practice-0",
        "capture_url": "https://asuevents.asu.edu/event/asu-games-community-practice-0",
        "capture_kind": "html",
        "related_asu": "true",
    },
    {
        "label": "asu_agentic_ai_2025_agenda",
        "url": "https://tech.asu.edu/events/agentic-ai-2025/agenda",
        "capture_url": "https://tech.asu.edu/events/agentic-ai-2025/agenda",
        "capture_kind": "html",
        "related_asu": "true",
    },
    {
        "label": "asu_empower_2026_event",
        "url": "https://tech.asu.edu/events/empower_2026",
        "capture_url": "https://tech.asu.edu/events/empower_2026",
        "capture_kind": "html",
        "related_asu": "true",
    },
    {
        "label": "asu_engage_2024_event",
        "url": "https://tech.asu.edu/events/engage2024",
        "capture_url": "https://tech.asu.edu/events/engage2024",
        "capture_kind": "html",
        "related_asu": "true",
    },
    {
        "label": "asu_zoom_spark_onepager",
        "url": "https://asuevents.asu.edu/sites/g/files/litvpz3506/files/2025-01/S2025%20Zoom%20Spark%201pgr.pdf",
        "capture_url": "https://asuevents.asu.edu/sites/g/files/litvpz3506/files/2025-01/S2025%20Zoom%20Spark%201pgr.pdf",
        "capture_kind": "pdf",
        "related_asu": "true",
    },
    {
        "label": "asu_folc_fest_posters",
        "url": "https://provost.asu.edu/sites/default/files/2024-03/FOLCFest_Posters-Session1.pdf",
        "capture_url": "https://provost.asu.edu/sites/default/files/2024-03/FOLCFest_Posters-Session1.pdf",
        "capture_kind": "pdf",
        "related_asu": "true",
    },
    {
        "label": "asu_folc_fest_demos",
        "url": "https://provost.asu.edu/sites/g/files/litvpz671/files/2024-03/FOLCFest_Demos-Session2.pdf",
        "capture_url": "https://provost.asu.edu/sites/g/files/litvpz671/files/2024-03/FOLCFest_Demos-Session2.pdf",
        "capture_kind": "pdf",
        "related_asu": "true",
    },
    {
        "label": "asu_enterprise_technology_fy25_year_end",
        "url": "https://tech.asu.edu/sites/g/files/litvpz4026/files/2025-11/FY25%20Strategic%20Priorities%20Year%20End%20Accomplishments%20-%20Enterprise%20Technology.pdf",
        "capture_url": "https://tech.asu.edu/sites/g/files/litvpz4026/files/2025-11/FY25%20Strategic%20Priorities%20Year%20End%20Accomplishments%20-%20Enterprise%20Technology.pdf",
        "capture_kind": "pdf",
        "related_asu": "true",
    },
    {
        "label": "asu_enterprise_technology_fy25_q1",
        "url": "https://tech.asu.edu/sites/default/files/2025-01/FY25%20Q1%20Strategic%20Priorities%20Issuu.pdf",
        "capture_url": "https://tech.asu.edu/sites/default/files/2025-01/FY25%20Q1%20Strategic%20Priorities%20Issuu.pdf",
        "capture_kind": "pdf",
        "related_asu": "true",
    },
    {
        "label": "asu_enterprise_technology_fy25_q2",
        "url": "https://tech.asu.edu/sites/g/files/litvpz601/files/2025-03/FY25%20Q2%20Accomplishments.pdf",
        "capture_url": "https://tech.asu.edu/sites/g/files/litvpz601/files/2025-03/FY25%20Q2%20Accomplishments.pdf",
        "capture_kind": "pdf",
        "related_asu": "true",
    },
    {
        "label": "asu_news_creativity_asuniverse",
        "url": "https://news.asu.edu/20221118-creativity-asuniverse-offer-immersive-campus-experiences-metaverse",
        "capture_url": "https://news.asu.edu/20221118-creativity-asuniverse-offer-immersive-campus-experiences-metaverse",
        "capture_kind": "html",
        "related_asu": "true",
    },
    {
        "label": "asu_news_digital_twins",
        "url": "https://news.asu.edu/20230220-campus-community-3-ways-digital-twins-can-enhance-experiences",
        "capture_url": "https://news.asu.edu/20230220-campus-community-3-ways-digital-twins-can-enhance-experiences",
        "capture_kind": "html",
        "related_asu": "true",
    },
    {
        "label": "asu_tech_students_next_lab_ai",
        "url": "https://tech.asu.edu/students-next-lab-work-bring-ai-all",
        "capture_url": "https://tech.asu.edu/students-next-lab-work-bring-ai-all",
        "capture_kind": "html",
        "related_asu": "true",
    },
    {
        "label": "asu_tech_ai_disparities_summit",
        "url": "https://tech.asu.edu/features/2-asu-experts-join-national-summit-address-ai-disparities",
        "capture_url": "https://tech.asu.edu/features/2-asu-experts-join-national-summit-address-ai-disparities",
        "capture_kind": "html",
        "related_asu": "true",
    },
    {
        "label": "ntsa_ifest_learning_paradigms",
        "url": "https://www.ntsa.org/-/media/sites/ntsa/events/2022/21d0/presentations/20220817_ifest_new-paradigms-of-learning.pdf",
        "capture_url": "https://www.ntsa.org/-/media/sites/ntsa/events/2022/21d0/presentations/20220817_ifest_new-paradigms-of-learning.pdf",
        "capture_kind": "pdf",
        "related_asu": "third_party_related",
    },
    {
        "label": "chronicle_digital_twin_higher_ed",
        "url": "https://connect.chronicle.com/rs/931-EKA-218/images/chronicle-intelligence-explainer-digital-twin-technology-and-higher-ed-uof.pdf?version=0",
        "capture_url": "https://connect.chronicle.com/rs/931-EKA-218/images/chronicle-intelligence-explainer-digital-twin-technology-and-higher-ed-uof.pdf?version=0",
        "capture_kind": "pdf",
        "related_asu": "third_party_related",
    },
    {
        "label": "next_lab_futurescape_missing_sitemap",
        "url": "https://nextlab.asu.edu/futurescape/",
        "capture_url": "https://nextlab.asu.edu/futurescape/",
        "capture_kind": "html",
        "related_asu": "true",
    },
)

YOUTUBE_SEEDS: tuple[dict[str, str], ...] = tuple(
    {
        "label": f"youtube_{video_id}",
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "capture_url": "https://www.youtube.com/oembed?format=json&url="
        + urllib.parse.quote(f"https://www.youtube.com/watch?v={video_id}", safe=""),
        "capture_kind": "json",
    }
    for video_id in ("AvgZscNZzYw", "iGNQPSnvf4w", "f-2JyjJPZpE", "MirHJNolc_k")
)

KNOWN_LEGACY_BLOCKERS: tuple[dict[str, str], ...] = tuple(
    {
        "label": f"{host}_{path.replace('/', '_')}_http_522",
        "url": f"https://{host}{path}",
        "capture_url": f"https://{host}{path}",
        "capture_kind": "known_blocker",
        "status": "522",
        "error": "known_inaccessible_legacy_domain",
    }
    for host in ("lf.asu.edu", "lfc.asu.edu")
    for path in ("/robots.txt", "/wp-json/wp/v2/posts?per_page=1", "/wp-sitemap.xml")
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonicalize_url(url: str, base_url: str | None = None) -> str:
    """Return a stable URL used for deduplication and document IDs.

    Fragments and common tracking parameters are removed.  Meaningful query
    parameters are retained, while URLs containing credential-like parameters
    are rejected by :func:`has_sensitive_query` before fetching.
    """

    if base_url:
        url = urllib.parse.urljoin(base_url, url)
    parsed = urllib.parse.urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        return ""
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        return ""
    try:
        port = parsed.port
    except ValueError:
        return ""
    netloc = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    path = parsed.path or "/"
    # Normalize repeated slashes and dot segments without decoding path bytes.
    trailing = path.endswith("/")
    path = posixpath.normpath(path)
    if not path.startswith("/"):
        path = "/" + path
    if trailing and path != "/":
        path += "/"
    path = urllib.parse.quote(urllib.parse.unquote(path), safe="/%:@-._~!$&'()*+,;=")
    query_pairs = []
    for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
        if key.lower() in TRACKING_QUERY_KEYS or key.lower().startswith("utm_"):
            continue
        query_pairs.append((key, value))
    query_pairs.sort()
    query = urllib.parse.urlencode(query_pairs, doseq=True)
    return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))


def has_sensitive_query(url: str) -> bool:
    """Detect query parameters that could carry credentials or signed URLs."""

    try:
        pairs = urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query, keep_blank_values=True)
    except ValueError:
        return True
    for key, value in pairs:
        probe = f"{key} {value}".lower()
        if any(part in key.lower() for part in SENSITIVE_QUERY_PARTS):
            return True
        if any(part in probe for part in ("bearer ", "basic ", "-----begin ")):
            return True
    return False


def hostname(url: str) -> str:
    return (urllib.parse.urlsplit(url).hostname or "").lower().rstrip(".")


def same_site(url: str, base_url: str) -> bool:
    target = hostname(url)
    base = hostname(base_url)
    return target in {base, f"www.{base}"} or base in {target, f"www.{target}"}


def is_private_url(url: str) -> bool:
    path = urllib.parse.urlsplit(url).path.lower()
    return any(path == part or path.startswith(part + "/") or path.startswith(part + "?") for part in PRIVATE_PATH_PARTS)


def url_suffix(url: str) -> str:
    path = urllib.parse.urlsplit(url).path.lower()
    return Path(path).suffix


def looks_like_asset(url: str, mime: str | None = None) -> bool:
    suffix = url_suffix(url)
    if suffix in ASSET_SUFFIXES:
        return True
    if mime:
        major = mime.lower().split(";", 1)[0].strip()
        return major.startswith(("image/", "video/", "audio/")) or major in {
            "application/pdf",
            "application/zip",
            "application/octet-stream",
            "font/woff",
            "font/woff2",
        }
    return False


def content_type_major(mime: str | None) -> str:
    return (mime or "").split(";", 1)[0].strip().lower()


def looks_like_json_body(data: bytes) -> bool:
    probe = data.lstrip()[:1]
    return probe in {b"{", b"["}


def looks_like_html_body(data: bytes) -> bool:
    probe = data.lstrip()[:512].lower()
    return probe.startswith((b"<!doctype html", b"<html", b"<head", b"<body")) or b"<html" in probe


def response_matches_kind(data: bytes, mime: str, kind: str | None) -> bool:
    """Reject successful responses whose payload does not match its role."""

    if not data:
        return False
    if not kind:
        return True
    major = content_type_major(mime)
    if kind == "json":
        return major == "application/json" or major.endswith("+json") or looks_like_json_body(data)
    if kind in {"xml", "feed"}:
        return major in {"application/xml", "text/xml", "application/rss+xml", "application/atom+xml"} or data.lstrip().startswith(b"<")
    if kind == "html":
        if major in {"application/json", "text/json"} or major.endswith("+json") or looks_like_json_body(data):
            return False
        return major in {"text/html", "application/xhtml+xml", "text/plain", ""} and (
            major != "" or looks_like_html_body(data)
        )
    if kind == "pdf":
        return major == "application/pdf" or data.startswith(b"%PDF-")
    if kind == "media":
        if major.startswith(("image/", "video/", "audio/")) or major in {
            "application/pdf",
            "application/octet-stream",
            "application/zip",
        }:
            return True
        return any(
            data.startswith(signature)
            for signature in (b"%PDF-", b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF8", b"RIFF", b"ID3", b"OggS")
        )
    if kind == "asset":
        return major not in {"text/html", "application/xhtml+xml", "application/json", "text/json"} and not looks_like_html_body(data)
    return True


def safe_filename(url: str, mime: str | None = None, default_suffix: str = ".bin") -> str:
    parsed = urllib.parse.urlsplit(url)
    basename = Path(urllib.parse.unquote(parsed.path)).name
    basename = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")
    suffix = Path(basename).suffix.lower()
    if not suffix and mime:
        suffix = mimetypes.guess_extension(mime.split(";", 1)[0].strip()) or default_suffix
    if not suffix:
        suffix = default_suffix
    return (basename or "resource")[:120] + suffix if not basename.endswith(suffix) else (basename or "resource")[:120]


def decode_body(data: bytes, content_type: str = "") -> str:
    charset = None
    match = re.search(r"charset\s*=\s*[\"']?([A-Za-z0-9._-]+)", content_type, flags=re.I)
    if match:
        charset = match.group(1)
    for encoding in (charset, "utf-8", "latin-1"):
        if not encoding:
            continue
        try:
            return data.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="replace")


def clean_text(value: str) -> str:
    value = html_lib.unescape(value)
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


class PageParser(HTMLParser):
    """Extract canonical metadata, links, and heading-delimited visible text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.canonical: str | None = None
        self.links: list[tuple[str, str]] = []
        self.feeds: list[str] = []
        self.sections: list[dict[str, Any]] = [{"heading": "", "level": 0, "parts": []}]
        self._skip_depth = 0
        self._title_depth = 0
        self._heading_tag: str | None = None
        self._heading_parts: list[str] = []
        self._block_depth = 0

    @staticmethod
    def _attrs(attrs: Sequence[tuple[str, str | None]]) -> dict[str, str]:
        return {key.lower(): value or "" for key, value in attrs}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attr = self._attrs(attrs)
        if tag in SKIP_HTML_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._title_depth += 1
        if tag == "link":
            rel = set(attr.get("rel", "").lower().split())
            href = attr.get("href", "").strip()
            if href and "canonical" in rel:
                self.canonical = href
            if href and ("alternate" in rel or "feed" in rel) and "rss" in attr.get("type", "").lower() + attr.get("rel", "").lower():
                self.feeds.append(href)
            if href:
                self.links.append((href, "link"))
        for key in ("href", "src", "data-src", "data-lazy-src", "poster", "data-url", "data-href"):
            value = attr.get(key, "").strip()
            if value:
                self.links.append((value, tag))
        srcset = attr.get("srcset", "") or attr.get("data-srcset", "")
        for candidate in srcset.split(","):
            candidate_url = candidate.strip().split(None, 1)[0] if candidate.strip() else ""
            if candidate_url:
                self.links.append((candidate_url, f"{tag}-srcset"))
        if tag in {f"h{i}" for i in range(1, 7)}:
            self._heading_tag = tag
            self._heading_parts = []
        elif tag in BLOCK_HTML_TAGS:
            self._block_depth += 1
            if self.sections[-1]["parts"] and self.sections[-1]["parts"][-1] != "\n":
                self.sections[-1]["parts"].append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in SKIP_HTML_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._title_depth = max(0, self._title_depth - 1)
        if self._heading_tag == tag:
            heading = clean_text(" ".join(self._heading_parts))
            if heading:
                self.sections.append({"heading": heading, "level": int(tag[1]), "parts": []})
            self._heading_tag = None
            self._heading_parts = []
        if tag in BLOCK_HTML_TAGS:
            self._block_depth = max(0, self._block_depth - 1)
            if self.sections[-1]["parts"] and self.sections[-1]["parts"][-1] != "\n":
                self.sections[-1]["parts"].append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._title_depth:
            self.title_parts.append(data)
        if self._heading_tag:
            self._heading_parts.append(data)
        else:
            self.sections[-1]["parts"].append(data)

    def result(self) -> dict[str, Any]:
        sections = []
        for section in self.sections:
            text = clean_text("".join(section["parts"]))
            if text:
                sections.append({"heading": section["heading"], "level": section["level"], "text": text})
        if not sections:
            sections = [{"heading": "", "level": 0, "text": ""}]
        return {
            "title": clean_text(" ".join(self.title_parts)),
            "canonical": self.canonical,
            "links": self.links,
            "feeds": self.feeds,
            "sections": sections,
            "text": clean_text("\n\n".join(section["text"] for section in sections if section["text"])),
        }


def parse_html_document(data: bytes | str, content_type: str = "") -> dict[str, Any]:
    parser = PageParser()
    parser.feed(data if isinstance(data, str) else decode_body(data, content_type))
    parser.close()
    return parser.result()


def parse_markdown_document(data: bytes | str, title: str = "") -> dict[str, Any]:
    """Turn a public README snapshot into heading-aware plain text."""

    text = decode_body(data if isinstance(data, bytes) else data.encode("utf-8"), "text/plain")
    sections: list[dict[str, Any]] = [{"heading": "", "level": 0, "parts": []}]
    for line in text.splitlines():
        heading_match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading_match:
            heading = clean_text(re.sub(r"[`*_]", "", heading_match.group(2)))
            if heading:
                sections.append({"heading": heading, "level": len(heading_match.group(1)), "parts": []})
            continue
        line = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"[`*_~]", "", line)
        if line.strip():
            sections[-1]["parts"].append(line)
    parsed_sections = []
    for section in sections:
        section_text = clean_text("\n".join(section["parts"]))
        if section_text:
            parsed_sections.append({"heading": section["heading"], "level": section["level"], "text": section_text})
    if not parsed_sections:
        parsed_sections = [{"heading": title, "level": 1, "text": clean_text(text)}] if clean_text(text) else []
    return {
        "title": clean_text(title),
        "canonical": None,
        "links": [],
        "feeds": [],
        "sections": parsed_sections,
        "text": clean_text("\n\n".join(section["text"] for section in parsed_sections)),
    }


def parse_sitemap(data: bytes | str) -> tuple[str, list[dict[str, str]]]:
    """Parse a sitemap into ``(kind, entries)`` without relying on prefixes."""

    root = ElementTree.fromstring(data if isinstance(data, bytes) else data.encode("utf-8"))
    kind = root.tag.rsplit("}", 1)[-1].lower()
    entries: list[dict[str, str]] = []
    for element in root:
        name = element.tag.rsplit("}", 1)[-1].lower()
        if name not in {"url", "sitemap"}:
            continue
        row: dict[str, str] = {}
        for child in element:
            child_name = child.tag.rsplit("}", 1)[-1].lower()
            if child_name in {"loc", "lastmod", "changefreq", "priority"} and child.text:
                row[child_name] = child.text.strip()
        if row.get("loc"):
            entries.append(row)
    return kind, entries


def parse_feed(data: bytes | str, content_type: str = "") -> list[dict[str, str]]:
    """Extract links/titles from RSS or Atom while retaining the raw feed."""

    root = ElementTree.fromstring(data if isinstance(data, bytes) else data.encode("utf-8"))
    items: list[dict[str, str]] = []
    for item in root.iter():
        name = item.tag.rsplit("}", 1)[-1].lower()
        if name not in {"item", "entry"}:
            continue
        row: dict[str, str] = {}
        for child in item:
            child_name = child.tag.rsplit("}", 1)[-1].lower()
            text = "".join(child.itertext()).strip() if child is not None else ""
            if child_name == "link":
                row["link"] = child.attrib.get("href", text)
            elif child_name in {"title", "pubdate", "published", "updated", "id"}:
                row[child_name] = text
        if row:
            items.append(row)
    return items


def chunk_text(text: str, max_chars: int = 1600, overlap: int = 200) -> list[str]:
    """Split text at word boundaries with a bounded character overlap."""

    text = clean_text(text)
    if not text:
        return []
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            chunk = " ".join(current).strip()
            chunks.append(chunk)
            overlap_words: list[str] = []
            overlap_len = 0
            for prior in reversed(current):
                prior_extra = len(prior) + (1 if overlap_words else 0)
                if overlap_len + prior_extra > overlap:
                    break
                overlap_words.append(prior)
                overlap_len += prior_extra
            current = list(reversed(overlap_words))
            current_len = len(" ".join(current))
        current.append(word)
        current_len += extra
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def _extract_pdf_text_pdftotext(path: Path, max_chars: int) -> tuple[str | None, str]:
    executable = shutil.which("pdftotext")
    if not executable:
        return None, "pdftotext_unavailable"
    try:
        completed = subprocess.run(
            [executable, "-layout", str(path), "-"],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return None, "pdftotext_timeout"
    except OSError:
        return None, "pdftotext_failed"
    if completed.returncode != 0:
        return None, "pdftotext_failed"
    text = clean_text(decode_body(completed.stdout[: max_chars * 4], "text/plain")[:max_chars])
    return (text, "extracted_pdftotext") if text else (None, "no_extractable_text")


def _extract_pdf_text_pypdf(path: Path, max_chars: int, max_pages: int = 500) -> tuple[str | None, str]:
    """Extract bounded text with optional pypdf, without making network calls."""

    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        return None, "pypdf_unavailable"
    try:
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")
            except Exception:
                return None, "encrypted"
            if not decrypted:
                return None, "encrypted"
        parts: list[str] = []
        extracted_chars = 0
        failed_pages = 0
        truncated = False
        for page_index, page in enumerate(reader.pages):
            if page_index >= max_pages:
                truncated = True
                break
            try:
                page_text = page.extract_text() or ""
            except Exception:
                failed_pages += 1
                continue
            if not page_text:
                continue
            remaining = max_chars - extracted_chars
            if remaining <= 0:
                truncated = True
                break
            page_text = page_text[:remaining]
            parts.append(page_text)
            extracted_chars += len(page_text)
            if extracted_chars >= max_chars:
                truncated = True
                break
        text = clean_text("\n\n".join(parts))
        if text:
            return text, "extracted_pypdf_truncated" if truncated else "extracted_pypdf"
        if failed_pages:
            return None, "pypdf_page_failed"
        return None, "no_extractable_text"
    except Exception:
        return None, "pypdf_failed"


def extract_pdf_text_with_status(path: Path, max_chars: int = 2_000_000) -> tuple[str | None, str]:
    """Extract bounded local PDF text and return a per-file status."""

    text, status = _extract_pdf_text_pdftotext(path, max_chars)
    if text:
        return text, status
    fallback_text, fallback_status = _extract_pdf_text_pypdf(path, max_chars)
    if fallback_text:
        return fallback_text, fallback_status
    return None, fallback_status if fallback_status != "pypdf_unavailable" else status


def extract_pdf_text(path: Path, max_chars: int = 2_000_000) -> str | None:
    """Extract bounded public PDF text using pdftotext or optional pypdf."""

    return extract_pdf_text_with_status(path, max_chars)[0]


@dataclass
class CollectorConfig:
    base_url: str = "https://nextlab.asu.edu/"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    delay_seconds: float = 0.25
    max_attempts: int = 2
    request_timeout_seconds: float = 35.0
    max_bytes: int = 64 * 1024 * 1024
    max_sitemaps: int = 25
    max_api_pages_per_type: int = 50
    max_html_pages: int = 1000
    max_assets: int = 750
    max_discovered_urls: int = 6000
    chunk_chars: int = 1600
    chunk_overlap: int = 200
    linkedin_input_glob: str = LINKEDIN_INPUT_GLOB
    max_linkedin_files: int = 25
    max_linkedin_posts: int = 500
    capture_external_seeds: bool = True
    related_url_file: str = "/tmp/munnerley-urls.txt"
    max_related_urls: int = 250

    def normalized(self) -> "CollectorConfig":
        self.base_url = canonicalize_url(self.base_url) or "https://nextlab.asu.edu/"
        if not self.base_url.endswith("/"):
            self.base_url += "/"
        return self


@dataclass
class FetchResult:
    requested_url: str
    final_url: str
    status: int
    mime: str
    body: bytes = b""
    path: str = ""
    sha256: str = ""
    error: str = ""
    fetched_at: str = ""
    headers: dict[str, str] = field(default_factory=dict)


class RobotsPolicy:
    def __init__(self, base_url: str, text: str) -> None:
        self.base_url = base_url
        self.text = text
        self.parser = urllib.robotparser.RobotFileParser()
        self.parser.set_url(urllib.parse.urljoin(base_url, "/robots.txt"))
        self.parser.parse(text.splitlines())
        self.sitemaps = [line.split(":", 1)[1].strip() for line in text.splitlines() if line.lower().startswith("sitemap:")]

    def can_fetch(self, url: str) -> bool:
        return not is_private_url(url) and self.parser.can_fetch(USER_AGENT, url)

    @property
    def crawl_delay(self) -> float:
        return max(self.parser.crawl_delay(USER_AGENT) or 0, self.parser.crawl_delay("*") or 0)


class Fetcher:
    def __init__(self, config: CollectorConfig, run_dir: Path, robots: RobotsPolicy | None = None) -> None:
        self.config = config
        self.run_dir = run_dir
        self.robots = robots
        self.last_request_at = 0.0
        self.seen: set[str] = set()
        self.manifest: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.bytes_by_source: Counter[str] = Counter()
        self.count_by_source: Counter[str] = Counter()
        self.url_status: dict[str, FetchResult] = {}

    def _pace(self) -> None:
        delay = max(self.config.delay_seconds, self.robots.crawl_delay if self.robots else 0)
        wait = delay - (time.monotonic() - self.last_request_at)
        if wait > 0:
            time.sleep(wait)
        self.last_request_at = time.monotonic()

    def _manifest_record(self, result: FetchResult, source_type: str, rights_status: str = "unreviewed-public") -> None:
        record = {
            "requested_url": result.requested_url,
            "final_url": result.final_url,
            "status": result.status,
            "mime": result.mime,
            "bytes": len(result.body),
            "sha256": result.sha256,
            "fetched_at": result.fetched_at,
            "source_type": source_type,
            "rights_status": rights_status,
            "path": result.path,
        }
        if result.error:
            record["error"] = result.error
            self.errors.append(record)
        self.manifest.append(record)
        self.count_by_source[source_type] += 1
        self.bytes_by_source[source_type] += len(result.body)

    def _write_body(self, body: bytes, url: str, mime: str, kind: str) -> tuple[str, str]:
        digest = hashlib.sha256(body).hexdigest()
        if kind == "robots":
            relative = Path("raw") / "robots.txt"
        else:
            suffix = safe_filename(url, mime).rsplit(".", 1)[-1]
            relative = Path("raw") / kind / digest[:2] / f"{digest}.{suffix}"
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(body)
        return str(relative), digest

    def fetch(
        self,
        url: str,
        source_type: str,
        kind: str,
        *,
        respect_robots: bool = True,
        force: bool = False,
        content_kind: str | None = None,
        attempts: int | None = None,
    ) -> FetchResult:
        canonical = canonicalize_url(url, self.config.base_url)
        now = utc_now()
        if not canonical:
            result = FetchResult(url, url, 0, "", error="invalid_url", fetched_at=now)
            self._manifest_record(result, source_type)
            return result
        if has_sensitive_query(canonical):
            result = FetchResult(canonical, canonical, 0, "", error="sensitive_query_refused", fetched_at=now)
            self._manifest_record(result, source_type)
            return result
        if self.robots and respect_robots and not self.robots.can_fetch(canonical):
            result = FetchResult(canonical, canonical, 0, "", error="robots_denied", fetched_at=now)
            self._manifest_record(result, source_type)
            return result
        if not force and canonical in self.seen:
            return self.url_status[canonical]
        self.seen.add(canonical)
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml,application/rss+xml,application/json,image/avif,image/webp,*/*;q=0.5",
            # urllib does not transparently decode Content-Encoding.  Asking
            # for identity keeps the saved raw payload parseable and avoids
            # mistaking a compressed response for malformed XML/JSON.
            "Accept-Encoding": "identity",
        }
        last_error = ""
        response_status = 0
        response_mime = ""
        max_attempts = max(1, attempts if attempts is not None else self.config.max_attempts)
        for attempt in range(max_attempts):
            self._pace()
            request = urllib.request.Request(canonical, headers=headers, method="GET")
            response = None
            response_deadline = time.monotonic() + self.config.request_timeout_seconds
            try:
                response = urllib.request.urlopen(request, timeout=self.config.request_timeout_seconds)
                status = int(getattr(response, "status", 200) or 200)
                final_url = canonicalize_url(response.geturl()) or canonical
                mime = response.headers.get("Content-Type", "").strip()
                response_status = status
                response_mime = mime
                try:
                    # A server that trickles a byte before every socket timeout
                    # can otherwise evade urllib's idle timeout forever.  Keep
                    # individual reads short and enforce the overall deadline
                    # below as well.
                    response.fp.raw._sock.settimeout(min(self.config.request_timeout_seconds, 10.0))
                except (AttributeError, OSError):
                    pass
                body = bytearray()
                while True:
                    if time.monotonic() >= response_deadline:
                        raise TimeoutError("response_deadline_exceeded")
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    body.extend(chunk)
                    if len(body) > self.config.max_bytes:
                        raise ValueError(f"response_exceeds_{self.config.max_bytes}_bytes")
                data = bytes(body)
                expected_length = response.headers.get("Content-Length", "").strip()
                if expected_length:
                    try:
                        if int(expected_length) != len(data):
                            raise ValueError(f"truncated_body_expected_{expected_length}_got_{len(data)}")
                    except ValueError as exc:
                        if str(exc).startswith("truncated_body_"):
                            raise
                if not data:
                    raise ValueError("empty_body")
                if not response_matches_kind(data, mime, content_kind):
                    relative, digest = self._write_body(data, final_url, mime, "rejected")
                    result = FetchResult(
                        canonical,
                        final_url,
                        status,
                        mime,
                        data,
                        relative,
                        digest,
                        error=f"unexpected_content_type:{content_type_major(mime) or 'missing'}",
                        fetched_at=utc_now(),
                    )
                    self.url_status[canonical] = result
                    self._manifest_record(result, source_type)
                    return result
                relative, digest = self._write_body(data, final_url, mime, kind)
                response_headers = {key.lower(): value for key, value in response.headers.items()}
                result = FetchResult(
                    canonical,
                    final_url,
                    status,
                    mime,
                    data,
                    relative,
                    digest,
                    fetched_at=utc_now(),
                    headers=response_headers,
                )
                self.url_status[canonical] = result
                self._manifest_record(result, source_type)
                return result
            except urllib.error.HTTPError as exc:
                last_error = f"http_{exc.code}"
                if exc.code in {429, 500, 502, 503, 504} and attempt + 1 < max_attempts:
                    retry_after = 0.0
                    try:
                        retry_after = min(float(exc.headers.get("Retry-After", "0")), 10.0)
                    except (TypeError, ValueError):
                        pass
                    if retry_after:
                        time.sleep(retry_after)
                    continue
                result = FetchResult(canonical, canonical, exc.code, exc.headers.get("Content-Type", ""), error=last_error, fetched_at=utc_now())
                self.url_status[canonical] = result
                self._manifest_record(result, source_type)
                return result
            except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
                last_error = type(exc).__name__ + (f": {exc}" if str(exc) else "")
                if attempt + 1 < max_attempts:
                    time.sleep(min(2.0 * (attempt + 1), 5.0))
                    continue
                result = FetchResult(canonical, canonical, response_status, response_mime, error=last_error, fetched_at=utc_now())
                self.url_status[canonical] = result
                self._manifest_record(result, source_type)
                return result
            finally:
                if response is not None:
                    response.close()
        result = FetchResult(canonical, canonical, 0, "", error=last_error or "fetch_failed", fetched_at=utc_now())
        self.url_status[canonical] = result
        self._manifest_record(result, source_type)
        return result

    def write_outputs(self) -> None:
        manifest_path = self.run_dir / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for record in self.manifest:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def record_local(self, payload: bytes, source_url: str, source_type: str, kind: str = "linkedin") -> str:
        """Persist an already-authorized local payload with provenance."""

        relative, digest = self._write_body(payload, source_url, "application/json", kind)
        result = FetchResult(
            requested_url=source_url,
            final_url=source_url,
            status=200,
            mime="application/json",
            body=payload,
            path=relative,
            sha256=digest,
            fetched_at=utc_now(),
        )
        self._manifest_record(result, source_type, rights_status="authorized-readonly-public")
        return relative


class Collector:
    def __init__(self, config: CollectorConfig) -> None:
        self.config = config.normalized()
        started = datetime.now(timezone.utc)
        self.run_id = started.strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = self.config.output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.robots: RobotsPolicy | None = None
        self.fetcher = Fetcher(self.config, self.run_dir)
        self.page_queue: deque[tuple[str, str]] = deque()
        self.asset_queue: deque[tuple[str, str]] = deque()
        self.sitemap_queue: deque[str] = deque()
        self.page_seen: set[str] = set()
        self.asset_seen: set[str] = set()
        self.sitemap_seen: set[str] = set()
        self.discovered_urls = 0
        self.documents: dict[str, dict[str, Any]] = {}
        self.wordpress_records: list[dict[str, Any]] = []
        self.wordpress_seen: set[tuple[str, str]] = set()
        self.duplicate_wordpress_records = 0
        self.api_pagination: dict[str, dict[str, Any]] = {}
        self.processing_errors: list[dict[str, Any]] = []
        self.feed_items: list[dict[str, Any]] = []
        self.feed_seen: set[str] = set()
        self.feed_item_seen: set[str] = set()
        self.external_links: dict[str, dict[str, Any]] = {}
        self.external_seed_records: list[dict[str, Any]] = []
        self.github_repo_summaries: list[dict[str, Any]] = []
        self.linkedin_posts: list[dict[str, Any]] = []
        self.linkedin_seen: set[str] = set()
        self.linkedin_post_urls: list[dict[str, Any]] = []
        self.linkedin_url_seen: set[str] = set()
        self.linkedin_profiles: list[dict[str, Any]] = []
        self.html_count = 0
        self.asset_count = 0
        self.related_asset_count = 0
        self.started_at = utc_now()

    def _record_processing_error(self, source_type: str, url: str, error: str, **details: Any) -> None:
        row: dict[str, Any] = {
            "source_type": source_type,
            "url": url,
            "error": error,
        }
        row.update(details)
        self.processing_errors.append(row)

    @staticmethod
    def _is_api_url(url: str) -> bool:
        parsed = urllib.parse.urlsplit(url)
        return parsed.path.rstrip("/").lower().startswith("/wp-json/") or parsed.query.lower().startswith("rest_route=")

    def _remember_external(self, url: str, source_url: str, **metadata: Any) -> None:
        if has_sensitive_query(url):
            return
        row = self.external_links.setdefault(url, {"url": url, "sources": []})
        if source_url and source_url not in row["sources"]:
            row["sources"].append(source_url)
        row.update(metadata)

    def _queue_url(self, target: str, source_url: str, queue: deque[tuple[str, str]], seen: set[str], source_type: str) -> None:
        if self.discovered_urls >= self.config.max_discovered_urls:
            return
        url = canonicalize_url(target, source_url)
        if self._is_api_url(url):
            return
        if not url or not same_site(url, self.config.base_url) or is_private_url(url) or has_sensitive_query(url):
            if url and not same_site(url, self.config.base_url):
                self._remember_external(url, source_url)
            return
        if url not in seen:
            seen.add(url)
            queue.append((url, source_type))
            self.discovered_urls += 1

    def _queue_link(self, target: str, source_url: str, source_type: str = "html_link") -> None:
        url = canonicalize_url(target, source_url)
        if not url:
            return
        if self._is_api_url(url) or has_sensitive_query(url):
            return
        if same_site(url, self.config.base_url) and not is_private_url(url) and not has_sensitive_query(url):
            if looks_like_asset(url):
                self._queue_url(url, source_url, self.asset_queue, self.asset_seen, source_type)
            else:
                self._queue_url(url, source_url, self.page_queue, self.page_seen, "discovered_page")
        elif not same_site(url, self.config.base_url):
            self._remember_external(url, source_url)

    def _fetch_robots(self) -> None:
        result = self.fetcher.fetch(
            urllib.parse.urljoin(self.config.base_url, "/robots.txt"),
            "robots",
            "robots",
            respect_robots=False,
            content_kind="text",
        )
        text = decode_body(result.body, result.mime) if result.status == 200 else ""
        self.robots = RobotsPolicy(self.config.base_url, text)
        self.fetcher.robots = self.robots
        self.sitemap_queue.extend(self.robots.sitemaps)

    def _discover_sitemaps(self) -> None:
        defaults = [
            urllib.parse.urljoin(self.config.base_url, "/sitemap.xml"),
            urllib.parse.urljoin(self.config.base_url, "/sitemap_index.xml"),
            urllib.parse.urljoin(self.config.base_url, "/wp-sitemap.xml"),
        ]
        for sitemap in defaults:
            if sitemap not in self.sitemap_queue:
                self.sitemap_queue.append(sitemap)
        while self.sitemap_queue and len(self.sitemap_seen) < self.config.max_sitemaps:
            sitemap = canonicalize_url(self.sitemap_queue.popleft(), self.config.base_url)
            if not sitemap or sitemap in self.sitemap_seen or not same_site(sitemap, self.config.base_url):
                continue
            self.sitemap_seen.add(sitemap)
            result = self.fetcher.fetch(sitemap, "sitemap", "sitemap", content_kind="xml")
            if result.status != 200 or not result.body:
                continue
            try:
                kind, entries = parse_sitemap(result.body)
            except (ElementTree.ParseError, ValueError) as exc:
                LOG.warning("Could not parse sitemap %s: %s", sitemap, exc)
                self._record_processing_error("sitemap", sitemap, "parse_error", detail=str(exc))
                continue
            for entry in entries:
                loc = canonicalize_url(entry.get("loc", ""), sitemap)
                if not loc:
                    continue
                if kind in {"sitemapindex", "sitemap"}:
                    if same_site(loc, self.config.base_url):
                        self.sitemap_queue.append(loc)
                elif kind in {"urlset", "url"} and same_site(loc, self.config.base_url):
                    if looks_like_asset(loc):
                        self._queue_url(loc, sitemap, self.asset_queue, self.asset_seen, "sitemap_asset")
                    else:
                        self._queue_url(loc, sitemap, self.page_queue, self.page_seen, "sitemap_page")

    def _fetch_wordpress_api(self) -> None:
        api_base = urllib.parse.urljoin(self.config.base_url, "/wp-json/wp/v2/")
        for collection in ("pages", "posts", "categories", "tags", "media"):
            self._fetch_wordpress_collection(api_base, collection, paginated=True)
        for collection in ("types", "taxonomies"):
            self._fetch_wordpress_collection(api_base, collection, paginated=False)

    def _fetch_wordpress_collection(self, api_base: str, collection: str, *, paginated: bool) -> None:
        expected_pages: int | None = None
        pages_fetched = 0
        complete = False
        stop_reason = ""
        max_pages = self.config.max_api_pages_per_type if paginated else 1
        for page_num in range(1, max_pages + 1):
            endpoint = urllib.parse.urljoin(api_base, collection)
            if paginated:
                endpoint += "?context=view&per_page=100&page=" + str(page_num)
            result = self.fetcher.fetch(
                endpoint,
                f"wordpress_{collection}",
                f"rest/{collection}",
                content_kind="json",
            )
            if result.status != 200 or result.error or not result.body:
                stop_reason = result.error or f"http_{result.status}"
                self._record_processing_error(
                    f"wordpress_{collection}", endpoint, "fetch_failed", detail=stop_reason, page=page_num
                )
                break
            try:
                payload = json.loads(decode_body(result.body, result.mime))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                stop_reason = "invalid_json"
                self._record_processing_error(
                    f"wordpress_{collection}", endpoint, "parse_error", detail=str(exc), page=page_num
                )
                break
            pages_fetched += 1
            if paginated:
                header_pages = result_header_int(result, "X-WP-TotalPages")
                if header_pages:
                    expected_pages = header_pages
            if isinstance(payload, list):
                rows = payload
            elif isinstance(payload, dict) and not paginated:
                rows = [payload]
            else:
                stop_reason = "unexpected_json_shape"
                self._record_processing_error(
                    f"wordpress_{collection}", endpoint, "unexpected_json_shape", page=page_num
                )
                break
            for row in rows:
                if not isinstance(row, dict):
                    continue
                record_id = row.get("id")
                if record_id is None:
                    record_key = hashlib.sha256(
                        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
                    ).hexdigest()
                else:
                    record_key = str(record_id)
                duplicate_key = (collection, record_key)
                if duplicate_key in self.wordpress_seen:
                    self.duplicate_wordpress_records += 1
                    continue
                self.wordpress_seen.add(duplicate_key)
                self.wordpress_records.append({"collection": collection, "record": row})
                if collection in {"pages", "posts"}:
                    link = row.get("link") or row.get("guid", {}).get("rendered")
                    if isinstance(link, str):
                        self._queue_url(link, endpoint, self.page_queue, self.page_seen, f"wordpress_{collection}")
                    content = row.get("content", {}).get("rendered", "")
                    for linked in self._links_from_html(content):
                        self._queue_link(linked, link or endpoint, f"wordpress_{collection}_content")
                elif collection == "media":
                    source = row.get("source_url")
                    if isinstance(source, str):
                        self._queue_url(source, endpoint, self.asset_queue, self.asset_seen, "wordpress_media")
                for value in self._record_urls(row):
                    if self._is_api_url(value):
                        continue
                    if same_site(value, self.config.base_url):
                        self._queue_link(value, endpoint, f"wordpress_{collection}_url")
                    else:
                        self._remember_external(value, endpoint)
            if not paginated:
                complete = True
                stop_reason = "metadata_object"
                break
            if expected_pages and page_num >= expected_pages:
                complete = True
                stop_reason = "header_total_pages"
                break
            if not rows:
                complete = True
                stop_reason = "empty_page"
                break
            if len(rows) < 100 and not expected_pages:
                complete = True
                stop_reason = "short_page"
                break
        else:
            stop_reason = "max_page_cap"
        if paginated and not complete and pages_fetched >= max_pages:
            self._record_processing_error(
                f"wordpress_{collection}", endpoint, "incomplete_pagination", page=pages_fetched, expected_pages=expected_pages
            )
        self.api_pagination[collection] = {
            "pages_fetched": pages_fetched,
            "expected_pages": expected_pages,
            "complete": complete,
            "stop_reason": stop_reason,
        }

    @staticmethod
    def _links_from_html(content: str) -> list[str]:
        parser = PageParser()
        parser.feed(content)
        parser.close()
        return [href for href, _ in parser.links]

    @staticmethod
    def _record_urls(value: Any) -> Iterator[str]:
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            yield value
        elif isinstance(value, Mapping):
            for nested in value.values():
                yield from Collector._record_urls(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from Collector._record_urls(nested)

    def _fetch_feeds(self) -> None:
        self._fetch_feed_url(urllib.parse.urljoin(self.config.base_url, "/feed/"))

    def _fetch_feed_url(self, feed_url: str) -> None:
        canonical_feed = canonicalize_url(feed_url, self.config.base_url)
        if not canonical_feed or canonical_feed in self.feed_seen:
            return
        self.feed_seen.add(canonical_feed)
        result = self.fetcher.fetch(canonical_feed, "rss", "rss", content_kind="feed")
        if result.status != 200 or result.error or not result.body:
            self._record_processing_error("rss", canonical_feed, "fetch_failed", detail=result.error or f"http_{result.status}")
            return
        try:
            items = parse_feed(result.body, result.mime)
        except ElementTree.ParseError as exc:
            self._record_processing_error("rss", canonical_feed, "parse_error", detail=str(exc))
            return
        for item in items:
            item["feed_url"] = canonical_feed
            link = canonicalize_url(item.get("link", ""), canonical_feed) if item.get("link") else ""
            key = link or hashlib.sha256(json.dumps(item, sort_keys=True).encode("utf-8")).hexdigest()
            if key in self.feed_item_seen:
                continue
            self.feed_item_seen.add(key)
            if link:
                item["canonical_url"] = link
            self.feed_items.append(item)
            if link:
                self._queue_url(link, canonical_feed, self.page_queue, self.page_seen, "rss_item")

    def _process_pages(self) -> None:
        while self.page_queue and self.html_count < self.config.max_html_pages:
            url, source_type = self.page_queue.popleft()
            result = self.fetcher.fetch(url, source_type, "html", content_kind="html")
            if result.status != 200 or result.error or not result.body:
                continue
            content_type = result.mime.lower()
            if content_type and not any(token in content_type for token in ("text/html", "application/xhtml", "text/plain")):
                self._record_processing_error(source_type, url, "unexpected_content_type", mime=result.mime)
                continue
            self.html_count += 1
            parsed = parse_html_document(result.body, result.mime)
            canonical = canonicalize_url(parsed.get("canonical") or result.final_url, result.final_url)
            if not canonical or not same_site(canonical, self.config.base_url):
                canonical = result.final_url
            if canonical != result.final_url:
                self._queue_url(canonical, result.final_url, self.page_queue, self.page_seen, "canonical_redirect")
            for href, tag in parsed["links"]:
                if href.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
                    continue
                self._queue_link(href, result.final_url, f"html_{tag}")
            for feed in parsed.get("feeds", []):
                feed_url = canonicalize_url(feed, result.final_url)
                if feed_url and same_site(feed_url, self.config.base_url):
                    self._fetch_feed_url(feed_url)
            self._store_document(parsed, canonical, result, source_type)

    def _store_document(self, parsed: Mapping[str, Any], canonical: str, result: FetchResult, source_type: str) -> None:
        title = str(parsed.get("title", ""))
        api_match = self._api_record_for_url(canonical)
        if api_match:
            api_title = api_match.get("title", {}).get("rendered", "")
            title = clean_text(re.sub(r"<[^>]+>", " ", str(api_title))) or title
        doc_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
        doc = {
            "doc_id": doc_id,
            "canonical_url": canonical,
            "requested_url": result.requested_url,
            "title": title,
            "text": parsed.get("text", ""),
            "sections": parsed.get("sections", []),
            "source_type": source_type,
            "html_path": result.path,
            "sha256": result.sha256,
            "fetched_at": result.fetched_at,
            "rights_status": "unreviewed-public",
        }
        if api_match:
            doc.update(
                {
                    "wordpress_id": api_match.get("id"),
                    "wordpress_type": api_match.get("type"),
                    "published": api_match.get("date"),
                    "modified": api_match.get("modified"),
                    "slug": api_match.get("slug"),
                    "categories": api_match.get("categories", []),
                    "tags": api_match.get("tags", []),
                }
            )
        self.documents[canonical] = doc

    def _store_external_text_document(
        self,
        canonical: str,
        title: str,
        parsed: Mapping[str, Any],
        result: FetchResult,
        source_type: str,
    ) -> None:
        """Store a non-WordPress seed (README/PDF) with the same schema."""

        canonical = canonicalize_url(canonical) or canonical
        document = {
            "doc_id": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20],
            "canonical_url": canonical,
            "requested_url": result.requested_url,
            "title": title,
            "text": parsed.get("text", ""),
            "sections": parsed.get("sections", []),
            "source_type": source_type,
            "html_path": result.path if source_type.endswith("html") else "",
            "raw_path": result.path,
            "sha256": result.sha256,
            "fetched_at": result.fetched_at,
            "rights_status": "unreviewed-public",
        }
        self.documents[canonical] = document

    def _api_record_for_url(self, canonical: str) -> dict[str, Any] | None:
        for row in self.wordpress_records:
            record = row["record"]
            link = record.get("link")
            if isinstance(link, str) and canonicalize_url(link) == canonical:
                return record
        return None

    def _process_assets(self) -> None:
        while self.asset_queue and self.asset_count < self.config.max_assets:
            url, source_type = self.asset_queue.popleft()
            media = url_suffix(url) in MEDIA_SUFFIXES
            result = self.fetcher.fetch(
                url,
                source_type,
                "media" if media else "assets",
                content_kind="media" if media else "asset",
            )
            if result.status == 200 and not result.error and result.body:
                self.asset_count += 1

    def _ingest_linkedin(self) -> None:
        """Ingest local, already-authorized LinkedIn MCP JSON outputs only."""

        paths = sorted(glob.glob(self.config.linkedin_input_glob))[: self.config.max_linkedin_files]
        for path_string in paths:
            path = Path(path_string)
            try:
                payload_bytes = path.read_bytes()
                if not payload_bytes or len(payload_bytes) > self.config.max_bytes:
                    raise ValueError("empty_or_oversized_payload")
                payload = json.loads(decode_body(payload_bytes, "application/json"))
            except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._record_processing_error("linkedin_mcp", path_string, "input_error", detail=str(exc))
                continue
            raw_url = payload.get("url") if isinstance(payload, dict) else ""
            if not isinstance(raw_url, str) or not raw_url:
                raw_url = f"https://www.linkedin.com/search/results/content/?source_file={path.name}"
            raw_path = self.fetcher.record_local(payload_bytes, raw_url, "linkedin_mcp_raw", "linkedin")
            post_urls = payload.get("post_urls", []) if isinstance(payload, dict) else []
            if isinstance(post_urls, list):
                for post_url in post_urls:
                    if not isinstance(post_url, str):
                        continue
                    canonical_post_url = canonicalize_url(post_url)
                    if not canonical_post_url or canonical_post_url in self.linkedin_url_seen:
                        continue
                    self.linkedin_url_seen.add(canonical_post_url)
                    self.linkedin_post_urls.append(
                        {
                            "canonical_url": canonical_post_url,
                            "source_file": path.name,
                            "raw_path": raw_path,
                            "source_type": "linkedin_mcp_url_inventory",
                            "body_available": False,
                        }
                    )
                    self._remember_external(
                        canonical_post_url,
                        f"linkedin_mcp:{path.name}",
                        source_type="linkedin_mcp_url_inventory",
                        body_available=False,
                    )
            rows = payload.get("posts", []) if isinstance(payload, dict) else payload
            if not isinstance(rows, list):
                self._record_processing_error("linkedin_mcp", path_string, "posts_not_list")
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                post_url = row.get("post_url") or row.get("url") or ""
                if not isinstance(post_url, str) or not post_url.startswith("http"):
                    post_url = ""
                text = clean_text(str(row.get("text") or ""))
                dedupe_key = post_url or hashlib.sha256(
                    f"{row.get('author','')}\n{text}".encode("utf-8")
                ).hexdigest()
                if dedupe_key in self.linkedin_seen:
                    continue
                self.linkedin_seen.add(dedupe_key)
                if len(self.linkedin_posts) >= self.config.max_linkedin_posts:
                    break
                post = dict(row)
                post.update(
                    {
                        "source_file": path.name,
                        "raw_path": raw_path,
                        "source_type": "linkedin_mcp",
                        "rights_status": "authorized-readonly-public",
                    }
                )
                if post_url:
                    post["canonical_url"] = canonicalize_url(post_url)
                    self._remember_external(post["canonical_url"], f"linkedin_mcp:{path.name}", source_type="linkedin_mcp")
                self.linkedin_posts.append(post)
                canonical = post.get("canonical_url") or f"https://www.linkedin.com/posts/{dedupe_key}"
                title = clean_text(str(row.get("headline") or row.get("author") or "LinkedIn post"))
                section = {"heading": title, "level": 1, "text": text}
                doc_id = hashlib.sha256(f"linkedin:{canonical}".encode("utf-8")).hexdigest()[:20]
                self.documents[canonical] = {
                    "doc_id": doc_id,
                    "canonical_url": canonical,
                    "requested_url": post_url or raw_url,
                    "title": title,
                    "text": text,
                    "sections": [section] if text else [],
                    "source_type": "linkedin_mcp",
                    "html_path": "",
                    "raw_path": raw_path,
                    "sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest(),
                    "fetched_at": utc_now(),
                    "rights_status": "authorized-readonly-public",
                    "author": row.get("author", ""),
                    "relative_time": row.get("relative_time", ""),
                }
        profile_payload = os.environ.get("NEXTLAB_LINKEDIN_PROFILE_JSON", "").strip()
        if profile_payload:
            try:
                profile = json.loads(profile_payload)
                if isinstance(profile, dict):
                    profile_bytes = profile_payload.encode("utf-8")
                    raw_path = self.fetcher.record_local(
                        profile_bytes,
                        str(profile.get("url") or "https://www.linkedin.com/company/asu-nextlab/"),
                        "linkedin_mcp_profile_raw",
                        "linkedin",
                    )
                    profile["raw_path"] = raw_path
                    profile["source_type"] = "linkedin_mcp_profile"
                    profile["rights_status"] = "authorized-readonly-public"
                    self.linkedin_profiles.append(profile)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._record_processing_error("linkedin_mcp_profile", "https://www.linkedin.com/company/asu-nextlab/", "input_error", detail=str(exc))

    def _fetch_related_author_sources(self) -> None:
        """Capture selected public munnerley.com author material separately."""

        related_base = "https://munnerley.com/"
        robots_url = urllib.parse.urljoin(related_base, "/robots.txt")
        self.fetcher.fetch(robots_url, "related_author_robots", "related-robots", respect_robots=False, content_kind="text", attempts=1)
        related_rows: list[dict[str, Any]] = []
        for category_id in (7, 235):
            endpoint = (
                "https://munnerley.com/wp-json/wp/v2/posts?categories="
                + str(category_id)
                + "&per_page=100&page=1"
            )
            result = self.fetcher.fetch(
                endpoint,
                "related_author_wordpress_posts",
                "related-rest",
                respect_robots=False,
                content_kind="json",
                attempts=1,
            )
            if result.status != 200 or result.error or not result.body:
                self._record_processing_error("related_author_wordpress_posts", endpoint, "fetch_failed", detail=result.error or f"http_{result.status}")
                continue
            try:
                payload = json.loads(decode_body(result.body, result.mime))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._record_processing_error("related_author_wordpress_posts", endpoint, "parse_error", detail=str(exc))
                continue
            if isinstance(payload, list):
                related_rows.extend(row for row in payload if isinstance(row, dict))
        related_seen: set[str] = set()
        related_urls: list[str] = []
        for row in related_rows:
            key = str(row.get("id") or row.get("link") or "")
            if key and key not in related_seen:
                related_seen.add(key)
                self.wordpress_records.append({"collection": "related_author_posts", "record": row})
            link = row.get("link")
            if isinstance(link, str):
                related_urls.append(link)
        try:
            source_file = Path(self.config.related_url_file)
            for line in source_file.read_text(encoding="utf-8").splitlines():
                candidate = canonicalize_url(line.strip())
                parsed = urllib.parse.urlsplit(candidate)
                if (
                    parsed.hostname == "munnerley.com"
                    and re.search(r"next-lab|learning-futures|asu|metaverse|digital-twin|edge-ai|edgeai|vr|zoom|guild|immers|future", parsed.path, re.I)
                ):
                    related_urls.append(candidate)
        except (OSError, UnicodeDecodeError):
            pass
        related_urls.append("https://munnerley.com/wp-content/uploads/2020/06/Dan-Munnerley-Portfolio1.pdf")
        seen_urls: set[str] = set()
        for url in related_urls[: self.config.max_related_urls]:
            canonical = canonicalize_url(url)
            if not canonical or canonical in seen_urls or has_sensitive_query(canonical):
                continue
            seen_urls.add(canonical)
            is_pdf = url_suffix(canonical) == ".pdf"
            result = self.fetcher.fetch(
                canonical,
                "related_author_media" if is_pdf else "related_author_html",
                "related-media" if is_pdf else "related-html",
                respect_robots=False,
                content_kind="media" if is_pdf else "html",
                attempts=1,
            )
            if result.status != 200 or result.error or not result.body:
                continue
            if is_pdf:
                self.related_asset_count += 1
                self._remember_external(canonical, "related_author_seed", source_type="related_author_media")
                continue
            self.html_count += 1
            parsed = parse_html_document(result.body, result.mime)
            canonical_doc = canonicalize_url(parsed.get("canonical") or result.final_url, result.final_url) or result.final_url
            self._store_document(parsed, canonical_doc, result, "related_author_html")
            for href, _tag in parsed.get("links", []):
                linked = canonicalize_url(href, result.final_url)
                if linked and not same_site(linked, self.config.base_url):
                    self._remember_external(linked, canonical_doc)

    def _fetch_external_seeds(self) -> None:
        if not self.config.capture_external_seeds:
            return
        for blocker in KNOWN_LEGACY_BLOCKERS:
            row = dict(blocker)
            row.update({"downloaded": False, "related_asu": "legacy_blocker", "license_status": "unknown"})
            self._remember_external(row["url"], "legacy_domain_audit", seed_label=row["label"], related_asu="legacy_blocker")
            self.external_seed_records.append(row)
        seeds = list(CURATED_EXTERNAL_SEEDS) + list(YOUTUBE_SEEDS)
        for seed in seeds:
            capture_url = seed["capture_url"]
            original_url = canonicalize_url(seed["url"])
            record: dict[str, Any] = {
                "label": seed["label"],
                "url": original_url or seed["url"],
                "capture_url": canonicalize_url(capture_url),
                "capture_kind": seed.get("capture_kind", "html"),
                "related_asu": seed.get("related_asu", "curated_public_seed"),
                "license_status": seed.get("license_status", "unknown"),
                "downloaded": False,
            }
            self._remember_external(record["url"], "curated_external_seed", seed_label=record["label"], related_asu=record["related_asu"])
            if record["capture_url"] != record["url"]:
                self._remember_external(record["capture_url"], "curated_external_capture", seed_label=record["label"])
            result = self.fetcher.fetch(
                record["capture_url"],
                "external_seed",
                "external",
                respect_robots=False,
                content_kind=record["capture_kind"],
                attempts=1,
            )
            record.update(
                {
                    "status": result.status,
                    "mime": result.mime,
                    "bytes": len(result.body),
                    "sha256": result.sha256,
                    "path": result.path,
                    "error": result.error,
                    "downloaded": bool(result.status == 200 and not result.error and result.body),
                }
            )
            capture_row = self.external_links.get(record["capture_url"])
            if capture_row is not None:
                capture_row.update(
                    {
                        "downloaded": record["downloaded"],
                        "capture_path": result.path,
                        "capture_status": result.status,
                        "capture_error": result.error,
                    }
                )
            source_row = self.external_links.get(record["url"])
            if source_row is not None:
                source_row.update(
                    {
                        "capture_url": record["capture_url"],
                        "capture_path": result.path,
                        "capture_status": result.status,
                        "capture_error": result.error,
                        "downloaded": record["downloaded"],
                    }
                )
            if seed.get("capture_kind") == "json" and result.status == 200 and not result.error and result.body:
                try:
                    payload = json.loads(decode_body(result.body, result.mime))
                    record["json_keys"] = sorted(payload)[:40] if isinstance(payload, dict) else []
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            if seed.get("capture_kind") == "html" and result.status == 200 and not result.error and result.body:
                parsed = parse_html_document(result.body, result.mime)
                canonical_doc = original_url or result.final_url
                self._store_external_text_document(canonical_doc, seed["label"], parsed, result, "external_seed_html")
                record["normalized"] = True
            if seed.get("capture_kind") == "pdf":
                if result.status == 200 and not result.error and result.path:
                    extracted, pdf_text_status = extract_pdf_text_with_status(self.run_dir / result.path)
                    if extracted:
                        parsed = parse_markdown_document(extracted, seed["label"])
                        pdf_result = FetchResult(
                            requested_url=result.requested_url,
                            final_url=result.final_url,
                            status=result.status,
                            mime=result.mime,
                            body=extracted.encode("utf-8"),
                            path=result.path,
                            sha256=result.sha256,
                            fetched_at=result.fetched_at,
                        )
                        self._store_external_text_document(original_url or result.final_url, seed["label"], parsed, pdf_result, "external_seed_pdf")
                        record["pdf_text_extracted"] = True
                    else:
                        record["pdf_text_extracted"] = False
                    record["pdf_text_status"] = pdf_text_status
                else:
                    record["pdf_text_extracted"] = False
                    if result.error:
                        record["pdf_text_status"] = f"not_downloaded:{result.error}"
                    elif result.status != 200:
                        record["pdf_text_status"] = f"not_downloaded_http_{result.status}"
                    else:
                        record["pdf_text_status"] = "not_downloaded_empty"
            if seed.get("capture_kind") == "github_repo" and result.status == 200 and not result.error and result.body:
                self._capture_github_snapshots(record, result)
            if seed.get("capture_kind") == "github_readme" and result.status == 200 and not result.error and result.body:
                self._capture_github_readme(record, result)
            self.external_seed_records.append(record)

    def _capture_github_snapshots(self, record: dict[str, Any], metadata_result: FetchResult) -> None:
        try:
            metadata = json.loads(decode_body(metadata_result.body, metadata_result.mime))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if not isinstance(metadata, dict):
            return
        owner = str(metadata.get("owner", {}).get("login") or "")
        repo = str(metadata.get("name") or "")
        branch = str(metadata.get("default_branch") or "")
        if not owner or not repo or not branch:
            return
        api_root = f"https://api.github.com/repos/{owner}/{repo}"
        tree_url = f"{api_root}/git/trees/{urllib.parse.quote(branch, safe='') }?recursive=1"
        tree_result = self.fetcher.fetch(tree_url, "github_tree", "github", respect_robots=False, content_kind="json", attempts=1)
        readme_url = f"{api_root}/readme"
        readme_result = self.fetcher.fetch(readme_url, "github_readme", "github", respect_robots=False, content_kind="json", attempts=1)
        summary: dict[str, Any] = {
            "repository": metadata.get("html_url") or record.get("url"),
            "owner": owner,
            "name": repo,
            "default_branch": branch,
            "license": (metadata.get("license") or {}).get("spdx_id") or record.get("license_status", "unknown"),
            "license_status": record.get("license_status", "unknown"),
            "metadata_path": metadata_result.path,
            "metadata_sha256": metadata_result.sha256,
            "tree_path": tree_result.path,
            "tree_sha256": tree_result.sha256,
            "readme_path": readme_result.path,
            "readme_sha256": readme_result.sha256,
            "tree_bytes": len(tree_result.body),
            "readme_bytes": len(readme_result.body),
            "tree_error": tree_result.error,
            "readme_error": readme_result.error,
        }
        if tree_result.status == 200 and not tree_result.error and tree_result.body:
            try:
                tree = json.loads(decode_body(tree_result.body, tree_result.mime))
                entries = tree.get("tree", []) if isinstance(tree, dict) else []
                summary["file_count"] = len(entries)
                summary["truncated"] = bool(tree.get("truncated")) if isinstance(tree, dict) else False
                if summary["truncated"]:
                    self._record_processing_error("github_tree", tree_url, "truncated_tree", repository=summary["repository"])
            except (json.JSONDecodeError, UnicodeDecodeError):
                summary["file_count"] = 0
        if readme_result.status == 200 and not readme_result.error and readme_result.body:
            try:
                readme = json.loads(decode_body(readme_result.body, readme_result.mime))
                if isinstance(readme, dict) and readme.get("encoding") == "base64" and readme.get("content"):
                    decoded = base64.b64decode(str(readme["content"]).encode("ascii"), validate=False)
                    summary["readme_snapshot_path"] = self.fetcher.record_local(decoded, readme_url, "github_readme_snapshot", "github-readme")
                    summary["readme_snapshot_bytes"] = len(decoded)
                    parsed = parse_markdown_document(decoded, f"{owner}/{repo} README")
                    readme_result = FetchResult(
                        requested_url=readme_url,
                        final_url=readme_result.final_url,
                        status=readme_result.status,
                        mime="text/markdown",
                        body=decoded,
                        path=summary["readme_snapshot_path"],
                        sha256=hashlib.sha256(decoded).hexdigest(),
                        fetched_at=readme_result.fetched_at,
                    )
                    self._store_external_text_document(
                        f"https://github.com/{owner}/{repo}#readme",
                        f"{owner}/{repo} README",
                        parsed,
                        readme_result,
                        "github_readme_text",
                    )
            except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
                pass
        self.github_repo_summaries.append(summary)

    def _capture_github_readme(self, record: dict[str, Any], result: FetchResult) -> None:
        try:
            payload = json.loads(decode_body(result.body, result.mime))
            if not isinstance(payload, dict) or payload.get("encoding") != "base64" or not payload.get("content"):
                return
            decoded = base64.b64decode(str(payload["content"]).encode("ascii"), validate=False)
            record["readme_snapshot_path"] = self.fetcher.record_local(decoded, result.final_url, "github_readme_snapshot", "github-readme")
            record["readme_snapshot_bytes"] = len(decoded)
            parsed = parse_markdown_document(decoded, f"{record.get('label', 'GitHub')} README")
            readme_result = FetchResult(
                requested_url=result.requested_url,
                final_url=result.final_url,
                status=result.status,
                mime="text/markdown",
                body=decoded,
                path=record["readme_snapshot_path"],
                sha256=hashlib.sha256(decoded).hexdigest(),
                fetched_at=result.fetched_at,
            )
            self._store_external_text_document(
                record.get("url") or result.final_url,
                f"{record.get('label', 'GitHub')} README",
                parsed,
                readme_result,
                "github_readme_text",
            )
        except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
            return

    def _write_normalized(self) -> tuple[int, int]:
        normalized_dir = self.run_dir / "normalized"
        normalized_dir.mkdir(parents=True, exist_ok=True)
        documents = sorted(self.documents.values(), key=lambda item: item["canonical_url"])
        documents_path = normalized_dir / "documents.jsonl"
        chunks_path = normalized_dir / "chunks.jsonl"
        chunk_count = 0
        with documents_path.open("w", encoding="utf-8") as docs_handle, chunks_path.open("w", encoding="utf-8") as chunks_handle:
            for document in documents:
                docs_handle.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")
                for section_index, section in enumerate(document.get("sections", [])):
                    for chunk_index, chunk in enumerate(chunk_text(section.get("text", ""), self.config.chunk_chars, self.config.chunk_overlap)):
                        row = {
                            "chunk_id": f"{document['doc_id']}-{section_index:03d}-{chunk_index:03d}",
                            "doc_id": document["doc_id"],
                            "canonical_url": document["canonical_url"],
                            "title": document["title"],
                            "section_heading": section.get("heading", ""),
                            "section_level": section.get("level", 0),
                            "chunk_index": chunk_index,
                            "text": chunk,
                            "source_type": document["source_type"],
                            "rights_status": document["rights_status"],
                        }
                        chunks_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        chunk_count += 1
        records_path = normalized_dir / "wordpress_records.jsonl"
        with records_path.open("w", encoding="utf-8") as handle:
            for row in self.wordpress_records:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        linkedin_path = normalized_dir / "linkedin_posts.jsonl"
        with linkedin_path.open("w", encoding="utf-8") as handle:
            for row in self.linkedin_posts:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        profile_path = normalized_dir / "linkedin_profiles.jsonl"
        with profile_path.open("w", encoding="utf-8") as handle:
            for row in self.linkedin_profiles:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        linkedin_urls_path = normalized_dir / "linkedin_post_urls.jsonl"
        with linkedin_urls_path.open("w", encoding="utf-8") as handle:
            for row in self.linkedin_post_urls:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        feeds_path = normalized_dir / "rss_items.jsonl"
        with feeds_path.open("w", encoding="utf-8") as handle:
            for item in self.feed_items:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        external_path = self.run_dir / "external-links.jsonl"
        with external_path.open("w", encoding="utf-8") as handle:
            for item in sorted(self.external_links.values(), key=lambda row: row["url"]):
                item["sources"] = sorted(set(item["sources"]))[:20]
                item.setdefault("downloaded", False)
                item.setdefault("reason", "external_link_inventory_only")
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        seeds_path = self.run_dir / "external-seeds.jsonl"
        with seeds_path.open("w", encoding="utf-8") as handle:
            for row in self.external_seed_records:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        github_path = self.run_dir / "github-repositories.jsonl"
        with github_path.open("w", encoding="utf-8") as handle:
            for row in self.github_repo_summaries:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return len(documents), chunk_count

    def _write_report(self, document_count: int, chunk_count: int) -> dict[str, Any]:
        finished = utc_now()
        all_errors = self.fetcher.errors + self.processing_errors
        ok_count = sum(
            1
            for row in self.fetcher.manifest
            if 200 <= int(row.get("status", 0)) < 400 and not row.get("error") and int(row.get("bytes", 0)) > 0
        )
        incomplete_collections = [name for name, value in self.api_pagination.items() if not value.get("complete")]
        report = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": finished,
            "base_url": self.config.base_url,
            "run_dir": str(self.run_dir),
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self.config).items()},
            "counts": {
                "manifest_records": len(self.fetcher.manifest),
                "successful_fetches": ok_count,
                "errors": len(all_errors),
                "robots_denied": sum(1 for row in self.fetcher.errors if row.get("error") == "robots_denied"),
                "html_pages": self.html_count,
                "assets_downloaded": self.asset_count,
                "related_assets_downloaded": self.related_asset_count,
                "documents": document_count,
                "chunks": chunk_count,
                "wordpress_records": len(self.wordpress_records),
                "rss_items": len(self.feed_items),
                "external_links_inventoried": len(self.external_links),
                "sitemaps": len(self.sitemap_seen),
                "linkedin_posts": len(self.linkedin_posts),
                "linkedin_post_urls": len(self.linkedin_post_urls),
                "linkedin_profiles": len(self.linkedin_profiles),
                "external_seed_attempts": len(self.external_seed_records),
                "external_seed_successes": sum(1 for row in self.external_seed_records if row.get("downloaded")),
                "github_repositories": len(self.github_repo_summaries),
                "external_html_documents": sum(
                    1 for row in self.documents.values() if row.get("source_type") == "external_seed_html"
                ),
                "github_readme_documents": sum(
                    1 for row in self.documents.values() if row.get("source_type") == "github_readme_text"
                ),
                "pdf_text_documents": sum(
                    1 for row in self.documents.values() if row.get("source_type") == "external_seed_pdf"
                ),
                "duplicate_wordpress_records": self.duplicate_wordpress_records,
                "incomplete_api_collections": len(incomplete_collections),
            },
            "bytes_downloaded": sum(
                int(row.get("bytes", 0))
                for row in self.fetcher.manifest
                if row.get("source_type") not in {"linkedin_mcp_raw", "linkedin_mcp_profile_raw", "github_readme_snapshot"}
            ),
            "bytes_ingested_local": sum(
                int(row.get("bytes", 0))
                for row in self.fetcher.manifest
                if row.get("source_type") in {"linkedin_mcp_raw", "linkedin_mcp_profile_raw", "github_readme_snapshot"}
            ),
            "bytes_by_source_type": dict(self.fetcher.bytes_by_source),
            "fetches_by_source_type": dict(self.fetcher.count_by_source),
            "errors_detail": all_errors[:200],
            "api_pagination": self.api_pagination,
            "github_repo_summaries": self.github_repo_summaries,
            "notes": [
                "robots.txt was fetched first and enforced for subsequent requests",
                "only anonymous public GET requests were made; wp-admin and authenticated/private paths were excluded",
                "external links were inventoried but not downloaded",
                "curated external seeds were fetched once each, without recursive crawling; failures are retained in external-seeds.jsonl",
                "LinkedIn data came only from local MCP JSON outputs and the explicitly authorized read-only profile payload",
                "LinkedIn search payloads without post bodies are retained as canonical URL inventory in normalized/linkedin_post_urls.jsonl",
                "WordPress API self-links were excluded from the HTML queue; successful files require nonzero bodies and matching content roles",
                "successful external HTML, GitHub README, and PDF text captures are normalized into documents.jsonl/chunks.jsonl; PDF extraction uses pdftotext first and optional pypdf fallback",
                "rights_status is a placeholder requiring human review",
            ],
        }
        (self.run_dir / "run-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        lines = [
            f"# ASU Next Lab corpus run {self.run_id}",
            "",
            f"- Base URL: {self.config.base_url}",
            f"- Started: {self.started_at}",
            f"- Finished: {finished}",
            f"- Requests recorded: {report['counts']['manifest_records']} ({ok_count} successful, {len(all_errors)} errors)",
            f"- HTML pages: {self.html_count}; assets: {self.asset_count}",
            f"- Normalized documents: {document_count}; RAG chunks: {chunk_count}",
            f"- WordPress records: {len(self.wordpress_records)}; RSS items: {len(self.feed_items)}",
            f"- External links inventoried: {len(self.external_links)}",
            f"- Bytes downloaded: {report['bytes_downloaded']:,}",
            "",
            "The full provenance manifest is in `manifest.jsonl`; `rights_status` is `unreviewed-public` pending review.",
            f"- LinkedIn URL-only inventory: {len(self.linkedin_post_urls)} canonical post URLs; body records: {len(self.linkedin_posts)}",
            f"- External normalized documents: HTML {report['counts']['external_html_documents']}; GitHub README {report['counts']['github_readme_documents']}; PDF text {report['counts']['pdf_text_documents']}",
        ]
        (self.run_dir / "run-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report

    def run(self) -> dict[str, Any]:
        LOG.info("Collecting %s into %s", self.config.base_url, self.run_dir)
        self._fetch_robots()
        self._discover_sitemaps()
        self._fetch_wordpress_api()
        self._fetch_feeds()
        self._process_pages()
        self._process_assets()
        self._fetch_related_author_sources()
        self._ingest_linkedin()
        self._fetch_external_seeds()
        document_count, chunk_count = self._write_normalized()
        self.fetcher.write_outputs()
        return self._write_report(document_count, chunk_count)


def result_header_int(result: FetchResult, key: str) -> int:
    try:
        return int(result.headers.get(key.lower(), "0"))
    except (TypeError, ValueError):
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://nextlab.asu.edu/", help="public site origin")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory receiving dated capture collections")
    parser.add_argument("--delay", type=float, default=0.25, help="minimum seconds between requests")
    parser.add_argument("--max-pages", type=int, default=1000, help="maximum HTML pages")
    parser.add_argument("--max-assets", type=int, default=750, help="maximum first-party assets")
    parser.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024, help="maximum response bytes")
    parser.add_argument("--verbose", action="store_true", help="log each high-level phase")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    config = CollectorConfig(
        base_url=args.base_url,
        output_dir=args.output_dir,
        delay_seconds=max(0.0, args.delay),
        max_html_pages=max(1, args.max_pages),
        max_assets=max(1, args.max_assets),
        max_bytes=max(1024, args.max_bytes),
    )
    try:
        report = Collector(config).run()
    except KeyboardInterrupt:
        LOG.error("Interrupted; partial files remain in the dated collection directory")
        return 130
    print(json.dumps({"run_id": report["run_id"], "run_dir": report["run_dir"], "counts": report["counts"], "bytes_downloaded": report["bytes_downloaded"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
