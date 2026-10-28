import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from nextlab_scraper.collector import (
    CollectorConfig,
    build_arg_parser,
    canonicalize_url,
    chunk_text,
    extract_pdf_text_with_status,
    has_sensitive_query,
    parse_feed,
    parse_html_document,
    parse_sitemap,
)


class CollectorLayoutTests(unittest.TestCase):
    def test_default_capture_location_is_sources_collections(self):
        self.assertEqual(CollectorConfig().output_dir, Path("sources/collections"))
        self.assertEqual(build_arg_parser().parse_args([]).output_dir, Path("sources/collections"))


class CanonicalizationTests(unittest.TestCase):
    def test_removes_fragments_tracking_and_default_port(self):
        value = canonicalize_url(
            "HTTPS://NextLab.ASU.edu:443/a/../about/?utm_source=x&b=2#section",
        )
        self.assertEqual(value, "https://nextlab.asu.edu/about/?b=2")

    def test_sensitive_query_is_refused(self):
        self.assertTrue(has_sensitive_query("https://nextlab.asu.edu/file.pdf?signature=abc"))
        self.assertFalse(has_sensitive_query("https://nextlab.asu.edu/search?page=2"))


class ParsingTests(unittest.TestCase):
    def test_sitemap_urlset(self):
        kind, rows = parse_sitemap(
            b'''<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://nextlab.asu.edu/</loc><lastmod>2025-01-01</lastmod></url></urlset>'''
        )
        self.assertEqual(kind, "urlset")
        self.assertEqual(rows[0]["loc"], "https://nextlab.asu.edu/")
        self.assertEqual(rows[0]["lastmod"], "2025-01-01")

    def test_html_canonical_sections_and_links(self):
        parsed = parse_html_document(
            b'''<html><head><title> Next Lab </title><link rel="canonical" href="/about/"></head><body><nav>Menu</nav><h1>About</h1><p>Public <b>text</b>.</p><h2>Guilds</h2><p>Build things.</p><img src="/wp-content/uploads/x.pdf"><script>secret()</script></body></html>'''
        )
        self.assertEqual(parsed["title"], "Next Lab")
        self.assertEqual(parsed["canonical"], "/about/")
        self.assertIn(("/wp-content/uploads/x.pdf", "img"), parsed["links"])
        self.assertNotIn("Menu", parsed["text"])
        self.assertIn("Build things.", parsed["text"])
        self.assertEqual([section["heading"] for section in parsed["sections"]], ["", "About", "Guilds"])

    def test_feed_items(self):
        rows = parse_feed(b'''<rss><channel><item><title>News</title><link>https://nextlab.asu.edu/news/</link></item></channel></rss>''')
        self.assertEqual(rows[0]["title"], "News")
        self.assertEqual(rows[0]["link"], "https://nextlab.asu.edu/news/")


class ChunkTests(unittest.TestCase):
    def test_chunks_are_bounded_and_nonempty(self):
        chunks = chunk_text("one two three four five six seven eight nine ten", max_chars=18, overlap=3)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk and len(chunk) <= 18 for chunk in chunks))


class PdfTests(unittest.TestCase):
    def test_pdf_extraction_reports_a_bounded_status_on_invalid_input(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "not-a-pdf.pdf"
            path.write_bytes(b"not a PDF")
            text, status = extract_pdf_text_with_status(path)
        self.assertIsNone(text)
        self.assertTrue(status)


if __name__ == "__main__":
    unittest.main()
