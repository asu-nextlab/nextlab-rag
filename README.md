# NextLab RAG tools

Public code for the Next Lab collector and local RAG prototype. The reviewed
corpus is private at `satyalyadav/nextlab-corpus`; access to that Hugging Face
dataset is required. Public access to source pages does not grant reuse rights.
Check each record's rights and license fields. No blanket reuse license is
provided.

The current export has 274 documents and 4,407 chunks. Indexing splits those
chunks into 5,452 passages of at most 384 tokenizer tokens.

## Quickstart

Create an environment, install the pinned packages, and sign in with an account
that can access the private dataset:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-rag.txt -r requirements-dev.txt
hf auth login
hf download satyalyadav/nextlab-corpus --repo-type dataset --local-dir . \
  --include 'dataset/**' --include '.rag/current.json' \
  --include '.rag/indexes/**' --max-workers 1
python -m nextlab_rag search 'What is ASUniverse?'
```

The private bundle includes a compact, roughly 27 MB Qdrant index. Search uses
it when its corpus, model, schema, and pinned library signatures match. The
first query may download model files into ignored `.rag/cache`.

## Verify source provenance

This optional check downloads the saved source files and the historical inputs
referenced by the active export. The full source store is about 791 MB.

```bash
hf download satyalyadav/nextlab-corpus --repo-type dataset --local-dir . \
  --include 'dataset/**' --include 'sources/**' \
  --include 'history/path-map.jsonl' --include 'history/collections/**' \
  --include '.rag/current.json' --include '.rag/indexes/**' --max-workers 1
python -m nextlab_scraper.review_export --verify-only --repo-root . --export-dir dataset
```

## Rebuild the index

If search reports stale signatures, build or resume the local CPU index:

```bash
./scripts/index-capped.sh
```

The runner needs user systemd. It limits memory to 2 GB, disables swap, uses
one CPU thread, stops after five minutes without a checkpoint, and has a
90-minute deadline. Systemd stops the service on OOM. To stop an active run:

```bash
systemctl --user stop nextlab-rag-memory-rebuild
```

`ask` needs an explicit chat-completions endpoint and model. Set
`NEXTLAB_RAG_BASE_URL` and `NEXTLAB_RAG_MODEL`; set `NEXTLAB_RAG_API_KEY` when
the endpoint requires bearer authentication. The CLI does not call a default
hosted service.

Run the tests with `python -m pytest -q`. The collector uses Python's standard
library. PDF extraction tries `pdftotext` first and can use optional `pypdf`
as a fallback.

Transcript and OCR text may be sampled or incomplete. Preserve source-specific
completeness notes and rights fields with derived results.
