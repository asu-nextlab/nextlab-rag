# Evaluation set and retrieval experiments

The 100-question evaluation set is derived from the private Next Lab corpus. It is not stored in this public repository. Restore it from the private Hugging Face bundle into `eval/` before running the scripts. `eval/` is gitignored.

Expected files: `eval/questions.jsonl`, `eval/candidates.jsonl`, `eval/manifest.json`.

The set has 100 questions and a 70/30 dev-test split:

| Type | Count |
| --- | ---: |
| single-hop | 30 |
| multi-hop | 24 |
| transcript | 10 |
| social | 5 |
| timeline | 16 |
| unanswerable | 10 |
| incomplete capture | 5 |

There is no conflict slice. The review found no clean factual contradictions in the corpus. Ten weak items were replaced rather than kept.

## Retrieval results

90 answerable questions, Qdrant local, top 10 unless noted.

| Configuration | hit@5 | hit@10 | MRR | multi-hop both@10 |
| --- | ---: | ---: | ---: | ---: |
| bge-small baseline | 81.1% | 92.2% | 0.622 | 50.0% |
| bge-small + reranker | 86.7% | 90.0% | 0.674 | 54.2% |
| bge-small + expansion | 81.1% | 93.3% | 0.621 | 50.0% |
| bge-small + reranker + expansion | 87.8% | 91.1% | 0.685 | 54.2% |
| bge-base baseline | 84.4% | 88.9% | 0.610 | 41.7% |
| bge-base + reranker | 85.6% | 86.7% | 0.674 | 54.2% |
| sentence-aware bge-small | 82.2% | 91.1% | 0.628 | 50.0% |
| sentence-aware bge-small + reranker | 85.6% | 87.8% | 0.720 | 41.7% |

## What the results mean

- **Use top 10, not top 5.** Coverage rises from 81.1% to 92.2% with no model change.
- **Do not switch to bge-base.** It gains 3.3 points at top 5, loses 3.3 at top 10, and drops multi-hop both-chunks from 50.0% to 41.7%. The published MTEB gain does not transfer.
- **Do not run Qwen3-Embedding.** A much cheaper model swap did not help.
- **The reranker is optional.** It improves top-5 ordering and MRR, but the gain is not statistically significant on 90 questions (McNemar p=0.18), and it can push a gold passage out of the top 10.
- **Acronym expansion is neutral.** One extra top-10 hit, no top-5 change.
- **Sentence-aware chunking is not worth it.** It increased the index from 5,452 to 8,299 passages, did not improve coverage, and hurt multi-hop with reranking.
- **Parent expansion is untested.** It does not change hit@k, so it needs an answer-quality comparison: same passages, with and without the enclosing chunk text in the prompt.

## Reproducing

```bash
# Rebuild the candidate pool from the private corpus
python3 scripts/build_eval_candidates.py

# Check schema, quotas, quotes, and duplicates
python3 scripts/validate_eval.py --profile final eval/questions.jsonl

# Retrieval hit@k on the active index
.venv-pdf/bin/python scripts/eval_retrieval.py

# Compare reranking and BM25 expansion
.venv-pdf/bin/python scripts/retrieval_experiments.py --name baseline
```

The experiment scripts write results under `eval/experiments/`, which is ignored with `eval/`.
