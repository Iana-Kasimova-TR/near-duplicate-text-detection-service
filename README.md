# Duplicate Detection Prototype

I rewired the service around a simple three-step pipeline: BM25 narrows the field, SimHash spots near duplicates fast, and sentence-transformer embeddings make the final call. Below is how everything fits together and what you can build on top of it.

## What Happens When Someone Uploads Content

1. `upload_document()` runs immediately when a draft is submitted. It:
   - normalises + tokenises the draft,
   - computes a SimHash fingerprint and an embedding *once*,
   - compares the draft against the cached corpus using **BM25 → SimHash → embeddings**.
2. If the draft survives those checks, it’s added to the in-memory cache (SimHash + embedding). BM25 is still rebuilt by default, but you can defer that by setting `recompute_index=False` and refreshing in a nightly batch.
3. Should the draft be rejected (duplicates found), nothing is persisted.

So the editor always gets real-time feedback before publishing, without waiting for a nightly sweep.

## Nightly Safety Net

Even with real-time checks, simultaneous uploads can slip through. Schedule a job that:

```python
for doc in new_docs_created_today:
    service.evaluate_document(doc_id)
    # if duplicates show up, notify editors right away
```

At the end of the run, send a notification (email/Slack/etc.) summarising the flagged pairs for manual review.

## Current Internals

- **BM25 gate** – built once on startup; keeps candidate lists short. Rebuild after bulk ingests or during the nightly job.
- **SimHash screening** – fingerprints cached per document; draft vs. candidate comparison is just XOR + popcount.
- **Embedding confirmation** – embeddings cached as well; cosine similarity via `Embedder.cosine_similarity` decides the final subset.
- **Feedback capture** – call `DuplicateDetectionService.record_feedback()` to persist reviewer decisions in a SQLite database (see `duplicate_detection/feedback.py`). Example:

```python
service.record_feedback(
    database_url="sqlite:///data/feedback.db",
    document_id="1031",
    duplicate_id="88",
    verdict="confirm",
    notes="Same onboarding checklist, keep the older one"
)
```

That keeps a growing dataset you can use to tune thresholds or train a classifier later.

## Environment Setup (via uv)

The project now ships with a `pyproject.toml`; you can use [uv](https://docs.astral.sh/uv/) to manage the environment:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

That installs all dependencies defined under `[project.dependencies]`.

## Running the API & Tools

Spin up the FastAPI service locally:

```bash
uvicorn duplicate_detection.api.app:app --reload
```

…or via Docker Compose:

```bash
docker compose -f docker/docker-compose.yml up --build
```

Key endpoints:

- `POST /duplicates/evaluate` – returns candidates for a draft
- `POST /duplicates/upload` – evaluate + ingest when no duplicate is found
- `POST /duplicates/feedback` – store reviewer verdicts (`document_id`, `duplicate_id`, `verdict`, optional `notes`)
- `GET /health` – simple liveness check

CLI demo now talks to the API:

```bash
python scripts/run_demo.py \
    --query-file docs/sample.txt \
    --top 3 \
    --api-url http://localhost:8000
```

Submission helper (works directly on local files while you transition):

```bash
python scripts/generate_submission.py \
    /path/to/all_docs \
    docs/submission.csv \
    --bm25-top-k 200 \
    --max-candidates 50 \
    --log-level INFO
```

## Ideas for the Next Iteration

- **Token-aware SimHash** → Dig into the paper’s MonoActive algorithm (weighted Jaccard via improved consistent weighted sampling). It’s more sensitive to token importance than vanilla SimHash - https://arxiv.org/pdf/2509.00627
- **Category-specific embeddings** → Let uploaders specify a document category, then pick (or fine-tune) the best embedding model for that domain. HR policy might use a different encoder from engineering handbooks.
- **GitHub-style diffing** → When duplicates are found, render the documents side-by-side with inline edits and let reviewers merge/update directly.
- **Chunk summarizing** → we have documents with wildly different lengths; splitting oversized pages into coherent chunks (or auto-summaries) keeps embedding comparisons fair and highlights partial overlap instead of drowning everything in a giant vector.
- **Feedback loop** → We already capture `record_feedback`; wire that into your analytics, retrain thresholds, or build a small classifier to predict reviewer decisions.
- **Nightly notifications** → Hook the batch job into your messaging stack so editors get a curated list of conflicts each morning.
- **Persisted indexes** → Move embeddings to pgvector (or another vector DB) and SimHash/BM25 metadata to Postgres/Redis, then Dockerise the whole stack (`app`, `postgres+pgvector`, optional worker containers) for production.
- **Partially duplicated** → Also it is really important to consider the case when documents are partially duplicated as it can also has bad influence on RAG performance

## Key points
- **User UI experience** → For better user UI experience I propose to show the duplication and difference as in we can see it when compare commits from Github. Also Notification to the user should be very soft, as we are not sure on 100 percent that is total duplicate
- **Duplicates brings RAG app to downgrade in performance** → for RAG it is really crucial to avoid duplicates, as we will get from retrieval top k documents and these position can be busy by the same documents(as we have duplicates there)


## Struggle

Running BM25 over entire documents froze the submission script because every query reprocessed the full corpus; switching to chunk-level indexing, adopting `bm25s`, and reusing chunk embeddings finally made the pipeline responsive enough for near real-time checks.

## Possible Solution

- Break documents into chunks and index those chunks with `bm25s` so retrieval stays bounded and latency-friendly.
- Cache the chunk embeddings alongside document vectors to reuse them during near real-time duplicate checks.
- When upload volume spikes, buffer incoming chunks and trigger a batched rebuild once the queue crosses a threshold or a timed window, serving queries from the last full index plus the buffer until the refreshed index is swapped in.

## File Map

```
.
├── scripts/
│   ├── run_demo.py           # calls FastAPI evaluate endpoint
│   └── generate_submission.py# Kaggle submission helper
├── docker/
│   ├── app/Dockerfile        # container for FastAPI service
│   └── docker-compose.yml    # compose setup
└── src/duplicate_detection/
    ├── api/app.py            # FastAPI service exposing the pipeline
    ├── service.py            # BM25 → SimHash → embedding orchestration
    ├── feedback.py           # tiny SQLite helper for reviewer feedback
    ├── embedder.py           # sentence-transformer wrapper
    ├── bm25_index.py         # BM25 candidate generator
    ├── loader.py             # load JSONL/CSV/dir datasets
    ├── preprocess.py         # normalisation + tokenisation
    └── models.py             # dataclasses & config knobs
```
