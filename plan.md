# Reddit IR — Final Project Plan
**Course:** 601.466/666 Information Retrieval  
**Project:** Reddit Thread Retrieval & Ranking Engine  

---

## 1. Project Overview

Build a search engine over a scraped Reddit corpus. Users enter a free-text query and receive ranked results from a multi-subreddit dataset. The system implements two retrieval models — TF-IDF cosine similarity and BM25 — and evaluates them head-to-head using standard IR metrics.

---

## 2. System Architecture

The pipeline flows through five stages:

1. **Data collection** — PRAW scrapes posts and top comments from selected subreddits, saves raw JSON to disk.
2. **Preprocessing** — Tokenization, lowercasing, stopword removal (including Reddit-specific terms like "upvote", "tldr", "edit"), and Porter stemming.
3. **Indexing** — An inverted index maps terms to document IDs and frequencies. Document frequencies are computed for IDF weighting. Index is persisted to disk.
4. **Retrieval & ranking** — Two rankers score documents against a query: TF-IDF with cosine similarity (via scikit-learn) and BM25 implemented from scratch in NumPy.
5. **Query interface** — A Flask web UI lets users enter queries, toggle between rankers, and browse ranked results with title, subreddit, score, snippet, and a link to the original post.

---

## 3. Tech Stack

| Layer | Tool | Reason |
|---|---|---|
| Data collection | PRAW (Python Reddit API Wrapper) | Official wrapper, handles auth and rate limits |
| Storage | SQLite + JSON flat files | Simple, no server needed, easy to submit |
| Preprocessing | NLTK | Stopwords, Porter stemmer, tokenizer — matches course HW tools |
| TF-IDF | scikit-learn TfidfVectorizer | Fast sparse matrix ops |
| BM25 | Python + NumPy (from scratch) | Demonstrates IR knowledge beyond library calls |
| Evaluation | ir-measures + custom scripts | Compute P@k, NDCG, MAP |
| Web interface | Flask + Jinja2 | Lightweight, runs locally for demo and grading |
| Writeup | PDF (LaTeX or Word) | Required submission format |

---

## 3.1 Proposed Repository Layout

```text
finalProject/
  data/
    raw/                # raw subreddit JSON dumps
    processed/          # cleaned docs + token lists
    qrels/              # query relevance labels
  src/
    crawler/
      __init__.py
      config.py
      client.py
      collectors.py
      filters.py
      serializers.py
      checkpoint.py
      runner.py
      cli.py
    preprocess.py
    index.py
    rankers/
      tfidf_ranker.py
      bm25_ranker.py
    evaluate.py
    app.py
    utils.py
  notebooks/            # optional exploratory analysis
  tests/                # unit tests for tokenizer/ranker/eval
  requirements.txt
  README.md
  plan.md
```

Keeping this layout from the beginning helps avoid last-week refactors and makes packaging for submission straightforward.

---

## 3.1.1 Initial Build Order (Structure-First)

To keep development clean, implement in this order:

1. `src/crawler/` package + `data/raw/` (crawler foundation)
2. `src/preprocess.py` + `data/processed/`
3. `src/index.py` + serialized index artifacts
4. `src/rankers/tfidf_ranker.py` and `src/rankers/bm25_ranker.py`
5. `src/evaluate.py` + `data/qrels/`
6. `src/app.py` (Flask UI) only after retrieval is stable

This prevents UI work from blocking core IR progress.

---

## 3.2 Reuse from Homework (What to Port)

Based on existing implementations in `hw2`/`hw3`/`hw4`, the project should explicitly reuse:

- **Field/region weighting (hw2):** score title/body/comments with different weights instead of flattening everything equally.
- **BM25-style TF saturation variants (hw2):** include a tunable saturation component and compare against plain TF-IDF.
- **Negation-aware query weighting (hw2):** reduce scores for terms following patterns like "not X", "without Y", "exclude Z".
- **Local collocation/proximity features (hw3):** add lightweight phrase-neighborhood signals for short/specific queries.
- **Robust parsing mindset (hw4):** keep fault-tolerant data handling and normalization when processing noisy web text.

Porting these avoids re-inventing solved components and directly improves both effectiveness and robustness.

---

## 4. Modules

### M1 — Reddit Crawler Package
- Connect to Reddit via PRAW using OAuth credentials
- Select 3–5 subreddits with distinct vocabularies (e.g. r/science, r/programming, r/cooking, r/AskHistorians)
- Fetch 5,000–10,000 posts including title, body, top comments, score, subreddit, and URL
- Handle pagination and API rate limits
- Save raw data as JSON files, one file per subreddit

### M1.1 — Crawler Specification (Detailed, First Implementation Target)

**Goal**
- Collect a reproducible, balanced Reddit corpus across selected subreddits for downstream indexing/ranking.

**Input configuration**
- `subreddits`: list of subreddit names (3-5)
- `post_limit_per_subreddit`: integer target (e.g., 2000 each)
- `comment_limit_per_post`: integer (e.g., top 5)
- `time_filter`: one of `day/week/month/year/all`
- `sort_mode`: one of `hot/new/top`
- `output_dir`: default `data/raw/`
- `resume`: boolean flag to continue from prior run

**Package structure (DRY + robust)**
- `config.py`
  - Typed config dataclass and env/CLI merge logic
  - Central place for rate, retry, and crawl parameters
- `client.py`
  - PRAW client creation and auth validation
  - Single API access layer (no raw PRAW calls outside this module)
- `collectors.py`
  - Subreddit/post/comment fetch logic
  - Iterator-style generators for memory-safe batching
- `filters.py`
  - Content quality filters (`deleted`, score threshold, NSFW policy, language if needed)
- `serializers.py`
  - JSON schema enforcement, normalization, and JSONL writing
- `checkpoint.py`
  - Resume state (`seen_post_ids`, per-subreddit progress, cursor/time window)
- `runner.py`
  - Orchestrates crawl loop, retries, sleep, and failure handling
- `cli.py`
  - CLI argument parsing and invocation of `runner.run()`

**Output files**
- `data/raw/<subreddit>.jsonl` (one JSON object per post)
- `data/raw/crawl_manifest.json` (run metadata + counters)
- Optional: `data/raw/errors.log` (failed fetches/post IDs)

**Per-document JSON schema**
```json
{
  "doc_id": "science_t3_abc123",
  "post_id": "abc123",
  "subreddit": "science",
  "title": "post title",
  "selftext": "body text",
  "top_comments": ["comment 1", "comment 2"],
  "url": "https://www.reddit.com/...",
  "permalink": "/r/science/comments/abc123/...",
  "created_utc": 1714195200,
  "score": 1532,
  "num_comments": 147,
  "over_18": false,
  "is_self": true,
  "retrieved_at": "2026-04-27T05:00:00Z"
}
```

**Crawling strategy**
- Use PRAW listing API (`subreddit.top/new/hot`) with explicit limits.
- For each post:
  - Skip removed/deleted stubs with empty `title` and `selftext`.
  - Call `replace_more(limit=0)` before reading comments.
  - Keep only top-N comments ranked by Reddit ordering in the listing.
- Balance corpus size by enforcing per-subreddit caps.

**Anti-crawler / anti-abuse resilience strategy**
- Respect authenticated API usage only (no HTML scraping).
- Enforce conservative request pacing:
  - fixed base delay + random jitter between pull batches
  - per-subreddit cool-down when 429/rate warnings appear
- Use bounded retries with exponential backoff and retryable error classification.
- Implement circuit-breaker behavior:
  - if repeated failures exceed threshold, pause subreddit crawl and move to next one.
- Persist checkpoint every N saved posts (e.g., every 50) to avoid losing progress on interruption.
- Add run-level idempotency:
  - repeated runs with `--resume` safely skip existing `post_id`s.

**Deduplication policy**
- Primary key: `post_id`
- If a post already exists in output, skip unless `--refresh` is set.
- Keep a `seen_post_ids` set loaded from existing JSONL when `resume=True`.

**Rate-limit and fault tolerance**
- Honor PRAW internal rate-limit handling; never bypass API controls.
- Retry transient API failures with exponential backoff (e.g., 1s, 2s, 4s; max 3 tries).
- Add jitter to retries to avoid synchronized retry bursts.
- On permanent failure, log post/subreddit and continue (never abort whole crawl).
- Emit structured error categories in logs: `auth`, `rate_limit`, `network`, `parse`, `unknown`.

**Data quality rules**
- Normalize whitespace in text fields.
- Preserve raw punctuation/case in raw files (normalization happens in preprocessing).
- Enforce UTF-8 writing and escape invalid characters safely.

**CLI contract (required)**
- `python -m src.crawler.cli --subreddits science programming cooking --limit 2000 --comments 5 --sort top --time month`
- Optional flags:
  - `--resume`
  - `--output data/raw`
  - `--min-score <int>` (optional filtering)
  - `--refresh` (re-fetch existing IDs)
  - `--batch-sleep-ms <int>` and `--jitter-ms <int>`
  - `--max-retries <int>` and `--checkpoint-every <int>`

**Crawler completion criteria**
- At least 5,000 total posts collected with all required fields non-null where applicable.
- `crawl_manifest.json` records:
  - total posts attempted/saved/skipped
  - per-subreddit counts
  - run start/end timestamps
  - crawler parameters used
- Re-running with `--resume` does not duplicate records.

**Testing strategy for crawler package**
- Unit tests for:
  - schema serialization
  - dedup/checkpoint resume
  - retry/backoff classification logic
- Integration smoke test:
  - crawl 1 subreddit with small limits (e.g., 20 posts, 2 comments/post)
  - verify JSONL + manifest + no duplicates after rerun

### M2 — Preprocessor
- Tokenize text using NLTK word tokenizer
- Lowercase all tokens
- Remove standard English stopwords plus a custom Reddit-specific list ("upvote", "downvote", "edit", "tldr", "op", "oc", etc.)
- Apply Porter stemmer
- Output clean token lists per document, stored alongside metadata
- Recommended combined text field for indexing:
  - `full_text = title + " " + selftext + " " + top_comments`
- Store document length after preprocessing (needed for BM25 length normalization)

### M3 — Inverted Index Builder
- Build a term → {doc_id: term_frequency} mapping from the preprocessed corpus
- Compute document frequency (DF) for every term
- Compute corpus-level statistics: total document count, average document length
- Persist the index to disk using pickle or shelve for fast reload at query time
- Persist a metadata table with:
  - `doc_id`, `title`, `subreddit`, `url`, `raw_score`, `doc_len`, `raw_text`

### M4 — TF-IDF Ranker
- Use scikit-learn's TfidfVectorizer to build a TF-IDF document-term matrix
- At query time, vectorize the query using the same vocabulary and IDF weights
- Compute cosine similarity between query vector and all document vectors
- Return ranked list of top-k documents with scores

### M5 — BM25 Ranker (from scratch)
- Implement BM25 scoring formula manually using NumPy
- Tunable parameters: k1 (term saturation, default 1.5) and b (length normalization, default 0.75)
- Use document frequencies from M3 for IDF component
- At query time, score each document containing at least one query term
- Return ranked list of top-k documents with scores
- This is the key differentiator — implementing BM25 from scratch rather than using a library demonstrates core IR understanding
- Suggested scoring formula:
  - `IDF(t) = log((N - df_t + 0.5) / (df_t + 0.5) + 1)`
  - `score(D, Q) = sum_{t in Q} IDF(t) * ((tf_{t,D} * (k1 + 1)) / (tf_{t,D} + k1 * (1 - b + b * |D|/avgdl)))`
- Add field-aware BM25 option:
  - `score(D, Q) = w_title * BM25(title, Q) + w_body * BM25(body, Q) + w_comments * BM25(comments, Q)`
  - Start with `w_title=2.0, w_body=1.0, w_comments=1.2`, then tune on dev queries

### M6 — Flask Query Interface
- Simple web UI with a search bar and ranker toggle (TF-IDF / BM25)
- Result cards showing: post title, subreddit, retrieval score, text snippet, link to original Reddit post
- Pagination for top-k results (default k=10)
- Optional: side-by-side comparison mode showing both rankers' results for the same query

### M7 — Evaluation Harness
- Define 15–20 test queries covering a range of topics and difficulty levels
- Before running the system, manually label the top-10 results per query as relevant (1) or not relevant (0)
- Compute the following metrics comparing TF-IDF vs BM25:
  - Precision@5 and Precision@10
  - NDCG@10 (Normalized Discounted Cumulative Gain)
  - MAP (Mean Average Precision)
- Produce a results table for the writeup
- **Important:** Label relevance judgments before running the system to avoid post-hoc bias (as required by the assignment)

### M8 — Writeup & Packaging
- Cover page: team names, section (466/666), email, 1–3 line project summary
- User guide: how to install dependencies, run the scraper, build the index, and launch the Flask app
- Achievements list: what the system does well, interesting design decisions
- Limitations and suggested extensions
- Screenshots: query interface, result cards, side-by-side ranker comparison
- Evaluation section: metric tables, discussion of TF-IDF vs BM25 differences
- Package all code and data into a named zip for Gradescope submission

---

## 5. Timeline

### Week 1 — Data & Preprocessing
- Set up Reddit API credentials and PRAW environment
- Choose 3–5 subreddits and run M1 scraper to collect 5,000–10,000 posts
- Build and test M2 preprocessor (tokenizer, stopword removal, stemming)
- Verify output: inspect token distributions, check stopword list coverage

### Week 2 — Indexing & Retrieval
- Build M3 inverted index and confirm it persists and reloads correctly
- Implement M4 TF-IDF ranker and run sanity-check queries
- Implement M5 BM25 ranker from scratch, validate scores against expected behavior
- Port field weighting and BM25 TF saturation ideas from homework code
- Run both rankers on a handful of informal test queries, compare results

### Week 3 — Interface & Evaluation
- Build M6 Flask UI with result cards and ranker toggle
- Add snippet highlighting and optional score breakdown per result (title/body/comments contributions)
- Write 15–20 test queries and hand-label relevance judgments (do this before running eval)
- Run M7 evaluation harness, compute P@5, P@10, NDCG@10, MAP for both rankers
- Analyze where TF-IDF and BM25 differ and why

### Week 4 — Polish & Submission
- Polish the Flask UI, fix edge cases, take screenshots
- Add one advanced feature if time permits (query expansion OR result clustering)
- Write full PDF writeup covering all required sections (M8)
- Package code + data + writeup into a named zip file
- Submit to Gradescope

---

## 5.1 Weekly Deliverables (Concrete Checkpoints)

- **End of Week 1:** at least 5,000 cleaned documents serialized, plus token stats report (top terms, avg length)
- **End of Week 2:** both rankers return top-10 results for the same query format; latency under 1s/query on local machine for 10k docs; field-aware BM25 baseline runs end-to-end
- **End of Week 3:** evaluation script outputs a CSV table with P@5, P@10, NDCG@10, MAP for each ranker, plus an ablation table (plain BM25 vs field-aware BM25)
- **End of Week 4:** full submission zip generated and validated from a clean environment using README steps only

---

## 6. Evaluation Plan

The evaluation compares TF-IDF cosine similarity against BM25 on 15–20 hand-labeled queries.

**Metrics:**
- **P@5 / P@10** — fraction of top-5 and top-10 results that are relevant
- **NDCG@10** — accounts for rank position, penalizes relevant results ranked lower
- **MAP** — mean average precision across all queries, overall system quality

**Process:**
1. Write queries before labeling to avoid selection bias
2. Label top-10 results per query independently before running the retrieval system
3. Run both rankers on all queries
4. Compute metrics using ir-measures or a custom script
5. Present results in a table in the writeup and discuss which ranker performs better and under what conditions

---

## 6.1 Query Set Design Guidelines

To avoid cherry-picking and to produce an informative analysis, include:
- 5 broad informational queries (e.g., "climate change evidence")
- 5 technical/specific queries (e.g., "python async websocket bug")
- 5 opinion/discussion-style queries (e.g., "best beginner cooking knife")
- Optional 2–5 ambiguous or short queries to stress-test both rankers

Keep this query list fixed before evaluation and include it in the appendix/writeup.

---

## 6.2 Ablation and Tuning Plan

To show clear technical contribution, run controlled comparisons:

1. **Baseline A:** TF-IDF cosine
2. **Baseline B:** plain BM25
3. **Variant C:** field-aware BM25
4. **Variant D (optional):** field-aware BM25 + query expansion or proximity boost

For each variant, report:
- P@5, P@10, NDCG@10, MAP
- Mean query latency
- Short failure analysis on 3 representative queries

Tune only a small set of parameters to avoid overfitting:
- BM25: `k1 in {1.2, 1.5, 1.8}`, `b in {0.5, 0.75, 0.9}`
- Field weights: a few fixed candidates (e.g., `(2.0,1.0,1.2)`, `(3.0,1.0,1.0)`, `(2.5,1.0,1.5)`)

---

## 7. Potential Extensions (if time allows)

- **Query expansion** using pseudo-relevance feedback (top-k results used to expand the query)
- **Region weighting** giving extra weight to post titles vs body text vs comments
- **Faceted filtering** by subreddit or post score in the UI
- **Snippet generation** highlighting matched query terms in the result snippet

---

## 7.1 Risks and Mitigations

- **API/rate-limit risk:** cache raw pulls and support resumable scraping by subreddit + after token.
- **Noisy Reddit text:** include lightweight normalization for URLs, markdown artifacts, and deleted/removed content.
- **Evaluation bias:** freeze queries and relevance labels before metric runs; keep labels in versioned files.
- **Time overrun:** treat BM25 + evaluation as non-negotiable core; side-by-side UI and extensions are optional.

---

## 8. Submission Checklist

- [ ] PDF writeup (all 7 required sections)
- [ ] Complete code archive (scraper, preprocessor, indexer, rankers, Flask app, eval scripts)
- [ ] Raw and processed data files
- [ ] Named zip: `firstname_lastname.zip` (or partner names)
- [ ] Submitted via Gradescope
- [ ] README tested from a fresh virtual environment (`pip install -r requirements.txt`)
- [ ] `requirements.txt` and reproducible run commands included
