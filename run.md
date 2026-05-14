### csv → raw JSONL (when using bundled `dataset/*.csv` only)

python scripts/csv_to_raw_jsonl.py --input-dir dataset --output-dir data/raw

### crawler

python -m src.crawler.cli \
  --subreddits science programming cooking AskHistorians \
  --limit 1000 --comments 5 --sort top --time month \
  --output data/raw

### preprocess

python -m src.preprocess.cli \
  --input data/raw \
  --output data/processed \
  --stemmer snowball

### index

python -m src.index.cli build \
  --input data/processed \
  --output data/index

python -m src.index.cli stats --index-dir data/index

### search

# Ranker ids — TF-IDF: tfidf_cosine, tfidf_dice, tfidf_jaccard
#                BM25: bm25_plain, bm25_field, bm25_prf
#                Two-stage: twostage_bm25_tfidf
#                LSI: lsi (needs: pip install scipy numpy; SVD at ranker init can take tens of seconds on a large index)
# Optional ablation only: tfidf_overlap (not in default eval matrix; pass --rankers tfidf_overlap)
# Legacy aliases: bm25 → bm25_plain, tfidf → tfidf_cosine

python -m src.rankers.cli search --index-dir data/index --query "python async websocket" --ranker bm25_plain
python -m src.rankers.cli search --index-dir data/index --query "python ml" --ranker lsi --lsi-k 80 --lsi-max-vocab 6000
python -m src.rankers.cli search --index-dir data/index --query "python ml" --ranker bm25_prf --prf-depth 5 --prf-expand 12
python -m src.rankers.cli search --index-dir data/index --query "python guide" --ranker tfidf_cosine --top-k 5
python -m src.rankers.cli search --index-dir data/index --query "python guide" --ranker tfidf_dice --top-k 5
python -m src.rankers.cli search --index-dir data/index --query "climate" --ranker bm25_field
python -m src.rankers.cli search --index-dir data/index --ranker twostage_bm25_tfidf --stage1-k 300
python -m src.rankers.cli search --index-dir data/index --ranker bm25_plain --field-aware    # same as bm25_field
python -m src.rankers.cli search --index-dir data/index    # interactive (default ranker: bm25_plain)

### evaluate

# 1. edit qrel/queries.json with 15-20 queries (or use the bundled file):

# {"queries": [{"qid": "q1", "text": "..."}, ...]}

# 2. generate pool template (defaults: pool -> qrel/pool.csv, run outputs -> qrel/)

python -m src.evaluate.cli pool \
  --queries qrel/queries.json \
  --index-dir data/index \
  --output qrel/pool.csv \
  --depth 20

# 3. open qrel/pool.csv, fill the 'label' column (1=relevant, blank/0=not),

# save as qrel/qrels.csv (or copy pool.csv to qrels.csv after labeling).

# 4. compute metrics for all rankers (default matrix: no tfidf_overlap; includes lsi, bm25_prf)

python -m src.evaluate.cli run \
  --queries qrel/queries.json \
  --qrels qrel/qrels.csv \
  --index-dir data/index \
  --output-dir qrel

# Summary table is written to qrel/summary.csv (and printed to the console).
# Example aggregate row (18 queries, one corpus snapshot — re-run after re-labeling):

# | ranker              | n_queries | P@5    | NDCG@5 | P@10   | NDCG@10 | MAP    |
# |---------------------|-----------|--------|--------|--------|---------|--------|
# | tfidf_cosine        | 18        | 0.8333 | 0.8548 | 0.7889 | 0.8182  | 0.306  |
# | tfidf_dice          | 18        | 0.8222 | 0.8384 | 0.7778 | 0.8035  | 0.2995 |
# | tfidf_jaccard       | 18        | 0.6778 | 0.6925 | 0.6611 | 0.6772  | 0.2405 |
# | bm25_plain          | 18        | 0.8333 | 0.8642 | 0.7944 | 0.8296  | 0.3082 |
# | bm25_field          | 18        | 0.6778 | 0.7145 | 0.5778 | 0.635   | 0.2087 |
# | twostage_bm25_tfidf | 18        | 0.8333 | 0.8548 | 0.7889 | 0.8182  | 0.306  |
# | lsi                 | 18        | 0.2667 | 0.3113 | 0.2278 | 0.2697  | 0.0632 |
# | bm25_prf            | 18        | 0.6667 | 0.7384 | 0.4278 | 0.5484  | 0.1663 |

### web UI

python -m src.app.cli serve \
  --index-dir data/index \
  --host 127.0.0.1 \
  --port 8080 \
  --default-ranker bm25_plain \
  --warmup

# then open http://127.0.0.1:8080/ in your browser

# routes:

# /             home

# /search?q=... single-ranker results (q, ranker, k)

# /compare?q=...side-by-side compare across all rankers