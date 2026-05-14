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

python -m src.rankers.cli search --index-dir data/index --query "python async websocket" --ranker bm25
python -m src.rankers.cli search --index-dir data/index --query "python guide" --ranker tfidf --top-k 5
python -m src.rankers.cli search --index-dir data/index --query "climate" --ranker bm25 --field-aware
python -m src.rankers.cli search --index-dir data/index --ranker bm25 --field-aware    # interactive

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

# 4. compute metrics for all rankers

python -m src.evaluate.cli run \
  --queries qrel/queries.json \
  --qrels qrel/qrels.csv \
  --index-dir data/index \
  --output-dir qrel

### web UI

python -m src.app.cli serve \
  --index-dir data/index \
  --host 127.0.0.1 \
  --port 8080 \
  --default-ranker bm25 \
  --warmup

# then open http://127.0.0.1:8080/ in your browser

# routes:

# /             home

# /search?q=... single-ranker results (q, ranker, k)

# /compare?q=...side-by-side compare across all rankers