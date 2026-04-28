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

python -m src.rankers.cli search --query "python async websocket" --ranker bm25
python -m src.rankers.cli search --query "python guide" --ranker tfidf --top-k 5
python -m src.rankers.cli search --query "climate" --ranker bm25 --field-aware
python -m src.rankers.cli search --ranker bm25 --field-aware    # interactive

### evaluate

# 1. write data/qrels/queries.json with 15-20 queries:
#    {"queries": [{"qid": "q1", "text": "..."}, ...]}

# 2. generate pool template
python -m src.evaluate.cli pool \
  --queries data/qrels/queries.json \
  --index-dir data/index \
  --output data/qrels/pool.csv \
  --depth 20

# 3. open data/qrels/pool.csv, fill the 'label' column (1=relevant, blank/0=not),
#    save as data/qrels/qrels.csv (or overwrite pool.csv).

# 4. compute metrics for all rankers
python -m src.evaluate.cli run \
  --queries data/qrels/queries.json \
  --qrels data/qrels/qrels.csv \
  --index-dir data/index \
  --output-dir data/qrels