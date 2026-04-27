### crawler

python -m src.crawler.cli \
  --subreddits science programming cooking AskHistorians \
  --limit 1000 --comments 5 --sort top --time month \
  --output data/raw