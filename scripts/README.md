# Scripts

## `csv_to_raw_jsonl.py`

Converts Reddit CSV dumps under **`dataset/`** into **raw JSONL** under
**`data/raw/`** (one `.jsonl` per input `.csv`, same basename).

**Typical use (submission includes `dataset/` only — run this first):**

```bash
python scripts/csv_to_raw_jsonl.py --input-dir dataset --output-dir data/raw
```

**Single file:**

```bash
python scripts/csv_to_raw_jsonl.py --input dataset/gaming_minecraft.csv --output data/raw/gaming_minecraft.jsonl
```

**Schemas:** auto-detected — numeric headers `,0,1,...,10` (current `dataset/*.csv`)
or named `text,id,subreddit,...` (`dataset/headers.txt` style).

**Then rebuild the IR pipeline:**

```bash
python -m src.preprocess.cli --input data/raw --output data/processed
python -m src.index.cli build --input data/processed --output data/index
```

If `dataset/` CSVs are missing, restore them from your course data source; this
script only converts what is present on disk.
