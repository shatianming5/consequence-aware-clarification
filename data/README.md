# Data

Datasets are not in this repository. Build them locally from WikiSQL, Nature
Source Data, and DS-1000:

```bash
python data/build_benchmark.py \
  --wikisql-dir /path/to/WikiSQL/data \
  --ds1000 /path/to/ds1000.jsonl.gz \
  --nature-root /path/to/nature_source_data \
  --out data/full
```

Each item is one JSON line: `request`, `table_csv`, `plausible[{desc, code}]`,
`label`, `axes`, `family`, `source`, `pair_id`. The label is ASK when the
reference probes return different successful outputs on that table, and ANSWER
when they coincide.

`eval/evaluate.py`, `eval/analyze.py`, `eval/frontier.py`, `train/sft.py`, and
`train/rlvr.py` take the resulting jsonl paths as arguments.
