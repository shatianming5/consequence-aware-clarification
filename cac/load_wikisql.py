"""Load real tables from WikiSQL tables.jsonl (SQL-domain, Wikipedia-derived).

WikiSQL is one of the three base corpora VALASQL is built on (Spider/BIRD/WikiSQL).
VALASQL's own value-ambiguity labels require the full SQLite databases plus an
LLM pass to regenerate; for an independent SQL-domain real-table validation set
we consume the underlying WikiSQL tables directly (self-contained header+rows).
"""
import json

import pandas as pd


def load_wikisql_tables(path, limit=None, min_rows=6, min_cols=2, seed=0):
    import random
    rng = random.Random(seed)
    raw = [json.loads(l) for l in open(path)]
    rng.shuffle(raw)
    out = []
    for t in raw:
        header = [str(h) for h in t.get("header", [])]
        rows = t.get("rows", [])
        if len(header) < min_cols or len(rows) < min_rows:
            continue
        try:
            df = pd.DataFrame(rows, columns=header)
        except Exception:  # noqa: BLE001
            continue
        # coerce 'real'-typed columns to numeric so aggregation prototypes fire
        for c, ty in zip(header, t.get("types", [])):
            if ty == "real":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        out.append({
            "topic": "WikiSQL table (SQL domain)",
            "df": df,
            "meta": {"id": f"wikisql_{t.get('id')}", "source_url": "salesforce/WikiSQL",
                     "table_id": t.get("id"), "types": t.get("types")},
        })
        if limit and len(out) >= limit:
            break
    return out


def load_wikisql_all(paths, limit=None, min_rows=6, min_cols=2, seed=0):
    """Load and dedup real tables across multiple WikiSQL splits (dev/test/train)."""
    import random
    seen = set()
    merged = []
    for p in paths:
        for t in load_wikisql_tables(p, limit=None, min_rows=min_rows,
                                     min_cols=min_cols, seed=seed):
            tid = t["meta"]["table_id"]
            if tid in seen:
                continue
            seen.add(tid)
            merged.append(t)
    rng = random.Random(seed)
    rng.shuffle(merged)
    if limit:
        merged = merged[:limit]
    return merged


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "/tmp/wikisql/data/dev.tables.jsonl"
    t = load_wikisql_tables(p, limit=500)
    print("loaded WikiSQL tables:", len(t))
    print("example:", t[0]["df"].shape, list(t[0]["df"].columns)[:6])
