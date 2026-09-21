"""Load real pandas DataFrames from DS-1000 prompts (StackOverflow-derived)."""
import gzip
import json
import re

import pandas as pd


def _extract_df(prompt):
    m = re.search(r"(df\w*\s*=\s*pd\.DataFrame\()", prompt)
    if not m:
        return None
    tail = prompt[m.start(1):]
    for stop in ["</code>", "\nresult", "\n\n\n", "\nprint(", "\ndef "]:
        i = tail.find(stop)
        if i > 0:
            tail = tail[:i]
    lv = {}
    try:
        exec("import pandas as pd\nimport numpy as np\n" + tail, {}, lv)
    except Exception:  # noqa: BLE001
        return None
    for k, v in lv.items():
        if isinstance(v, pd.DataFrame):
            return v
    return None


def load_ds1000_tables(path, min_rows=4, min_cols=2):
    rows = [json.loads(l) for l in gzip.open(path, "rt")]
    out = []
    for r in rows:
        if r.get("metadata", {}).get("library") != "Pandas":
            continue
        df = _extract_df(r["prompt"])
        if df is None:
            continue
        if df.shape[0] < min_rows or df.shape[1] < min_cols:
            continue
        pid = r.get("metadata", {}).get("problem_id", None)
        out.append({
            "topic": "DS-1000 pandas task",
            "df": df,
            "meta": {"id": f"ds1000_{pid}", "source_url": "xlang-ai/DS-1000",
                     "library": "Pandas", "problem_id": pid},
        })
    return out


if __name__ == "__main__":
    import sys
    t = load_ds1000_tables(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ds1000.jsonl.gz")
    print("loaded DS-1000 tables:", len(t))
    print("example shape:", t[0]["df"].shape, "cols:", list(t[0]["df"].columns)[:8])
