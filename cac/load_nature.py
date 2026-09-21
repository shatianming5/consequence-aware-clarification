"""Conservatively extract real 2D tables from Nature Source Data XLSX workbooks.

Reuses the header-detection heuristic from the scan pass: for each sheet, score
the first rows to find the true header, drop the prose/caption rows above it,
keep only sheets that parse as a regular table with >=1 numeric column.
Provenance (DOI, journal, year, CC-BY license) is attached per table.
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = None


def _nonempty(x):
    return not (pd.isna(x) or (isinstance(x, str) and not x.strip()))


def _score_header(row):
    vals = [str(x).strip() for x in row if _nonempty(x)]
    if len(vals) < 2:
        return -99
    unique = len(set(vals)) / max(1, len(vals))
    text = sum(bool(re.search(r"[A-Za-z_]", x)) for x in vals)
    numeric = sum(bool(re.fullmatch(r"[-+]?\d+(\.\d+)?", x)) for x in vals)
    return 2 * len(vals) + text - 2 * numeric + unique


def _extract_sheet(raw):
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if raw.shape[0] < 4 or raw.shape[1] < 2:
        return None
    scores = [_score_header(raw.iloc[i].tolist()) for i in range(min(8, len(raw)))]
    hi = max(range(len(scores)), key=scores.__getitem__)
    hdr = raw.iloc[hi].tolist()
    data = raw.iloc[hi + 1:].copy()
    seen, cols = {}, []
    for i, x in enumerate(hdr):
        c = str(x).strip() if _nonempty(x) else f"col_{i}"
        c = re.sub(r"\s+", " ", c).strip()[:60] or f"col_{i}"
        base, k = c, 1
        while c in seen:            # guarantee global uniqueness
            k += 1
            c = f"{base}_{k}"
        seen[c] = True
        cols.append(c)
    data.columns = cols
    data = data.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if len(data) < 3 or data.shape[1] < 2:
        return None
    numcols = []
    for c in data.columns:
        col = data[c]
        if isinstance(col, pd.DataFrame):   # defensive: never expect this now
            col = col.iloc[:, 0]
        if pd.to_numeric(col, errors="coerce").notna().mean() >= 0.6:
            numcols.append(c)
    if not numcols:
        return None
    return data.reset_index(drop=True)


def _provenance(doi_dir):
    for cand in [doi_dir / "meta" / "provenance.json",
                 doi_dir / "provenance.json"]:
        if cand.exists():
            try:
                p = json.loads(cand.read_text())
                return {"doi": p.get("doi"), "journal": p.get("journal"),
                        "year": p.get("year"),
                        "license": p.get("license", {}).get("license_id")}
            except Exception:  # noqa: BLE001
                pass
    return {"doi": doi_dir.name}


def load_nature_tables(root=None, max_workbooks=None, min_rows=4, min_cols=2,
                       max_rows=120, max_cols=15, verbose=False,
                       max_bytes=25_000_000, max_sheets=40, dir_glob="*"):
    if root is None:
        raise ValueError("root is required: directory of Nature Source Data workbooks")
    root = Path(root)
    # dir_glob="*" spans ALL CC-BY biology source-data directories:
    # Nature Communications (s41467), eLife, EMBO Press (s44318/9/20/1 + legacy
    # embj/embr/emmm/msb) and Life Science Alliance (lsa). The original release
    # used "s*" (Nature Communications only); broadened here to the full corpus.
    paths = sorted(root.glob(f"{dir_glob}/source_data/*.xlsx")) + \
        sorted(root.glob(f"{dir_glob}/source_data/*.xls"))
    if max_workbooks:
        paths = paths[:max_workbooks]
    out = []
    stats = Counter()
    for ix, path in enumerate(paths, 1):
        doi_dir = path.parent.parent
        prov = _provenance(doi_dir)
        try:
            if path.stat().st_size > max_bytes:
                stats["skipped_oversize"] += 1
                if verbose and ix % 20 == 0:
                    print(f"...loaded {ix}/{len(paths)} workbooks, "
                          f"{stats['usable_tables']} tables", flush=True)
                continue
            xl = pd.ExcelFile(path)
        except Exception:  # noqa: BLE001
            stats["unreadable_workbook"] += 1
            continue
        for sh in xl.sheet_names[:max_sheets]:
            try:
                raw = pd.read_excel(path, sheet_name=sh, header=None, nrows=200)
            except Exception:  # noqa: BLE001
                continue
            df = _extract_sheet(raw)
            if df is None:
                continue
            if not (min_rows <= len(df) <= max_rows and
                    min_cols <= df.shape[1] <= max_cols):
                continue
            stats["usable_tables"] += 1
            out.append({
                "topic": f"{prov.get('journal', 'Nature')} source data",
                "df": df,
                "meta": {"id": f"nature_{prov.get('doi', doi_dir.name)}_{sh}",
                         "doi": prov.get("doi"), "journal": prov.get("journal"),
                         "year": prov.get("year"), "license": prov.get("license"),
                         "sheet": str(sh), "workbook": path.name},
            })
        if verbose and ix % 20 == 0:
            print(f"...loaded {ix}/{len(paths)} workbooks, {stats['usable_tables']} tables",
                  flush=True)
    return out, dict(stats)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m cac.load_nature /path/to/nature_source_data")
    tabs, st = load_nature_tables(root=sys.argv[1], verbose=True)
    print("STATS", st)
    print("total usable Nature tables:", len(tabs))
    if tabs:
        t = tabs[0]
        print("example:", t["meta"]["doi"], "| shape", t["df"].shape,
              "| cols", list(t["df"].columns)[:6])
