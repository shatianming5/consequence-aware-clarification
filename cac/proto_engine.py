"""Build ASK/ANSWER items from real tables.

Each prototype writes two pandas readings of one request. Execution on the
table assigns the label: ASK if the readings diverge, ANSWER if they coincide.
Hard-negative pairs keep the request and schema fixed and only change cells.
"""
import os
import re

import pandas as pd

from cac.harness import eq1_verdict, run_probes


def label_from_csv(table_csv, plausible):
    """Same execute/normalize/Eq. (1) rule as the evaluator."""
    codes = [p["code"] for p in plausible]
    outcomes, n_ok = run_probes(codes, table_csv)
    has_err = n_ok < len(codes)
    return eq1_verdict(outcomes), outcomes, has_err


# --------------------------------------------------------------------------- #
# Column typing on real (messy) tables
# --------------------------------------------------------------------------- #

_SAFE_NAME = re.compile(r"^[A-Za-z0-9 _.()%/+\-]+$")


def _safe(colname):
    s = str(colname)
    return bool(_SAFE_NAME.match(s)) and len(s) <= 40 and "'" not in s and '"' not in s


def col_types(df):
    """Return (numeric_cols, categorical_cols) using content, not just dtype."""
    num_cols, cat_cols = [], []
    n = len(df)
    for c in df.columns:
        if not _safe(c):
            continue
        s = df[c]
        coerced = pd.to_numeric(s, errors="coerce")
        frac_num = float(coerced.notna().mean()) if n else 0.0
        nun = int(s.nunique(dropna=True))
        if frac_num >= 0.8 and coerced.nunique(dropna=True) >= 2:
            num_cols.append(c)
        elif 2 <= nun <= max(2, n - 1):
            # a groupable / filterable categorical column
            cat_cols.append(c)
    return num_cols, cat_cols


def q(name):
    """repr() a column name for safe embedding in generated code strings."""
    return repr(str(name))


# --------------------------------------------------------------------------- #
# Prototype library.  Each prototype is data-dependent and ships a
# `make_converge` that removes the divergence-causing condition from the SAME
# real table, producing the ANSWER counterpart of a hard-negative pair.
# --------------------------------------------------------------------------- #

class Proto:
    family = "base"
    axes = 1  # number of independent decision points (>=2 => composite)

    def applicable(self, df, num, cat):
        raise NotImplementedError

    def request(self, num, cat):
        raise NotImplementedError

    def plausible(self, num, cat):
        raise NotImplementedError

    def make_converge(self, df, num, cat):
        """Return a copy of df with the divergence condition removed (=> ANSWER)."""
        raise NotImplementedError


class CountGranularity(Proto):
    """Distinct vs row-count.  Diverges iff the category column has duplicates."""
    family = "count_granularity"
    axes = 1

    def applicable(self, df, num, cat):
        return cat is not None

    def request(self, num, cat):
        return f"How many {cat} does this dataset cover?"

    def plausible(self, num, cat):
        return [
            {"desc": "distinct values", "code": f"result = df[{q(cat)}].nunique()"},
            {"desc": "record count", "code": f"result = len(df)"},
        ]

    def make_converge(self, df, num, cat):
        return df.drop_duplicates(subset=[cat]).reset_index(drop=True)


class DedupSum(Proto):
    """Sum vs sum-after-dropping-full-duplicate-rows.  Diverges iff dup rows."""
    family = "dedup_sum"
    axes = 1

    def applicable(self, df, num, cat):
        return num is not None

    def request(self, num, cat):
        return f"What is the total {num} in this table?"

    def plausible(self, num, cat):
        return [
            {"desc": "sum every row", "code": f"result = round(float(df[{q(num)}].sum()), 6)"},
            {"desc": "drop exact duplicate rows first",
             "code": f"result = round(float(df.drop_duplicates()[{q(num)}].sum()), 6)"},
        ]

    def make_converge(self, df, num, cat):
        return df.drop_duplicates().reset_index(drop=True)


class NanMean(Proto):
    """skipna mean vs zero-filled mean.  Diverges iff the numeric col has NaN."""
    family = "nan_mean"
    axes = 1

    def applicable(self, df, num, cat):
        return num is not None

    def request(self, num, cat):
        return f"Compute the mean {num}."

    def plausible(self, num, cat):
        return [
            {"desc": "skip missing", "code": f"result = round(float(df[{q(num)}].mean()), 6)"},
            {"desc": "treat missing as zero",
             "code": f"result = round(float(df[{q(num)}].fillna(0).mean()), 6)"},
        ]

    def make_converge(self, df, num, cat):
        out = df.copy()
        out = out[out[num].notna()].reset_index(drop=True)
        return out


class FilterBoundary(Proto):
    """positive (>0) vs non-negative (>=0) mean.  Diverges iff zeros present."""
    family = "filter_boundary"
    axes = 1

    def applicable(self, df, num, cat):
        return num is not None

    def request(self, num, cat):
        return f"What is the average of the positive {num} values?"

    def plausible(self, num, cat):
        return [
            {"desc": "strictly greater than zero",
             "code": f"result = round(float(df[df[{q(num)}] > 0][{q(num)}].mean()), 6)"},
            {"desc": "zero counts as positive",
             "code": f"result = round(float(df[df[{q(num)}] >= 0][{q(num)}].mean()), 6)"},
        ]

    def make_converge(self, df, num, cat):
        out = df.copy()
        col = pd.to_numeric(out[num], errors="coerce")
        return out[col != 0].reset_index(drop=True)


class TieBreak(Proto):
    """Top group by total: single argmax vs all tied leaders.  Diverges iff tie."""
    family = "tie_break"
    axes = 1

    def applicable(self, df, num, cat):
        return num is not None and cat is not None

    def request(self, num, cat):
        return f"Which {cat} has the greatest total {num}?"

    def plausible(self, num, cat):
        base = f"s = df.groupby({q(cat)})[{q(num)}].sum()"
        return [
            {"desc": "the single top group", "code": f"{base}; result = [s.idxmax()]"},
            {"desc": "all groups tied at the maximum",
             "code": f"{base}; result = sorted([str(k) for k in s[s == s.max()].index.tolist()])"},
        ]

    def make_converge(self, df, num, cat):
        # perturb the current top group upward so the maximum is unique
        out = df.copy()
        out[num] = pd.to_numeric(out[num], errors="coerce")
        g = out.groupby(cat)[num].sum()
        if g.empty:
            return out
        top = g.idxmax()
        idx = out.index[out[cat] == top]
        if len(idx):
            out.loc[idx[0], num] = float(out[num].abs().sum()) + 1.0
        return out.reset_index(drop=True)


class MeanOfMeans(Proto):
    """COMPOSITE (grouping granularity x aggregation): pooled row mean vs
    mean-of-group-means.  Diverges iff group sizes are unequal."""
    family = "mean_of_means"
    axes = 2

    def applicable(self, df, num, cat):
        return num is not None and cat is not None

    def request(self, num, cat):
        return f"What is the average {num} per {cat}?"

    def plausible(self, num, cat):
        return [
            {"desc": "pooled mean over all rows",
             "code": f"result = round(float(df[{q(num)}].mean()), 6)"},
            {"desc": "mean of each group's mean",
             "code": f"result = round(float(df.groupby({q(cat)})[{q(num)}].mean().mean()), 6)"},
        ]

    def make_converge(self, df, num, cat):
        # keep an equal number of rows per group (=> pooled == mean-of-means)
        out = df.copy()
        k = int(out.groupby(cat).size().min())
        if k < 1:
            return out
        out = out.groupby(cat, group_keys=False).head(k).reset_index(drop=True)
        return out


class CaseMatch(Proto):
    """COMPOSITE (match semantics x count): exact vs case-insensitive equality on
    a chosen category value.  Diverges iff case variants of that value exist."""
    family = "case_match"
    axes = 2

    def _pick_value(self, df, cat):
        vals = df[cat].dropna().astype(str)
        low = vals.str.lower()
        for v in vals.unique():
            if (low == str(v).lower()).sum() > (vals == v).sum():
                return v
        # fall back to the modal value (may or may not have case variants)
        return vals.mode().iloc[0] if not vals.empty else None

    def applicable(self, df, num, cat):
        if cat is None:
            return False
        v = self._pick_value(df, cat)
        self._v = v
        return v is not None and _SAFE_NAME.match(str(v)) is not None and "'" not in str(v)

    def request(self, num, cat):
        return f"How many records have {cat} equal to '{self._v}'?"

    def plausible(self, num, cat):
        v = str(self._v)
        return [
            {"desc": "exact match",
             "code": f"result = int((df[{q(cat)}].astype(str) == {q(v)}).sum())"},
            {"desc": "case-insensitive match",
             "code": f"result = int((df[{q(cat)}].astype(str).str.lower() == {q(v)}.lower()).sum())"},
        ]

    def make_converge(self, df, num, cat):
        # normalize every value in the column to a single canonical case =>
        # exact and case-insensitive counts coincide
        out = df.copy()
        v = str(self._v)
        mask = out[cat].astype(str).str.lower() == v.lower()
        out.loc[mask, cat] = v
        return out.reset_index(drop=True)


class NanShare(Proto):
    """COMPOSITE (missing-value policy x denominator): the share of rows above
    the average.  Reading 1 drops missing rows and divides by the surviving
    count; reading 2 treats missing as zero and divides by every row.  Diverges
    iff the numeric column has missing values."""
    family = "nan_share"
    axes = 2

    def applicable(self, df, num, cat):
        return num is not None and bool(df[num].isna().any()) and len(df) > 1

    def request(self, num, cat):
        return f"What share of the rows have an above-average {num}?"

    def plausible(self, num, cat):
        return [
            {"desc": "drop missing rows, divide by the rows that remain",
             "code": (f"d = df.dropna(subset=[{q(num)}])\n"
                      f"result = round(float((d[{q(num)}] > d[{q(num)}].mean()).sum() / len(d)), 6)")},
            {"desc": "treat missing as zero, divide by every row",
             "code": (f"s = df[{q(num)}].fillna(0)\n"
                      f"result = round(float((s > df[{q(num)}].mean()).sum() / len(df)), 6)")},
        ]

    def make_converge(self, df, num, cat):
        # Remove the missing values and the two policies coincide by construction.
        return df.dropna(subset=[num]).reset_index(drop=True)


class RoundStage(Proto):
    """COMPOSITE (rounding stage x aggregation granularity): a whole-number
    average.  Reading 1 rounds every row and averages the rows; reading 2
    averages within each group, rounds each group, and averages the groups."""
    family = "round_stage"
    axes = 2

    def applicable(self, df, num, cat):
        if num is None or cat is None:
            return False
        return df[cat].nunique(dropna=True) >= 2 and df[num].notna().sum() >= 4

    def request(self, num, cat):
        return f"What is the average {num} across {cat}, as a whole number?"

    def plausible(self, num, cat):
        return [
            {"desc": "round every row, then average the rows",
             "code": f"result = round(float(df[{q(num)}].round().mean()), 6)"},
            {"desc": "average within each group, round each group, then average the groups",
             "code": (f"result = round(float(df.groupby({q(cat)})[{q(num)}]"
                      f".mean().round().mean()), 6)")},
        ]

    def make_converge(self, df, num, cat):
        # Give every group one integer value and equal size: rounding then
        # commutes with averaging, and the two group sizes no longer reweight.
        out = df.dropna(subset=[num, cat]).copy()
        if out.empty:
            return out
        k = int(out.groupby(cat).size().min())
        if k < 1:
            return out
        out = out.groupby(cat, group_keys=False).head(k).reset_index(drop=True)
        rounded = out.groupby(cat)[num].transform(lambda s: float(round(float(s.mean()))))
        out[num] = rounded
        return out.reset_index(drop=True)


class GroupThreshold(Proto):
    """COMPOSITE (comparison boundary x scope of the comparison): how many groups
    reach the average.  Reading 1 asks whether the group's own mean is at least
    the overall mean; reading 2 asks whether the group holds any row strictly
    above it."""
    family = "group_threshold"
    axes = 2

    def applicable(self, df, num, cat):
        if num is None or cat is None:
            return False
        return df[cat].nunique(dropna=True) >= 2 and df[num].notna().sum() >= 4

    def request(self, num, cat):
        return f"How many {cat} groups reach the overall average {num}?"

    def plausible(self, num, cat):
        return [
            {"desc": "the group's own mean is at least the overall mean",
             "code": (f"g = df.groupby({q(cat)})[{q(num)}].mean()\n"
                      f"result = int((g >= df[{q(num)}].mean()).sum())")},
            {"desc": "the group holds at least one row strictly above the overall mean",
             "code": (f"m = df[{q(num)}].mean()\n"
                      f"result = int(df[df[{q(num)}] > m][{q(cat)}].nunique())")},
        ]

    def make_converge(self, df, num, cat):
        # One value per group, and no group sitting exactly on the overall mean:
        # "mean at least" and "any row above" then pick out the same groups.
        out = df.dropna(subset=[num, cat]).copy()
        if out.empty:
            return out
        out[num] = out.groupby(cat)[num].transform("mean")
        m = float(out[num].mean())
        on_boundary = out[num] == m
        if bool(on_boundary.any()):
            spread = float(out[num].std(ddof=0)) or 1.0
            out.loc[on_boundary, num] = m + spread / 8.0
        return out.reset_index(drop=True)


PROTOS = [
    CountGranularity(), DedupSum(), NanMean(), FilterBoundary(),
    TieBreak(), MeanOfMeans(), CaseMatch(),
]

# Added for T1.3.  The published artifacts were generated from PROTOS alone, so
# these stay OFF unless PROTO_SET=v2 is set in the environment.  Never fold them
# into PROTOS: doing so silently changes what every existing build script emits.
PROTOS_EXTRA_COMPOSITE = [NanShare(), RoundStage(), GroupThreshold()]


def active_protos():
    if os.environ.get("PROTO_SET") == "v2":
        return PROTOS + PROTOS_EXTRA_COMPOSITE
    return PROTOS


# --------------------------------------------------------------------------- #
# Table shaping + item construction
# --------------------------------------------------------------------------- #

def shape_table(df, max_rows=30, max_cols=10):
    """Trim a real table to a promptable size WITHOUT altering cell values."""
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    df = df.loc[:, [c for c in df.columns if _safe(c)]]
    # drop fully-empty columns
    df = df.dropna(axis=1, how="all")
    if df.shape[1] > max_cols:
        df = df.iloc[:, :max_cols]
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df.reset_index(drop=True)


def _mk_item(topic, df, proto, num, cat, source, meta, variant):
    csv = df.to_csv(index=False)
    plausible = proto.plausible(num, cat)
    label, _outcomes, has_err = label_from_csv(csv, plausible)
    return {
        "topic": topic,
        "table_csv": csv,
        "request": proto.request(num, cat),
        "plausible": plausible,
        "label": label,
        "has_err": has_err,
        "family": proto.family,
        "axes": proto.axes,
        "variant": variant,          # "raw" or "converge"
        "source": source,            # ds1000 / nature / valasql
        "meta": meta,                # provenance (doi, problem id, db, ...)
    }


def build_pairs_for_table(topic, df, source, meta, max_combos=6):
    """Apply every applicable prototype to one real table and emit hard-negative
    (raw ASK, converge ANSWER) pairs, labeled purely by execution."""
    items = []
    df = shape_table(df)
    if df.shape[0] < 4 or df.shape[1] < 2:
        return items
    num_cols, cat_cols = col_types(df)
    if not num_cols and not cat_cols:
        return items

    combos = []
    for num in (num_cols or [None]):
        for cat in (cat_cols or [None]):
            combos.append((num, cat))
    combos = combos[:max_combos]

    for proto in active_protos():
        for (num, cat) in combos:
            try:
                if not proto.applicable(df, num, cat):
                    continue
                raw = _mk_item(topic, df, proto, num, cat, source, meta, "raw")
                # Only keep the raw item if it is a genuine divergence (ASK);
                # otherwise the table simply doesn't trigger this ambiguity.
                # Reject if either interpretation errored (degenerate item).
                if raw["label"] != "ASK" or raw["has_err"]:
                    continue
                conv_df = shape_table(proto.make_converge(df, num, cat))
                if conv_df.shape[0] < 2:
                    continue
                conv = _mk_item(topic, df=conv_df, proto=proto, num=num, cat=cat,
                                source=source, meta=meta, variant="converge")
                # We want the counterpart to actually converge; only then is it a
                # valid hard negative. If it still diverges, skip (do not fake it).
                if conv["label"] != "ANSWER" or conv["has_err"]:
                    continue
                raw["pair_id"] = f"{source}:{meta.get('id','?')}:{proto.family}:{num}:{cat}"
                conv["pair_id"] = raw["pair_id"]
                items.append(raw)
                items.append(conv)
            except Exception:  # noqa: BLE001
                continue
    return items


def dedup_and_balance(items, seed=0):
    """Drop duplicate (table_csv, request) items; return balanced ASK/ANSWER."""
    import random
    rng = random.Random(seed)
    seen = set()
    uniq = []
    for it in items:
        key = (it["table_csv"], it["request"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
    ask = [it for it in uniq if it["label"] == "ASK"]
    ans = [it for it in uniq if it["label"] == "ANSWER"]
    rng.shuffle(ask)
    rng.shuffle(ans)
    k = min(len(ask), len(ans))
    bal = ask[:k] + ans[:k]
    rng.shuffle(bal)
    return bal, {"n_ask_raw": len(ask), "n_answer_raw": len(ans), "n_balanced": len(bal)}
