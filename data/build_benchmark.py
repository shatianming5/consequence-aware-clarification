"""Build the real-table ASK/ANSWER pool from WikiSQL, Nature Source Data, DS-1000.

Paper: 81,960 balanced items, then a stratified eval subset. Pass local paths to
the three public sources; this script does not download them.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import random
from collections import Counter, defaultdict

import cac.proto_engine as E
from cac.load_ds1000 import load_ds1000_tables
from cac.load_nature import load_nature_tables
from cac.load_wikisql import load_wikisql_all


def build_source_items(name, tables):
    raw = []
    for table in tables:
        raw += E.build_pairs_for_table(table["topic"], table["df"], name, table["meta"])
    return raw


def diversify(items, per_table_cap=3, per_family_frac=0.40, seed=0):
    rng = random.Random(seed)
    rng.shuffle(items)
    by_table = defaultdict(int)
    kept = []
    for item in items:
        tid = item["meta"].get("id", "?") + "|" + item["variant"]
        if by_table[tid] < per_table_cap:
            by_table[tid] += 1
            kept.append(item)
    fam_cap = max(1, int(len(kept) * per_family_frac))
    by_fam = defaultdict(int)
    kept2 = []
    for item in kept:
        if by_fam[item["family"]] < fam_cap:
            by_fam[item["family"]] += 1
            kept2.append(item)
    return kept2


def dedup(items):
    seen, uniq = set(), []
    for item in items:
        key = (item["table_csv"], item["request"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    return uniq


def balance(items, seed, cap=None):
    rng = random.Random(seed)
    ask = [x for x in items if x["label"] == "ASK"]
    ans = [x for x in items if x["label"] == "ANSWER"]
    rng.shuffle(ask)
    rng.shuffle(ans)
    k = min(len(ask), len(ans))
    if cap:
        k = min(k, cap // 2)
    bal = ask[:k] + ans[:k]
    rng.shuffle(bal)
    return bal


def count_pairs(items):
    pairs = defaultdict(list)
    for item in items:
        if "pair_id" in item:
            pairs[item["pair_id"]].append(item["label"])
    return sum(1 for v in pairs.values() if sorted(v) == ["ANSWER", "ASK"])


def summarize(items):
    return {
        "n": len(items),
        "labels": dict(Counter(x["label"] for x in items)),
        "sources": dict(Counter(x["source"] for x in items)),
        "families": dict(Counter(x["family"] for x in items)),
        "axes": dict(Counter(x["axes"] for x in items)),
        "complete_hard_neg_pairs": count_pairs(items),
    }


def write(path, items):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def carve_eval(full_items, target, seed):
    rng = random.Random(seed)
    pairs = defaultdict(list)
    for item in full_items:
        if "pair_id" in item:
            pairs[item["pair_id"]].append(item)
    complete = [g for g in pairs.values() if sorted(x["label"] for x in g) == ["ANSWER", "ASK"]]
    rng.shuffle(complete)
    eval_items = []
    for group in complete:
        eval_items.extend(group)
        if len(eval_items) >= target // 2:
            break
    remaining = [item for item in full_items if item not in eval_items]
    strata = defaultdict(list)
    for item in remaining:
        strata[(item["source"], item["axes"], item["label"])].append(item)
    for bucket in strata.values():
        rng.shuffle(bucket)
    need = target - len(eval_items)
    keys = list(strata.keys())
    rng.shuffle(keys)
    idx = 0
    while need > 0 and any(strata.values()):
        key = keys[idx % len(keys)]
        idx += 1
        if strata[key]:
            eval_items.append(strata[key].pop())
            need -= 1
        if idx > len(keys) * 10000:
            break
    return balance(eval_items, seed)


def reuse_or_none(path, reuse):
    if reuse and os.path.exists(path):
        rows = [json.loads(line) for line in open(path)]
        print(f"REUSE {path}: {len(rows)} items", flush=True)
        return rows
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/full")
    parser.add_argument("--ds1000", help="DS-1000 jsonl.gz")
    parser.add_argument("--wikisql-dir", help="directory with {dev,test,train}.tables.jsonl")
    parser.add_argument("--wikisql-limit", type=int, default=None)
    parser.add_argument("--nature-root", help="Nature Source Data workbook tree")
    parser.add_argument("--nature-max-bytes", type=int, default=60_000_000)
    parser.add_argument("--eval-target", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    stats = {}
    per_source = {}

    if args.ds1000:
        path = f"{args.out}/ood_ds1000_full.jsonl"
        ds = reuse_or_none(path, args.reuse)
        if ds is None:
            tables = load_ds1000_tables(args.ds1000)
            ds = balance(dedup(diversify(build_source_items("ds1000", tables), seed=args.seed)), args.seed)
            write(path, ds)
        per_source["ds1000"] = ds
        stats["ds1000"] = summarize(ds)
        print("DS-1000:", stats["ds1000"]["n"], flush=True)

    if args.wikisql_dir:
        path = f"{args.out}/ood_wikisql_full.jsonl"
        ws = reuse_or_none(path, args.reuse)
        if ws is None:
            splits = [
                p for p in (
                    f"{args.wikisql_dir}/{split}.tables.jsonl" for split in ("dev", "test", "train")
                ) if os.path.exists(p)
            ]
            tables = load_wikisql_all(splits, limit=args.wikisql_limit, seed=args.seed)
            ws = balance(dedup(diversify(build_source_items("wikisql", tables), seed=args.seed)), args.seed)
            write(path, ws)
        per_source["wikisql"] = ws
        stats["wikisql"] = summarize(ws)
        print("WikiSQL:", stats["wikisql"]["n"], flush=True)

    if args.nature_root:
        path = f"{args.out}/ood_nature_full.jsonl"
        nat = reuse_or_none(path, args.reuse)
        if nat is None:
            tables, _st = load_nature_tables(root=args.nature_root, verbose=True, max_bytes=args.nature_max_bytes)
            nat = balance(dedup(diversify(build_source_items("nature", tables), seed=args.seed)), args.seed)
            write(path, nat)
        per_source["nature"] = nat
        stats["nature"] = summarize(nat)
        print("Nature:", stats["nature"]["n"], flush=True)

    if not per_source:
        raise SystemExit("pass at least one of --ds1000 --wikisql-dir --nature-root")

    combined = [item for rows in per_source.values() for item in rows]
    full = balance(combined, args.seed)
    write(f"{args.out}/ood_real_full.jsonl", full)
    stats["combined_full"] = summarize(full)
    print("COMBINED FULL:", stats["combined_full"], flush=True)

    eval_set = carve_eval(full, args.eval_target, args.seed)
    write(f"{args.out}/ood_eval.jsonl", eval_set)
    stats["eval_subset"] = summarize(eval_set)
    print("EVAL SUBSET:", stats["eval_subset"], flush=True)

    with open(f"{args.out}/ood_full_STATS.json", "w") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
