"""Paper Table 1 / Table B readouts from a per-item dump.

Reports single-axis vs composite, hard-negative pairs (both members correct),
ultra-clean composite (n=372 in the paper: neither table nor request in the
composite SFT mix), and hardest pairs (n=577: complete pairs whose both
members are ultra-clean). McNemar is the exact two-sided test.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
from collections import defaultdict

from cac.stats import exact_binomial_two_sided, exact_mcnemar_p


def norm_text(s) -> str:
    return hashlib.md5(" ".join(str(s).split()).strip().lower().encode()).hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def acc(rows, key):
    if not rows:
        return 0.0
    return 100.0 * sum(r[key] == r["gold"] for r in rows) / len(rows)


def report_slice(name, rows):
    if not rows:
        print(f"  {name:22}  n=0")
        return
    sim = acc(rows, "simulation")
    harn = acc(rows, "harness")
    b = sum(r["harness"] == r["gold"] and r["simulation"] != r["gold"] for r in rows)
    c = sum(r["simulation"] == r["gold"] and r["harness"] != r["gold"] for r in rows)
    print(
        f"  {name:22}  n={len(rows):4d}  sim={sim:5.1f}  harn={harn:5.1f}  "
        f"Δ={harn - sim:+5.1f}  McNemar p={exact_mcnemar_p(b, c):.2e}"
    )


def hard_pair_rows(rows):
    """Both-members-correct items collapsed to one score per pair_id."""
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("pair_id")].append(r)
    out = []
    for pid, members in groups.items():
        labs = sorted(m["gold"] for m in members)
        if labs != ["ANSWER", "ASK"] or len(members) != 2:
            continue
        out.append({
            "pair_id": pid,
            "gold": "PAIR",
            "simulation": "PAIR" if all(m["simulation"] == m["gold"] for m in members) else "MISS",
            "harness": "PAIR" if all(m["harness"] == m["gold"] for m in members) else "MISS",
        })
    return out


def ultra_mask(eval_items, train_items):
    tbl = {norm_text(r.get("table_csv", "")) for r in train_items}
    req = {norm_text(r.get("request", "")) for r in train_items}
    return [
        norm_text(r.get("table_csv", "")) not in tbl and norm_text(r.get("request", "")) not in req
        for r in eval_items
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", help="eval_peritem.jsonl from eval/evaluate.py")
    parser.add_argument("--eval", default="data/eval.jsonl")
    parser.add_argument("--train", default="data/sft_mixed.jsonl",
                        help="composite SFT mix used to define ultra-clean / hardest pairs")
    args = parser.parse_args()

    dump_rows = load_jsonl(args.dump)
    eval_items = load_jsonl(args.eval)
    train_items = load_jsonl(args.train)
    ultra = ultra_mask(eval_items, train_items)

    by_model = defaultdict(list)
    for row in dump_rows:
        by_model[row["model"]].append(row)

    for model, rows in by_model.items():
        if len(rows) != len(eval_items):
            print(f"\n{model}  n_dump={len(rows)} n_eval={len(eval_items)} (align by index skipped)")
            report_slice("all", rows)
            report_slice("single", [r for r in rows if r.get("axes") == 1])
            report_slice("composite", [r for r in rows if (r.get("axes") or 1) >= 2])
            report_slice("hard pair", hard_pair_rows(rows))
            continue
        for r, item, is_ultra in zip(rows, eval_items, ultra):
            r["axes"] = item.get("axes")
            r["pair_id"] = item.get("pair_id")
            r["source"] = item.get("source")
            r["_ultra"] = is_ultra
        print(f"\n{model}  n={len(rows)}")
        report_slice("all", rows)
        report_slice("single", [r for r in rows if r.get("axes") == 1])
        report_slice("composite", [r for r in rows if r.get("axes") == 2])
        report_slice("hard pair", hard_pair_rows(rows))
        report_slice("ultra-clean composite", [r for r in rows if r.get("axes") == 2 and r["_ultra"]])
        report_slice("hardest pairs", hard_pair_rows([r for r in rows if r["_ultra"]]))
        for src in ("wikisql", "nature", "ds1000"):
            report_slice(src, [r for r in rows if r.get("source") == src])
        ultra_comp = [r for r in rows if r.get("axes") == 2 and r["_ultra"]]
        if ultra_comp:
            harn_ok = sum(r["harness"] == r["gold"] for r in ultra_comp)
            print(
                f"  ultra-clean binomial   n={len(ultra_comp)} harness_correct={harn_ok} "
                f"p={exact_binomial_two_sided(min(harn_ok, len(ultra_comp) - harn_ok), len(ultra_comp)):.2e}"
            )


if __name__ == "__main__":
    main()
