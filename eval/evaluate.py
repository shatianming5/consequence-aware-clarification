"""Score the same generations twice: stated tag (simulation) vs executed probes (harness)."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cac.harness import extract_codes, harness_decide, stated_decision
from cac.prompt import SYSTEM


def load_dataset(path, holdout=None):
    rows = [json.loads(line) for line in open(path)]
    return rows[-holdout:] if holdout else rows


def gold_label(item: dict) -> str:
    return item["label"]


def build_prompt(tokenizer, item: dict) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Table (CSV):\n{item['table_csv']}\n\nRequest: {item['request']}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def eval_model(path, items, batch_size=16):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    prompts = [build_prompt(tok, item) for item in items]
    mental, harness = [], []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        out = model.generate(**enc, do_sample=False, max_new_tokens=512, pad_token_id=tok.pad_token_id)
        for j in range(len(batch)):
            text = tok.decode(out[j][enc.input_ids.shape[1] :], skip_special_tokens=True)
            mental.append(stated_decision(text))
            harness.append(harness_decide(extract_codes(text), items[i + j]["table_csv"]))
        print(f"  {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return mental, harness


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(p, 4), round(max(0, center - half), 4), round(min(1, center + half), 4)]


def score(items, decisions):
    n = len(items)
    ask_items = answer_items = silent = over = correct = malformed = 0
    for item, dec in zip(items, decisions):
        gold = gold_label(item)
        if dec == "MALFORMED":
            malformed += 1
        if gold == "ASK":
            ask_items += 1
            if dec == "ANSWER":
                silent += 1
            if dec == "ASK":
                correct += 1
        else:
            answer_items += 1
            if dec == "ASK":
                over += 1
            if dec == "ANSWER":
                correct += 1
    return {
        "n": n,
        "ask_items": ask_items,
        "answer_items": answer_items,
        "silent_default": silent,
        "over_ask": over,
        "correct": correct,
        "malformed": malformed,
        "silent_default_rate": wilson(silent, ask_items),
        "over_ask_rate": wilson(over, answer_items),
        "decision_accuracy": wilson(correct, n),
        "malformed_rate": wilson(malformed, n),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="name=path pairs")
    parser.add_argument("--dataset", default="data/eval.jsonl")
    parser.add_argument("--holdout", type=int, default=0)
    parser.add_argument("--out", default="artifacts/eval_results.json")
    parser.add_argument("--dump", default="artifacts/eval_peritem.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    items = load_dataset(args.dataset, args.holdout or None)
    n_ask = sum(1 for item in items if gold_label(item) == "ASK")
    print(f"dataset n={len(items)} ASK={n_ask} ANSWER={len(items) - n_ask}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.dump)) or ".", exist_ok=True)
    results = {}
    with open(args.dump, "w") as dump:
        for spec in args.models:
            name, path = spec.split("=", 1) if "=" in spec else (os.path.basename(spec.rstrip("/")), spec)
            print(f"\n=== {name} ({path}) ===", flush=True)
            ment, harn = eval_model(path, items, batch_size=args.batch_size)
            results[name + "_simulation"] = score(items, ment)
            results[name + "_harness"] = score(items, harn)
            print(f"{name}_simulation acc={results[name + '_simulation']['decision_accuracy']}", flush=True)
            print(f"{name}_harness    acc={results[name + '_harness']['decision_accuracy']}", flush=True)
            for item, m, h in zip(items, ment, harn):
                dump.write(json.dumps({
                    "model": name,
                    "gold": gold_label(item),
                    "simulation": m,
                    "harness": h,
                    "source": item.get("source"),
                    "axes": item.get("axes"),
                    "family": item.get("family"),
                    "pair_id": item.get("pair_id"),
                    "variant": item.get("variant"),
                }) + "\n")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"EVAL_DONE {args.out}  peritem={args.dump}", flush=True)


if __name__ == "__main__":
    main()
