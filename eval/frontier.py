"""Evaluate frontier APIs with mental and execution-harness decisions.

Paper Table (frontier): n=300 composite, balanced 150/150, both decision modes,
exact two-sided McNemar, rows sorted by probe crash rate.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from cac.harness import extract_codes, harness_decide, stated_decision
from cac.prompt import SYSTEM
from cac.stats import exact_mcnemar_p

# Explicit output contract so a zero-shot frontier model emits probes our
# execution harness can run -- the same contract our SFT model learned from data
# (>=2 flush-left ```python blocks, each using the provided df and assigning
# `result`).  This levels the field: the frontier follows OUR protocol.
FRONTIER_CONTRACT = (
    "\n\nOutput format (follow EXACTLY):\n"
    "- Give at least TWO candidate interpretations of the request.\n"
    "- For EACH interpretation, write one fenced Python code block whose fence "
    "```python starts at the very beginning of the line (no indentation, no "
    "bullet or numbered list around the code block).\n"
    "- Each code block MUST use the already-loaded pandas DataFrame named `df` "
    "(do NOT re-read, re-create, or redefine the table) and MUST assign the "
    "final answer to a variable named `result`.\n"
    "- After the code blocks, output exactly one tag on its own line: "
    "<ASK_CLARIFICATION> if the interpretations would give different results on "
    "the CURRENT table, or <FINAL_ANSWER> if they converge."
)

DRY_RUN_COMPLETION = """<thinking>
One interpretation is to return the total value:
```python
result = df["value"].sum()
```
Another interpretation is to return the average value:
```python
result = df["value"].mean()
```
These produce different results on the current table.
</thinking>
<ASK_CLARIFICATION>"""


def build_messages(item):
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Table (CSV):\n{item['table_csv']}\n\nRequest: {item['request']}"
            + FRONTIER_CONTRACT,
        },
    ]


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(
        p * (1 - p) / n + z * z / (4 * n * n)
    )
    return [
        round(p, 4),
        round(max(0, center - half), 4),
        round(min(1, center + half), 4),
    ]


def score(records, decision_key):
    n = len(records)
    ask_items = sum(r["gold"] == "ASK" for r in records)
    answer_items = n - ask_items
    correct = sum(r[decision_key] == r["gold"] for r in records)
    under_ask = sum(
        r["gold"] == "ASK" and r[decision_key] == "ANSWER" for r in records
    )
    over_ask = sum(
        r["gold"] == "ANSWER" and r[decision_key] == "ASK" for r in records
    )
    malformed = sum(r[decision_key] == "MALFORMED" for r in records)
    return {
        "n": n,
        "ask_items": ask_items,
        "answer_items": answer_items,
        "accuracy": {"count": correct, "wilson95": wilson(correct, n)},
        "under_ask": {
            "count": under_ask,
            "denominator": ask_items,
            "wilson95": wilson(under_ask, ask_items),
        },
        "over_ask": {
            "count": over_ask,
            "denominator": answer_items,
            "wilson95": wilson(over_ask, answer_items),
        },
        "malformed": {
            "count": malformed,
            "denominator": n,
            "wilson95": wilson(malformed, n),
        },
    }


def mcnemar(records):
    harness_only = sum(
        r["frontier_harness"] == r["gold"]
        and r["frontier_mental"] != r["gold"]
        for r in records
    )
    mental_only = sum(
        r["frontier_mental"] == r["gold"]
        and r["frontier_harness"] != r["gold"]
        for r in records
    )
    discordant = harness_only + mental_only
    p_value = exact_mcnemar_p(harness_only, mental_only)
    return {
        "test": "two-sided exact McNemar",
        "harness_only_correct": harness_only,
        "mental_only_correct": mental_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def call_http(base_url, auth_token, model, item, max_tokens, timeout):
    """Direct POST to an Anthropic-compatible /v1/messages proxy (no SDK dep).

    The proxy routes every model family (gpt-4o, claude-*, gemini-*) through the
    same /v1/messages schema, so this single path covers all frontier baselines.
    """
    messages = build_messages(item)
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": messages[0]["content"],
        "messages": messages[1:],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/messages",
        data=data,
        headers={
            "content-type": "application/json",
            "x-api-key": auth_token,
            "authorization": f"Bearer {auth_token}",
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if "error" in body:
        raise RuntimeError("api error: " + json.dumps(body["error"])[:300])
    if "content" not in body:
        raise RuntimeError("bad response: " + json.dumps(body)[:300])
    return "".join(
        b.get("text", "") for b in body["content"] if b.get("type") == "text"
    )


def call_http_with_retries(base_url, auth_token, model, item, max_tokens, timeout, retries):
    for attempt in range(retries + 1):
        try:
            return call_http(base_url, auth_token, model, item, max_tokens, timeout)
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                raise RuntimeError(
                    f"http API failed after {retries + 1} attempt(s): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            delay = min(30, 2 ** attempt)
            print(
                f"API attempt {attempt + 1} failed ({type(exc).__name__}); "
                f"retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def make_client(provider, api_key, timeout):
    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required: pip install 'openai>=1.0'"
            ) from exc
        return OpenAI(api_key=api_key, timeout=timeout)

    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError(
            "The anthropic package is required: pip install anthropic"
        ) from exc
    return Anthropic(api_key=api_key, timeout=timeout)


def call_api(client, provider, model, item, max_tokens):
    messages = build_messages(item)
    if provider == "openai":
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    response = client.messages.create(
        model=model,
        system=messages[0]["content"],
        messages=messages[1:],
        temperature=0,
        max_tokens=max_tokens,
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )


def call_with_retries(client, provider, model, item, max_tokens, retries):
    for attempt in range(retries + 1):
        try:
            return call_api(client, provider, model, item, max_tokens)
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                raise RuntimeError(
                    f"{provider} API failed after {retries + 1} attempt(s): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            delay = min(30, 2**attempt)
            print(
                f"API attempt {attempt + 1} failed ({type(exc).__name__}); "
                f"retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def evaluate_item(item, idx, completion):
    gold = item["label"]
    if gold not in {"ASK", "ANSWER"}:
        raise ValueError(f"item {idx} has invalid label: {gold!r}")
    codes = extract_codes(completion)
    return {
        "idx": idx,
        "gold": gold,
        "frontier_mental": stated_decision(completion),
        "frontier_harness": harness_decide(codes, item["table_csv"]),
        "text": completion,
        "codes": codes,
    }


def load_items(path, limit):
    with open(path, encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    return items[:limit] if limit is not None else items


def output_path(model, requested):
    if requested:
        return Path(requested)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    return Path("artifacts") / f"frontier_{safe_model}.jsonl"


def run(args):
    if args.dry_run:
        sentinel = "GOLD_PLAUSIBLE_MUST_NOT_LEAK"
        items = [
            {
                "table_csv": "group,value\nA,1\nA,2\nB,10\n",
                "request": "What is the value?",
                "label": "ASK",
                "plausible": sentinel,
            }
        ]
        messages = build_messages(items[0])
        if sentinel in json.dumps(messages):
            raise AssertionError("gold plausible data leaked into the model prompt")
        get_completion = lambda item: DRY_RUN_COMPLETION
        dataset_name = "builtin_dry_run_fixture"
    else:
        items = load_items(args.dataset, args.limit)
        base_url = args.base_url or os.environ.get("ANTHROPIC_BASE_URL")
        auth_token = (
            args.auth_token
            or args.api_key
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if args.provider == "http":
            if not base_url or not auth_token:
                raise RuntimeError(
                    "http provider needs --base-url and --auth-token "
                    "(or env ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN)"
                )
            get_completion = lambda item: call_http_with_retries(  # noqa: E731
                base_url, auth_token, args.model, item,
                args.max_tokens, args.timeout, args.retries,
            )
        else:
            env_name = (
                "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
            )
            api_key = args.api_key or os.environ.get(env_name)
            if not api_key:
                raise RuntimeError(
                    f"Missing API key: pass --api-key or set {env_name}"
                )
            client = make_client(args.provider, api_key, args.timeout)
            get_completion = lambda item: call_with_retries(  # noqa: E731
                client, args.provider, args.model, item,
                args.max_tokens, args.retries,
            )
        dataset_name = str(args.dataset)

    out = output_path(args.model, args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = {}
    if not args.dry_run and args.resume and out.exists():
        for line in open(out, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done[r["idx"]] = r
        print(f"resume: {len(done)} items already done", flush=True)

    todo = [(idx, item) for idx, item in enumerate(items) if idx not in done]
    records = list(done.values())
    lock = threading.Lock()
    handle = open(out, "a" if done else "w", encoding="utf-8")

    def work(pair):
        idx, item = pair
        completion = get_completion(item)
        record = evaluate_item(item, idx, completion)
        with lock:
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{len(records)}/{len(items)} (idx {idx})", flush=True)
        return idx

    workers = 1 if args.dry_run else max(1, args.workers)
    if workers == 1:
        for pair in todo:
            work(pair)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, todo))
    handle.close()
    records.sort(key=lambda r: r["idx"])

    summary = {
        "provider": args.provider,
        "model": args.model,
        "dataset": dataset_name,
        "dry_run": args.dry_run,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frontier_mental": score(records, "frontier_mental"),
        "frontier_harness": score(records, "frontier_harness"),
        "mcnemar_harness_vs_mental": mcnemar(records),
    }
    summary_path = out.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2), flush=True)
    sim = summary["frontier_mental"]["accuracy"]["wilson95"][0] * 100
    harn = summary["frontier_harness"]["accuracy"]["wilson95"][0] * 100
    malf = summary["frontier_harness"]["malformed"]["wilson95"][0] * 100
    p_value = summary["mcnemar_harness_vs_mental"]["p_value"]
    print(
        f"{args.model:24}  malf={malf:5.1f}  sim={sim:5.1f}  harn={harn:5.1f}  "
        f"Δ={harn - sim:+5.1f}  p={p_value:.3g}",
        flush=True,
    )
    print(f"per_item={out} summary={summary_path}", flush=True)
    return out, summary_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("openai", "anthropic", "http"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="data/frontier300.jsonl")
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url", help="Anthropic-compatible proxy base URL (http provider)")
    parser.add_argument("--auth-token", help="auth token for the http provider")
    parser.add_argument("--workers", type=int, default=6, help="concurrent HTTP requests")
    parser.add_argument("--resume", action="store_true", help="skip idxs already in --out")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


if __name__ == "__main__":
    run(parse_args())
