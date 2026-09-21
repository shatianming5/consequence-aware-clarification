"""Composite Exec-RLVR with GRPO. ``--reward v2`` is the paper's training signal."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
from dataclasses import fields

from cac.prompt import SYSTEM
from cac.reward import REWARDS

os.environ["WANDB_DISABLED"] = "true"


def cap_table(table_csv, max_chars=6000, max_rows=40) -> str:
    lines = table_csv.split("\n")
    if not lines:
        return table_csv
    kept = [lines[0]]
    total = len(lines[0])
    truncated = False
    for line in lines[1 : 1 + max_rows]:
        if total + len(line) + 1 > max_chars:
            truncated = True
            break
        kept.append(line)
        total += len(line) + 1
    if len(lines) - 1 > len(kept) - 1:
        truncated = True
    out = "\n".join(kept)
    if truncated:
        out += "\n... [table truncated for display; full table used for grading]"
    return out


def format_prompts(path):
    from datasets import Dataset

    rows = [json.loads(line) for line in open(path) if line.strip()]
    prompts, tables, plausibles, labels = [], [], [], []
    for item in rows:
        prompts.append([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Table (CSV):\n{cap_table(item['table_csv'])}\n\nRequest: {item['request']}"},
        ])
        tables.append(item["table_csv"])
        plausibles.append(json.dumps([p["code"] for p in item["plausible"]]))
        labels.append(item["label"])
    return Dataset.from_dict({
        "prompt": prompts,
        "table_csv": tables,
        "plausible": plausibles,
        "label": labels,
    })


def make_config(**kwargs):
    from trl import GRPOConfig

    valid = {f.name for f in fields(GRPOConfig)}
    accepted = {k: v for k, v in kwargs.items() if k in valid}
    dropped = [k for k in kwargs if k not in valid]
    if dropped:
        print(f"[grpo] dropped unsupported config args: {dropped}", flush=True)
    return GRPOConfig(**accepted)


def main():
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="warm-start SFT checkpoint")
    parser.add_argument("--data", default="data/rlvr_comp.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--reward", choices=sorted(REWARDS), default="v2")
    parser.add_argument("--holdout", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--bsz", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--num-gen", type=int, default=4)
    parser.add_argument("--max-comp", type=int, default=448)
    parser.add_argument("--steps-per-gen", type=int, default=4)
    args = parser.parse_args()

    full = format_prompts(args.data)
    train = full.select(range(len(full) - args.holdout)) if args.holdout > 0 else full
    print(f"RLVR prompts={len(train)} reward={args.reward}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    with torch.no_grad():
        dummy = tokenizer("Hello", return_tensors="pt").to(model.device)
        if torch.isnan(model(**dummy).logits).any().item():
            raise RuntimeError("warm-start checkpoint produced NaN logits")

    run_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"rlvr_run_{args.reward}")
    os.makedirs(run_dir, exist_ok=True)
    cfg = make_config(
        output_dir=run_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_gen,
        steps_per_generation=args.steps_per_gen,
        max_steps=args.max_steps,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        beta=0.01,
        max_completion_length=args.max_comp,
        max_prompt_length=768,
        gradient_checkpointing=True,
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[REWARDS[args.reward]],
        args=cfg,
        train_dataset=train,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train()
    merged = trainer.model.merge_and_unload()
    dummy = tokenizer("Hello", return_tensors="pt").to(merged.device)
    with torch.no_grad():
        if torch.isnan(merged(**dummy).logits).any().item():
            raise RuntimeError("merged RLVR checkpoint produced NaN logits")
    os.makedirs(args.out, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    print("RLVR_DONE", flush=True)


if __name__ == "__main__":
    main()
