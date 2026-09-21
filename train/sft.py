"""Warm-start SFT: the model writes two probes and a decision tag."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import io
import json
import os
import random
import string

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from cac.prompt import SYSTEM

os.environ["WANDB_DISABLED"] = "true"


class LossPrinter(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(
                "STEP %d loss=%.4f lr=%.2e epoch=%.3f"
                % (state.global_step, logs["loss"], logs.get("learning_rate", 0.0), logs.get("epoch", 0.0)),
                flush=True,
            )


def perturb_csv(csv_str: str) -> str:
    try:
        df = pd.read_csv(io.StringIO(csv_str))
        df = df.sample(frac=1).reset_index(drop=True)
        df["id_" + "".join(random.choices(string.ascii_lowercase, k=4))] = [
            random.randint(1000, 9999) for _ in range(len(df))
        ]
        out = io.StringIO()
        df.to_csv(out, index=False)
        return out.getvalue().strip()
    except Exception:  # noqa: BLE001
        return csv_str


def build_assistant_content(item: dict) -> str:
    think = "<thinking>\n"
    for i, probe in enumerate(item["plausible"]):
        think += f"Interpretation {i + 1}: {probe['desc']}\nCode {i + 1}:\n```python\n{probe['code']}\n```\n"
    if item["label"] == "ASK":
        think += "These interpretations produce DIFFERENT results on the given table.\n</thinking>\n<ASK_CLARIFICATION>"
    else:
        think += "These interpretations produce the SAME result on the given table.\n</thinking>\n<FINAL_ANSWER>"
    return think


def load_and_format_data(path, tokenizer, holdout, max_len):
    rows = [json.loads(line) for line in open(path)]
    train_rows = rows[:-holdout] if holdout > 0 else rows
    inputs, labels, masks = [], [], []
    for item in train_rows:
        csv_str = item["table_csv"]
        if not item.get("no_shuffle", False) and random.random() < 0.5:
            csv_str = perturb_csv(csv_str)
        messages = [
            {"role": "system", "content": item.get("system", SYSTEM)},
            {"role": "user", "content": item.get("user", f"Table (CSV):\n{csv_str}\n\nRequest: {item['request']}")},
            {"role": "assistant", "content": item["assistant"] if "assistant" in item else build_assistant_content(item)},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tok_out = tokenizer(text, truncation=True, max_length=max_len)
        input_ids = tok_out["input_ids"]
        label = list(input_ids)
        header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        mask_idx = 0
        for i in range(len(input_ids) - len(header)):
            if input_ids[i : i + len(header)] == header:
                mask_idx = i + len(header)
                break
        for i in range(mask_idx):
            label[i] = -100
        inputs.append(input_ids)
        labels.append(label)
        masks.append([1] * len(input_ids))
    return Dataset.from_dict({"input_ids": inputs, "labels": labels, "attention_mask": masks})


class DynamicPadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        ii, lb, am = [], [], []
        for f in feats:
            n = maxlen - len(f["input_ids"])
            ii.append(f["input_ids"] + [self.pad_id] * n)
            lb.append(f["labels"] + [-100] * n)
            am.append(f["attention_mask"] + [0] * n)
        return {
            "input_ids": torch.tensor(ii, dtype=torch.long),
            "labels": torch.tensor(lb, dtype=torch.long),
            "attention_mask": torch.tensor(am, dtype=torch.long),
        }


class MegabatchLengthSampler(Sampler):
    def __init__(self, lengths, batch_size, mega_factor=64, seed=42):
        self.lengths = lengths
        self.bs = batch_size
        self.mega = batch_size * mega_factor
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        idx = torch.randperm(len(self.lengths), generator=g).tolist()
        batches = []
        for start in range(0, len(idx), self.mega):
            mega = idx[start : start + self.mega]
            mega.sort(key=lambda i: self.lengths[i], reverse=True)
            for b in range(0, len(mega), self.bs):
                batches.append(mega[b : b + self.bs])
        order = []
        for p in torch.randperm(len(batches), generator=g).tolist():
            order.extend(batches[p])
        return iter(order)


class GroupedTrainer(Trainer):
    def __init__(self, *args, train_lengths=None, **kwargs):
        self._train_lengths = train_lengths
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, *args, **kwargs):
        if self._train_lengths is not None:
            return MegabatchLengthSampler(self._train_lengths, self.args.per_device_train_batch_size)
        return super()._get_train_sampler(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--data", default="data/sft_mixed.jsonl")
    parser.add_argument("--out", default="artifacts/models/sft_qwen_7b")
    parser.add_argument("--holdout", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    parser.add_argument("--train-lm-head", action="store_true")
    parser.add_argument("--no-gc", action="store_true")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = load_and_format_data(args.data, tok, args.holdout, args.max_len)
    print(f"SFT formatted {len(ds)} items", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation=args.attn,
    )
    model.enable_input_require_grads()
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if args.train_lm_head:
        targets.append("lm_head")
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, target_modules=targets, task_type="CAUSAL_LM"))

    run_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "sft_run")
    os.makedirs(run_dir, exist_ok=True)
    targs = TrainingArguments(
        output_dir=run_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        max_grad_norm=1.0,
        gradient_checkpointing=not args.no_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
    )
    trainer = GroupedTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=DynamicPadCollator(tok.pad_token_id),
        train_lengths=[len(x) for x in ds["input_ids"]],
        callbacks=[LossPrinter()],
    )
    trainer.train()
    merged = trainer.model.merge_and_unload()
    dummy = tok("Hello", return_tensors="pt").to(merged.device)
    with torch.no_grad():
        logits = merged(**dummy).logits
        if torch.isnan(logits).any().item():
            raise RuntimeError("merged checkpoint produced NaN logits")
    os.makedirs(args.out, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print("SFT_DONE", flush=True)


if __name__ == "__main__":
    main()
