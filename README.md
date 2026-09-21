# Consequence-Aware Clarification for Data Agents

Code for the AAAI 2027 paper **Consequence-Aware Clarification for Data Agents: Breaking the Illusion of Mental Sandboxes via Execution-Grounded RL**.

A data agent can execute, return a well-formed table, and still compute the wrong quantity, because the request was ambiguous and the model never checked whether the competing readings differ on *this* table. This repo moves that decision out of the model's weights and into a sandbox: the model writes pandas probes, the sandbox executes them, and ASK vs ANSWER is read off whether the outputs diverge (Eq. 1). Table 1 still reports Malformed when fewer than two probes run.

## Layout

| folder | what it is |
|---|---|
| `cac/` | Prompt, execution harness, RLVR reward, exact tests, ambiguity prototypes, source loaders |
| `train/` | SFT (`sft.py`) and GRPO (`rlvr.py`) |
| `eval/` | Local checkpoints (`evaluate.py`), Table 1/B readout (`analyze.py`), frontier APIs (`frontier.py`) |
| `data/` | Benchmark builder. jsonl datasets are not committed |
| `tests/` | CPU checks of the harness and reward |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

RLVR needs an extra pin that does not mix with the eval/SFT stack (TRL 0.19.1 on the older torch/transformers). See the commented block in `requirements.txt`.

## Check the harness (no GPU)

```bash
python tests/test_harness.py
```

## Evaluate a checkpoint (Table 1)

Same generations are scored twice: the stated tag (simulation) and the sandbox verdict (harness).

```bash
python eval/evaluate.py \
  --models sft=artifacts/models/sft_qwen_7b \
  --dataset data/eval.jsonl \
  --out artifacts/eval_results.json \
  --dump artifacts/eval_peritem.jsonl

python eval/analyze.py artifacts/eval_peritem.jsonl \
  --eval data/eval.jsonl \
  --train data/sft_mixed.jsonl
```

`analyze.py` prints single vs composite, hard pairs, the ultra-clean slice, and exact McNemar. Pass the eval jsonl and the SFT mix you built.

## Frontier table (\(n{=}300\) composite)

```bash
python eval/frontier.py --provider http --model claude-opus-4.8 \
  --dataset data/frontier300.jsonl \
  --base-url "$ANTHROPIC_BASE" --auth-token "$ANTHROPIC_TOKEN"
```

`--provider openai` / `anthropic` use the respective SDKs. `--dry-run` exercises the harness without an API.

## Train

```bash
python train/sft.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --data data/sft_mixed.jsonl \
  --out artifacts/models/sft_qwen_7b

python train/rlvr.py \
  --model artifacts/models/sft_qwen_7b \
  --data data/rlvr_comp.jsonl \
  --reward v2 \
  --out artifacts/models/rlvr_qwen_7b
```

`--reward v2` is Eq. (2). Point `--data` at a jsonl you built; none is committed.

Weights are not in git.
