"""Verifiable rewards for GRPO. ``reward_v2`` is Eq. (2) in the paper.

``orig`` pays only for harness-vs-label agreement (and format). That can be
maximized by any pair of blocks that happen to disagree. ``v2`` adds the
stated-tag term and coverage of the reference outcomes ``G``, so a
manufactured divergence scores strictly worse than an honest probe.
"""
from __future__ import annotations

import json

from cac.harness import extract_codes, harness_outcomes, stated_decision


def _completion_text(completion) -> str:
    if isinstance(completion, list):
        return completion[0]["content"]
    return str(completion)


def reward_orig(completions, table_csv, label, **kwargs) -> list[float]:
    rewards = []
    for i, completion in enumerate(completions):
        text = _completion_text(completion)
        codes = extract_codes(text)
        outcomes, crashed = harness_outcomes(codes, table_csv[i])
        reward = 0.0
        if "<thinking>" in text and "</thinking>" in text:
            reward += 0.2
        if len(codes) >= 2:
            reward += 0.3
        if crashed or len(codes) < 2:
            rewards.append(reward - 4.0)
            continue
        harness = "ASK" if len(outcomes) > 1 else "ANSWER"
        reward += 2.0 if harness == label[i] else -2.0
        rewards.append(reward)
    return rewards


def reward_v2(completions, table_csv, label, plausible, **kwargs) -> list[float]:
    rewards = []
    for i, completion in enumerate(completions):
        text = _completion_text(completion)
        codes = extract_codes(text)
        outcomes, crashed = harness_outcomes(codes, table_csv[i])
        reward = 0.0
        if "<thinking>" in text and "</thinking>" in text:
            reward += 0.2
        if len(codes) >= 2:
            reward += 0.3
        if crashed or len(codes) < 2:
            rewards.append(reward - 4.0)
            continue
        harness = "ASK" if len(outcomes) > 1 else "ANSWER"
        reward += 2.0 if harness == label[i] else -2.0
        tag = stated_decision(text)
        if tag == "MALFORMED":
            reward -= 0.5
        else:
            reward += 0.5 if tag == label[i] else -0.5
        gold_codes = json.loads(plausible[i]) if isinstance(plausible[i], str) else list(plausible[i])
        gold, gold_crash = harness_outcomes(gold_codes, table_csv[i])
        if gold and not gold_crash:
            overlap = len(outcomes & gold)
            reward += overlap / len(gold)
            if len(outcomes) > 1 and overlap == 0:
                reward -= 1.0
        rewards.append(reward)
    return rewards


REWARDS = {"orig": reward_orig, "v2": reward_v2}
