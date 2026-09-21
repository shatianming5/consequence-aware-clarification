"""CPU-only checks: Eq. (1), reward, ultra-clean mask, exact McNemar."""
from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cac.harness import extract_codes, harness_decide, stated_decision
from cac.proto_engine import label_from_csv
from cac.reward import reward_v2
from cac.stats import exact_mcnemar_p
from eval.analyze import ultra_mask

ITEM = {
    "label": "ASK",
    "table_csv": "x\n1\n2\n",
    "plausible": [
        {"desc": "sum", "code": "result = int(df['x'].sum())"},
        {"desc": "count", "code": "result = len(df)"},
    ],
}


def test_label_matches_execution():
    codes = [p["code"] for p in ITEM["plausible"]]
    assert harness_decide(codes, ITEM["table_csv"]) == "ASK"
    label, _outcomes, has_err = label_from_csv(ITEM["table_csv"], ITEM["plausible"])
    assert label == "ASK"
    assert has_err is False


def test_stated_decision_and_extract():
    text = (
        "<thinking>\nInterpretation 1:\n```python\nresult = 1\n```\n"
        "Interpretation 2:\n```python\nresult = 2\n```\n</thinking>\n"
        "<ASK_CLARIFICATION>"
    )
    assert extract_codes(text) == ["result = 1", "result = 2"]
    assert stated_decision(text) == "ASK"
    assert stated_decision("<FINAL_ANSWER>") == "ANSWER"
    assert stated_decision("no tag") == "MALFORMED"


def test_eq1_malformed_vs_collapse():
    table = "x\n1\n2\n"
    assert harness_decide(["result = 1", "result = 1"], table) == "ANSWER"
    assert harness_decide(["result = 1", "result = 2"], table) == "ASK"
    assert harness_decide(["result = 1", "raise ValueError('x')"], table) == "MALFORMED"


def test_reward_penalizes_manufactured_divergence():
    gold_codes = json.dumps([p["code"] for p in ITEM["plausible"]])
    honest = (
        "<thinking>\n```python\n"
        + ITEM["plausible"][0]["code"]
        + "\n```\n```python\n"
        + ITEM["plausible"][1]["code"]
        + "\n```\n</thinking>\n<ASK_CLARIFICATION>"
    )
    junk = (
        "<thinking>\n```python\nresult = 0\n```\n```python\nresult = 1\n```\n"
        "</thinking>\n<ASK_CLARIFICATION>"
    )
    rh = reward_v2([honest], [ITEM["table_csv"]], [ITEM["label"]], [gold_codes])[0]
    rj = reward_v2([junk], [ITEM["table_csv"]], [ITEM["label"]], [gold_codes])[0]
    assert rh > rj, (rh, rj)


def test_ultra_mask_drops_shared_table_or_request():
    train = [{"table_csv": "a\n1\n", "request": "shared"}]
    ev = [
        {"table_csv": "a\n1\n", "request": "other"},
        {"table_csv": "b\n1\n", "request": "shared"},
        {"table_csv": "c\n1\n", "request": "fresh"},
    ]
    assert ultra_mask(ev, train) == [False, False, True]


def test_exact_mcnemar_small():
    assert abs(exact_mcnemar_p(0, 0) - 1.0) < 1e-12
    p = exact_mcnemar_p(1, 9)
    assert 0.02 < p < 0.03


if __name__ == "__main__":
    test_label_matches_execution()
    test_stated_decision_and_extract()
    test_eq1_malformed_vs_collapse()
    test_reward_penalizes_manufactured_divergence()
    test_ultra_mask_drops_shared_table_or_request()
    test_exact_mcnemar_small()
    print("OK")
