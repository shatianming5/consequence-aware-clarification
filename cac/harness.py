"""Execute candidate pandas probes and read ASK/ANSWER off the outcome set.

Eq. (1): ASK iff the set of successful outcomes has size > 1, else ANSWER.
Table 1 still reports Malformed when fewer than two probes execute; accuracy
does not credit those rows. Eq. (2) is stricter: any exception is fmt−4.
"""
from __future__ import annotations

import contextlib
import io
import re
import textwrap

import pandas as pd

_CODE = re.compile(
    r"```[ \t]*(?:python|py)?[ \t]*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_codes(text: str) -> list[str]:
    blocks = _CODE.findall(text or "")
    return [textwrap.dedent(b).strip("\n") for b in blocks if b.strip()]


def execute_code(code: str, df: pd.DataFrame):
    loc = {"df": df.copy()}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, {"pd": pd}, loc)  # noqa: S102
        return loc.get("result", None)
    except Exception as exc:  # noqa: BLE001
        return "__ERR__:" + str(exc)


def normalize(value) -> str:
    if isinstance(value, str) and value.startswith("__ERR__"):
        return value
    try:
        if isinstance(value, pd.DataFrame):
            frame = value.reset_index(drop=True)
            frame.columns = list(map(str, frame.columns))
            frame = frame.sort_values(by=list(frame.columns), kind="stable").reset_index(drop=True)
            return "DF|" + frame.to_csv(index=False)
        if isinstance(value, pd.Series):
            return "S|" + "|".join(sorted(str(v) for v in value.tolist()))
        if isinstance(value, (list, tuple, set, frozenset)):
            return "S|" + "|".join(sorted(str(v) for v in value))
        return "N|" + repr(round(float(value), 6))
    except Exception:  # noqa: BLE001
        return "V|" + str(value).strip()


def run_probes(codes: list[str], table_csv: str) -> tuple[set[str], int]:
    """Successful outcome set O and how many probes ran."""
    if not codes:
        return set(), 0
    df = pd.read_csv(io.StringIO(table_csv))
    outcomes: set[str] = set()
    n_ok = 0
    for code in codes:
        result = normalize(execute_code(code, df))
        if str(result).startswith("__ERR__"):
            continue
        outcomes.add(result)
        n_ok += 1
    return outcomes, n_ok


def harness_outcomes(codes: list[str], table_csv: str) -> tuple[set[str], bool]:
    """O for Eq. (1), and whether any block crashed (Eq. 2 format penalty)."""
    outcomes, n_ok = run_probes(codes, table_csv)
    crashed = (not codes) or n_ok < len(codes)
    return outcomes, crashed


def eq1_verdict(outcomes: set[str]) -> str:
    return "ASK" if len(outcomes) > 1 else "ANSWER"


def harness_decide(codes: list[str], table_csv: str) -> str:
    """Table 1 harness decision: MALFORMED if <2 probes run, else Eq. (1)."""
    outcomes, n_ok = run_probes(codes, table_csv)
    if n_ok < 2:
        return "MALFORMED"
    return eq1_verdict(outcomes)


def stated_decision(text: str) -> str:
    asked = "<ASK_CLARIFICATION>" in (text or "")
    answered = "<FINAL_ANSWER>" in (text or "")
    if asked and not answered:
        return "ASK"
    if answered and not asked:
        return "ANSWER"
    return "MALFORMED"
