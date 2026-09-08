"""Diff scoring across iterations: which punch-list items moved, which didn't.
Pure text/code diffing - the recurring-revenue workflow (re-verify after the
founder's agent applies fixes) costs compute, not tokens."""
from __future__ import annotations

from dataclasses import dataclass

from vmodel_contracts import PunchListItem, TestResult


@dataclass(frozen=True)
class IterationDelta:
    resolved: list[str]        # punch-list item ids whose tests now pass
    still_open: list[str]
    regressions: list[str]     # test ids that passed before and fail now


def score_delta(prev_results: list[TestResult], new_results: list[TestResult],
                punch_list: list[PunchListItem]) -> IterationDelta:
    prev = {r.test_id: r.passed for r in prev_results}
    new = {r.test_id: r.passed for r in new_results}
    regressions = [tid for tid, ok in new.items() if not ok and prev.get(tid, False)]
    resolved, still_open = [], []
    for item in punch_list:
        blocked_tests = [tid for tid in item.blocks if tid.startswith("T-")]
        if blocked_tests and all(new.get(tid, False) for tid in blocked_tests):
            resolved.append(item.item_id)
        else:
            still_open.append(item.item_id)
    return IterationDelta(resolved=resolved, still_open=still_open, regressions=regressions)
