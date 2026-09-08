"""The semantic residue: the ONLY LLM call in verification (Design B).

Deterministic tooling answers structure, execution, and matching. What remains
is judgment: 'does this behavior satisfy the INTENT of R-12?' That call is
irreducibly semantic (Cost Addendum section 3) and it is the product - do not
eliminate it, shrink it to only these calls.
"""
from __future__ import annotations

import json

from vmodel_contracts import Divergence, Requirement

JUDGE_PROMPT = """You are verifying whether a built web application satisfies a requirement's INTENT.
Requirement: {req}
Acceptance criteria: {acs}
Observed evidence from deterministic checks and test execution:
{evidence}

Answer STRICT JSON: {{"satisfied": true|false|uncertain, "reasoning": "one paragraph", "divergences": [{{"summary": "...", "severity": "blocking|important|nice-to-have"}}]}}.
Judge only what the evidence supports. 'uncertain' is a legitimate answer - it routes to a human."""


def judge_residue(requirement: Requirement, evidence: list[str], chat_model) -> tuple[bool | None, list[dict]]:
    acs = "\n".join(f"- {a.criterion_id}: {a.statement}" for a in requirement.acceptance)
    resp = chat_model.invoke(JUDGE_PROMPT.format(
        req=f"{requirement.requirement_id} {requirement.title}: {requirement.statement}",
        acs=acs, evidence="\n".join(evidence) or "(none)",
    ))
    data = json.loads(resp.content.strip().strip("`").removeprefix("json").strip())
    sat = data["satisfied"]
    return (True if sat is True else False if sat is False else None), data.get("divergences", [])
