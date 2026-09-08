"""Assemble the VerificationReport: the demo's success artifact.

Success criterion (scoping doc): the report finds at least one real
divergence the founder cares about. Every divergence carries its detection
provenance (deterministic vs llm_judged) - the honest accounting of what the
deterministic-first design actually caught."""
from __future__ import annotations

from vmodel_contracts import (
    AcceptanceTest,
    Divergence,
    DivergenceKind,
    RequirementSet,
    TestResult,
    VerificationReport,
    VerificationVerdict,
)


def assemble_report(
    session_id: str,
    target_ref: str,
    requirements: RequirementSet,
    tests: list[AcceptanceTest],
    results: list[TestResult],
    trace_map: dict[str, list[str]],   # from embeddings_trace
    extra_divergences: list[Divergence],
    semantic_notes: list[str],
) -> VerificationReport:
    divergences = list(extra_divergences)

    result_by_test = {r.test_id: r for r in results}
    for t in tests:
        res = result_by_test.get(t.test_id)
        if res and res.passed is False:
            divergences.append(Divergence(
                kind=DivergenceKind.MISSING_CAPABILITY if t.criterion_id else DivergenceKind.CONTRADICTS_SPEC,
                summary=f"{t.test_id} failed: {t.description}",
                detail=res.evidence,
                test_id=t.test_id, severity="blocking",
                detection="deterministic",
            ))
        if res is None:
            divergences.append(Divergence(
                kind=DivergenceKind.UNVERIFIABLE,
                summary=f"{t.test_id} was not executed",
                detail="The founder-side runner produced no result for this test.",
                test_id=t.test_id, severity="important", detection="deterministic",
            ))

    for rid, symbols in trace_map.items():
        if not symbols:
            divergences.append(Divergence(
                kind=DivergenceKind.MISSING_CAPABILITY,
                summary=f"{rid}: no code realizes this requirement",
                detail="Embeddings traceability found no code symbol above threshold for this requirement.",
                requirement_id=rid, severity="important", detection="deterministic",
            ))

    blocking = any(d.severity == "blocking" for d in divergences)
    verdict = (VerificationVerdict.DIVERGES if divergences and blocking
               else VerificationVerdict.INCOMPLETE if any(d.kind == DivergenceKind.UNVERIFIABLE for d in divergences)
               else VerificationVerdict.DIVERGES if divergences
               else VerificationVerdict.CONFORMS)
    return VerificationReport(
        session_id=session_id, target_ref=target_ref,
        results=results, divergences=divergences, verdict=verdict,
        semantic_residue_notes=semantic_notes,
    )
