from vmodel_contracts import (
    AcceptanceCriterion, AcceptanceTest, Divergence, DivergenceKind, Priority,
    Requirement, RequirementKind, RequirementSet, TestResult, VModelLevel,
)
from verification_loop.diff_scoring import score_delta
from verification_loop.embeddings_trace import CodeSymbol, cosine, traceability_map
from vmodel_contracts import PunchListItem
from verification_loop.report import assemble_report


def _reqs():
    return RequirementSet(session_id="s", requirements=[
        Requirement(requirement_id="R-1", title="Booking", statement="Users book lessons",
                    priority=Priority.MUST, kind=RequirementKind.FUNCTIONAL, need_ids=["N-1"], intent_fields=["in_scope"],
                    acceptance=[AcceptanceCriterion(criterion_id="AC-1", statement="POST /bookings -> 201")]),
        Requirement(requirement_id="R-2", title="Cancel", statement="Users cancel lessons",
                    priority=Priority.MUST, kind=RequirementKind.FUNCTIONAL, need_ids=["N-1"], intent_fields=["in_scope"],
                    acceptance=[AcceptanceCriterion(criterion_id="AC-2", statement="DELETE /bookings/id -> 204")]),
    ])


def test_traceability_flags_orphans():
    # fake embed: identical texts -> 1.0, else 0.0
    def fake_embed(texts):
        return [[1.0, 0.0] if "R-1" in t else [0.0, 1.0] for t in texts]
    syms = [CodeSymbol("app.ts:createBooking", "R-1 Booking: Users book lessons")]
    m = traceability_map(_reqs(), syms, fake_embed, threshold=0.9)
    assert m["R-1"] == ["app.ts:createBooking"]
    assert m["R-2"] == []  # orphan: the report turns this into a divergence


def test_report_marks_failed_test_as_blocking_divergence():
    tests = [AcceptanceTest(test_id="T-1", level=VModelLevel.ACCEPTANCE, criterion_id="AC-1",
                            description="booking works", runner="http", definition={"preconditions": [], "steps": ["run"], "expected_outcomes": ["pass"], "cleanup": [], "cases": [], "execution": {}})]
    results = [TestResult(test_id="T-1", passed=False, evidence="POST /bookings -> 500")]
    rep = assemble_report("s", "http://localhost:3000", _reqs(), tests, results,
                          {"R-1": ["app.ts:createBooking"], "R-2": []}, [], [])
    assert rep.verdict.value == "diverges"
    assert any(d.detection == "deterministic" and d.severity == "blocking" for d in rep.divergences)
    assert any(d.kind == DivergenceKind.MISSING_CAPABILITY and d.requirement_id == "R-2" for d in rep.divergences)


def test_diff_scoring_finds_resolved_and_regressions():
    pl = [PunchListItem(item_id="PL-1", description="booking 500s", blocks=["T-1"], severity="blocking")]
    prev = [TestResult(test_id="T-1", passed=False, evidence="500"), TestResult(test_id="T-2", passed=True, evidence="ok")]
    new = [TestResult(test_id="T-1", passed=True, evidence="201"), TestResult(test_id="T-2", passed=False, evidence="boom")]
    d = score_delta(prev, new, pl)
    assert d.resolved == ["PL-1"] and d.regressions == ["T-2"] and d.still_open == []


def test_cosine_sanity():
    assert cosine([1, 0], [1, 0]) > 0.99
    assert cosine([1, 0], [0, 1]) < 0.01



def test_runner_serializes_canonical_test_definition():
    from verification_loop.founder_runner import generate_runner

    test = AcceptanceTest(
        test_id="T-1", level=VModelLevel.ACCEPTANCE, criterion_id="AC-1",
        description="booking works", runner="semantic",
        definition={
            "preconditions": ["service is running"],
            "steps": ["create a booking"],
            "expected_outcomes": ["booking is persisted"],
            "cleanup": [],
            "cases": [{"name": "E-1", "inputs": ["valid request"],
                       "expected_outcomes": ["accepted"]}],
            "execution": {"url": "/bookings", "method": "POST", "expected_status": 201},
        },
    )
    runner = generate_runner([test])
    assert '"preconditions": [' in runner and '"name": "E-1"' in runner
