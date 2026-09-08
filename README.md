# verification-loop

The differentiator. Answers the question no competitor answers: **does the
build match the spec?** - with a verdict report and a traceable punch list,
not vibes.

## Deterministic-first (Design B, ARCHITECTURE.md ADR-0004)

| Job | Technique | Marginal cost |
|---|---|---|
| Acceptance-test execution | `founder_runner.py` generates a stdlib runner the founder's agent executes in THEIR environment | $0 (their compute) |
| Requirement-to-code traceability | embeddings similarity + orphan flagging (`embeddings_trace.py`) | ~$0.01/run |
| Structure / interface checks | Semgrep rules derived from spec constraints (`static_checks.py`) | compute, cents |
| Contract / schema conformance | JSON Schema validation (`schema_checks.py`) | negligible |
| Iteration change scoring | diff scoring (`diff_scoring.py`) | negligible |
| **Semantic residue** | LLM judges only what the above cannot (`residue.py`) | ~$0.20-0.40/run |

The spec is authored machine-checkably (acceptance criteria as executable
assertions) precisely so this stage stays cheap. Every divergence in the
report carries its detection provenance: `deterministic` or `llm_judged`.

## Never on our infra

Tester code executes only in the tester's own environment at demo stage
(scoping doc Decision 3b). Sandboxed execution on our side is the first
post-demo investment, and it is a security project, not a feature.

## Boundary rule

Consumes `SpecBundle` + `AcceptanceTest` set + a build artifact reference
(repo, archive, or deployed URL). Emits `VerificationReport`. No sibling
imports.
