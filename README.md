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

## Local runs with Ollama (offline, $0)

The semantic-residue judge and the embeddings both have a local option:

Model size: `qwen2.5-coder:14b` is the standing default (proven on a 16GB M4); `qwen2.5-coder:7b` is the lighter fallback for tighter machines. Set
`OLLAMA_NUM_PARALLEL=1` so Ollama never loads two models at once (the chat
model plus `nomic-embed-text` would otherwise compete for RAM).

```bash
ollama pull qwen2.5-coder:14b nomic-embed-text
export VERIFICATION_LOOP_MODEL_RESIDUE_JUDGE=ollama:qwen2.5-coder:14b
# embeddings_trace.ollama_embed() instead of openai_embed()
```

Deterministic stages (runner scoring, Semgrep, schema checks, diff) need no
model at all. Founder-facing verdicts should come from the strong judge -
the verdicts are the product.
