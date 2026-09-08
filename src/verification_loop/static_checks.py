"""Structure and interface checks: Semgrep rules derived FROM THE SPEC.

The spec's constraints become Semgrep rules ('no unauthenticated endpoints',
'no PII in logs'). Cost: serverless compute seconds, not tokens. If Semgrep
isn't installed the check reports itself as skipped - a missing tool is a
fact in the report, never a silent pass.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Finding:
    rule_id: str
    path: str
    line: int
    message: str
    severity: str


# Spec-constraint -> Semgrep rule skeletons. These grow with the archetype;
# the point of the demo is the mechanism (spec text -> executable rule),
# not a complete ruleset.
BUILTIN_RULES = {
    "no-secrets-in-source": {
        "id": "no-secrets-in-source",
        "message": "hardcoded secret (spec non-functional: secrets in env)",
        "severity": "ERROR",
        "languages": ["generic"],
        "patterns": [{"pattern-regex": r"(?i)(api[_-]?key|secret|password)\s*[:=]\s*['\"][^'\"]{8,}['\"]"}],
    },
    "no-pii-in-logs": {
        "id": "no-pii-in-logs",
        "message": "possible PII in log statement (spec non-functional: privacy)",
        "severity": "WARNING",
        "languages": ["javascript", "typescript", "python"],
        "patterns": [{"pattern-regex": r"(?i)(console\.(log|info)|logging?\.(info|debug))\(.*(email|password|ssn|phone)"}],
    },
}


def run_semgrep(target_dir: str | Path, rules: dict | None = None) -> tuple[list[Finding], bool]:
    """Returns (findings, ran). ran=False means Semgrep unavailable."""
    if shutil.which("semgrep") is None:
        return [], False
    rules = rules or BUILTIN_RULES
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as f:
        import yaml  # semgrep dep, present when semgrep is

        yaml.safe_dump({"rules": list(rules.values())}, f)
        rules_path = f.name
    proc = subprocess.run(
        ["semgrep", "--config", rules_path, "--json", "--quiet", str(target_dir)],
        capture_output=True, text=True, timeout=300,
    )
    findings = [
        Finding(rule_id=r["check_id"], path=r["path"], line=r["start"]["line"],
                message=r["extra"]["message"], severity=r["extra"]["severity"])
        for r in json.loads(proc.stdout or "{}").get("results", [])
    ]
    return findings, True
