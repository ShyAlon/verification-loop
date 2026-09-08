"""Contract/schema conformance: does the build's API surface and data models
validate against the JSON Schemas in the spec bundle? Pure stdlib + contracts;
negligible cost, high signal."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SchemaViolation:
    schema_name: str
    instance_path: str
    message: str


def validate_against_schemas(bundle_dir: str | Path, build_schema_dump: dict) -> list[SchemaViolation]:
    """`build_schema_dump`: {schema_name: instance} extracted from the build
    (e.g. its OpenAPI components, its migrations). We validate with
    jsonschema if available, else report the check as unverifiable upstream
    by returning a single UNVERIFIABLE-shaped violation only when schemas
    exist but the library is missing."""
    schemas_dir = Path(bundle_dir) / "schemas"
    if not schemas_dir.exists():
        return []
    try:
        import jsonschema
    except ImportError:
        return [SchemaViolation("(runner)", "-", "jsonschema not installed; schema conformance unverified")]
    violations: list[SchemaViolation] = []
    for schema_file in sorted(schemas_dir.glob("*.schema.json")):
        name = schema_file.name.removesuffix(".schema.json")
        if name not in build_schema_dump:
            continue  # build doesn't claim this surface; traceability flags absence
        schema = json.loads(schema_file.read_text())
        try:
            jsonschema.validate(build_schema_dump[name], schema)
        except jsonschema.ValidationError as e:
            violations.append(SchemaViolation(name, "/".join(map(str, e.absolute_path)) or "/", e.message))
    return violations
