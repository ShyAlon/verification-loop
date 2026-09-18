"""Independent loader/oracle for artifacts exactly as delivered on disk."""
from __future__ import annotations
import hashlib, json
from dataclasses import asdict, dataclass
from pathlib import Path
from vmodel_contracts import (AcceptanceTest, ItemSpec, RequirementSet, SpecManifest,
    StakeholderNeedsSet, TraceabilityEntry, validate_traceability_graph)

@dataclass(frozen=True)
class BundleIntegrityReport:
    bundle_dir: str
    files_checked: int
    graph_nodes: dict[str,int]
    diagnostics: list[dict]
    hashes: dict[str,str]
    @property
    def ok(self): return not self.diagnostics
    def to_dict(self): return {**asdict(self),"ok":self.ok}

def _read(path: Path): return json.loads(path.read_text())
def verify_bundle(bundle_dir: str|Path) -> BundleIntegrityReport:
    root=Path(bundle_dir); diagnostics=[]
    required=("needs.json","srs.json","items.json","tests.json","manifest.json","traceability.json")
    for name in required:
        if not (root/name).is_file(): diagnostics.append({"code":"missing-file","path":name,"message":"required delivered artifact is missing"})
    if diagnostics: return BundleIntegrityReport(str(root.resolve()),0,{},diagnostics,{})
    needs=StakeholderNeedsSet.model_validate(_read(root/"needs.json"))
    srs=_read(root/"srs.json")
    requirements=RequirementSet(session_id=srs["session_id"],requirements=(srs.get("functional") or [])+(srs.get("non_functional") or []))
    items=[ItemSpec.model_validate(x) for x in _read(root/"items.json")]
    tests=[AcceptanceTest.model_validate(x) for x in _read(root/"tests.json")]
    trace=[TraceabilityEntry.model_validate(x) for x in _read(root/"traceability.json")]
    manifest=SpecManifest.model_validate(_read(root/"manifest.json"))
    for logical,relative in manifest.files.items():
        p=root/relative
        if not p.is_file(): diagnostics.append({"code":"missing-manifest-file","path":f"manifest/files/{logical}","message":f"declared file missing: {relative}"})
    diagnostics.extend(asdict(x) for x in validate_traceability_graph(needs,requirements,items,tests,trace))
    test_ids={x.test_id for x in tests}; spec_ids={x.spec_id for x in items}
    for m in manifest.milestones:
        if set(m.spec_ids)-spec_ids: diagnostics.append({"code":"unknown-milestone-spec","path":f"manifest/milestones/{m.order}","message":str(sorted(set(m.spec_ids)-spec_ids))})
        if set(m.test_ids)-test_ids: diagnostics.append({"code":"unknown-milestone-test","path":f"manifest/milestones/{m.order}","message":str(sorted(set(m.test_ids)-test_ids))})
    checked=sorted({*required,*manifest.files.values()})
    hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in checked if (root/n).is_file()}
    return BundleIntegrityReport(str(root.resolve()),len(hashes),{"needs":len(needs.needs),"requirements":len(requirements.requirements),"specs":len(items),"tests":len(tests)},diagnostics,hashes)
