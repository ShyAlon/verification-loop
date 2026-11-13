import json
from pathlib import Path
from verification_loop.bundle_integrity import verify_bundle
from verification_loop.cli import main

def make(root:Path):
 from vmodel_contracts import CONTRACT_VERSION
 v=CONTRACT_VERSION  # Current-producer fixture follows the pinned contract release.
 (root/"specs").mkdir()
 def w(n,x): (root/n).write_text(json.dumps(x))
 w("needs.json",{"contract_version":v,"session_id":"s","needs":[{"contract_version":v,"need_id":"N-1","stakeholder":"u","statement":"x","kind":"functional","intent_fields":[],"intent_need_ids":["SN-1"]}]})
 req={"contract_version":v,"requirement_id":"R-1","title":"x","statement":"System shall x","priority":"must","kind":"functional","acceptance":[{"contract_version":v,"criterion_id":"AC-1","statement":"x"}],"need_ids":["N-1"],"intent_fields":[]}
 w("srs.json",{"contract_version":v,"session_id":"s","intent_model_ref":"s","purpose":"p","scope_summary":"s","functional":[req],"non_functional":[],"definitions":{},"assumptions":[],"constraints":[],"open_questions":[]})
 w("items.json",[{"contract_version":v,"spec_id":"SPEC-1","slug":"x","kind":"api_surface","title":"x","body_markdown":"x","requirement_ids":["R-1"],"depends_on":[]}]);(root/"specs/SPEC-1.md").write_text("x");(root/"AGENTS.md").write_text("x");(root/"architecture.md").write_text("x");(root/"needs.md").write_text("x");(root/"srs.md").write_text("x")
 definition={"contract_version":v,"preconditions":[],"steps":[],"expected_outcomes":[],"cleanup":[],"cases":[],"execution":{"contract_version":v}}
 w("tests.json",[{"contract_version":v,"test_id":"T-1","level":"acceptance","criterion_id":"AC-1","spec_id":None,"components":["booking-service"],"description":"x","runner":"http","definition":definition}])
 w("traceability.json",[{"contract_version":v,"requirement_id":"R-1","spec_ids":["SPEC-1"],"acceptance_criterion_ids":["AC-1"],"test_ids":["T-1"]}])
 w("manifest.json",{"contract_version":v,"bundle_version":"0.2.0","files":{"needs":"needs.md","srs":"srs.md","architecture":"architecture.md","agents":"AGENTS.md","traceability":"traceability.json","manifest":"manifest.json","SPEC-1":"specs/SPEC-1.md"},"milestones":[{"contract_version":v,"order":1,"title":"x","spec_ids":["SPEC-1"],"test_ids":[],"done_when":"x"}],"target_agent":"claude-code"})

def test_complete_delivered_bundle_passes(tmp_path): make(tmp_path);assert verify_bundle(tmp_path).ok
def test_mutation_fails_with_reverse_edge_path(tmp_path):
 make(tmp_path);p=tmp_path/"traceability.json";x=json.loads(p.read_text());x[0]["test_ids"]=[];p.write_text(json.dumps(x));r=verify_bundle(tmp_path);assert not r.ok and r.diagnostics[0]["path"]=="traceability/R-1/test_ids"
def test_cli_writes_report(tmp_path):
 make(tmp_path);out=tmp_path/"integrity.json";assert main([str(tmp_path),"--out",str(out)])==0 and json.loads(out.read_text())["ok"]
