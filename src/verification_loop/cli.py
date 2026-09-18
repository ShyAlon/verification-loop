from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from .bundle_integrity import verify_bundle

def main(argv=None):
 p=argparse.ArgumentParser(prog="verify-bundle");p.add_argument("bundle_dir",type=Path);p.add_argument("--out",type=Path)
 a=p.parse_args(argv)
 try: report=verify_bundle(a.bundle_dir)
 except Exception as e:
  print(f"Error: bundle integrity could not be checked.\nLikely cause: {type(e).__name__}: {str(e)[:400]}\nNext: inspect the named artifact, regenerate it if stale, then rerun verify-bundle.",file=sys.stderr);return 2
 text=json.dumps(report.to_dict(),indent=2)
 if a.out:a.out.write_text(text+"\n")
 print(f"boundary=bundle-integrity files={report.files_checked} nodes={report.graph_nodes} diagnostics={len(report.diagnostics)} decision={'pass' if report.ok else 'fail'}",file=sys.stderr)
 print(text)
 return 0 if report.ok else 1
