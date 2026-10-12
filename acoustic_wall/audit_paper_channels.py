"""Reproduce the independently tabulated single-channel audit (no curve fitting)."""
import json
from .paper_benchmark import CASES,ROOT,isolated_response


def main():
    for convention in ['inclusive','added']:
        for layout in ['centred','end_anchored']:
            path=ROOT/'results'/f'isolated_audit_{convention}_{layout}.json'
            if path.exists():continue
            rows=[dict(case=c,response=isolated_response(c,.00025,convention,layout)) for c in CASES]
            path.write_text(json.dumps(rows,indent=2))
    path=ROOT/'results'/'isolated_audit_inclusive_centred_refined.json'
    if not path.exists():
        rows=[dict(case=c,response=isolated_response(c,.0003,corner_step=.0000125)) for c in CASES]
        path.write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
