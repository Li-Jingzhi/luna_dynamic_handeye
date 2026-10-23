"""Two-stage calibration, query, heldout evaluation and passive live CLI."""
import argparse
import json
import numpy as np
from .calibrate import fit_reference,fit_head
from .query import Mapping
from .records import load_rows,load_lens,write_new,prepare_multisession,write_jsonl_new
from .evaluate import evaluate


def main():
    ap=argparse.ArgumentParser(description="Two-DOF CAPT, passive observation only")
    sub=ap.add_subparsers(dest="cmd",required=True)
    prep=sub.add_parser("prepare-data",help="Unify batch-local reference metadata; preserve true measured angles")
    prep.add_argument("--data",nargs="+",required=True)
    prep.add_argument("--reference-head-deg",required=True,nargs=2,type=float)
    prep.add_argument("--output",required=True)
    ref=sub.add_parser("fit-reference",help="Fit fresh reference DLT from stationary multiXYZ pairs")
    ref.add_argument("--data",nargs="+",required=True)
    ref.add_argument("--lens")
    ref.add_argument("--allow-raw-research",action="store_true")
    ref.add_argument("--output",required=True)
    head=sub.add_parser("fit-head",help="Fit measured two-axis motion against reference DLT")
    head.add_argument("--data",nargs="+",required=True)
    head.add_argument("--reference",required=True)
    head.add_argument("--output",required=True)
    query=sub.add_parser("query",help="Pixel pair and measured head angles -> base XYZ")
    query.add_argument("--model",required=True)
    query.add_argument("--pixels",required=True,nargs=4,type=float)
    query.add_argument("--head-deg",required=True,nargs=2,type=float)
    ev=sub.add_parser("evaluate",help="Evaluate independent heldout data")
    ev.add_argument("--model",required=True)
    ev.add_argument("--data",required=True,nargs="+")
    ev.add_argument("--output")
    live=sub.add_parser("live",help="Passive, settled-head-only sidecar observer")
    live.add_argument("--model",required=True)
    live.add_argument("--latest",required=True,help="Calibrated SH5 1280x992 ROS2/Apex yellow sidecar")
    live.add_argument("--head-state",required=True,help="Measured head feedback sidecar on the same Orin clock")
    live.add_argument("--output",default="runtime/latest_base_xyz.json")
    live.add_argument("--duration",type=float,default=0.)
    live.add_argument("--allow-research-preview",action="store_true")
    inf=sub.add_parser("inspect-model")
    inf.add_argument("--model",required=True)
    a=ap.parse_args()
    if a.cmd=="prepare-data":
        rows=prepare_multisession(load_rows(a.data),a.reference_head_deg)
        write_jsonl_new(a.output,rows)
        print(json.dumps(dict(output=a.output,valid_stationary=len(rows),
                  reference_head_deg=a.reference_head_deg,
                  raw_pixels_head_feedback_and_base_xyz_preserved=True),
                  indent=2,ensure_ascii=False))
    elif a.cmd=="fit-reference":
        model=fit_reference(load_rows(a.data),lens=load_lens(a.lens),allow_raw_research=a.allow_raw_research)
        write_new(a.output,model)
        print(json.dumps(dict(output=a.output,reference_fit=model["reference_fit"],
                              research_only=True,independently_verified=False),indent=2,ensure_ascii=False))
    elif a.cmd=="fit-head":
        model=fit_head(load_rows(a.data),Mapping.load(a.reference).data)
        write_new(a.output,model)
        print(json.dumps(dict(output=a.output,dynamic_fit=model["dynamic_fit"],
                              research_only=True,independently_verified=False),indent=2,ensure_ascii=False))
    elif a.cmd=="query":
        print(json.dumps(Mapping.load(a.model).query(a.pixels,np.radians(a.head_deg)),indent=2,ensure_ascii=False))
    elif a.cmd=="evaluate":
        report=evaluate(Mapping.load(a.model),load_rows(a.data))
        if a.output: write_new(a.output,report)
        print(json.dumps(report,indent=2,ensure_ascii=False))
    elif a.cmd=="live":
        if not a.allow_research_preview:
            ap.error("Live not independently validated. Explicit --allow-research-preview required.")
        from .live import run
        run(Mapping.load(a.model),a.latest,a.head_state,a.output,a.duration)
    elif a.cmd=="inspect-model":
        m=Mapping.load(a.model).data
        fields=("stage","calibration_id","image_size","sensor_ids","reference_head_angles_rad",
                "research_only","independently_verified","fit_quality_passed","reference_fit","dynamic_fit")
        print(json.dumps({k:m.get(k) for k in fields},indent=2,ensure_ascii=False))

if __name__=="__main__": main()
