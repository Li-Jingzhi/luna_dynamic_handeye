#!/usr/bin/env python3
"""Generate side-by-side rectified previews with horizontal guide lines."""
import argparse, json
from pathlib import Path
import cv2
import numpy as np

p=argparse.ArgumentParser(description=__doc__); p.add_argument("--calibration",required=True); p.add_argument("--images",required=True); p.add_argument("--output",default="rectified_preview"); p.add_argument("--count",type=int,default=5); a=p.parse_args()
c=json.loads(Path(a.calibration).read_text()); size=tuple(c["image_size"]); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
maps=[]
for eye,i in (("left",1),("right",2)):
    maps.append(cv2.fisheye.initUndistortRectifyMap(np.array(c[f"{eye}_K"]),np.array(c[f"{eye}_D"]),np.array(c[f"rectify_R{i}"]),np.array(c[f"rectify_P{i}"])[:,:3],size,cv2.CV_16SC2))
paths=sorted(Path(a.images).glob("*_left.png"))[:a.count]
if not paths: raise SystemExit("No *_left.png images found")
for n,left in enumerate(paths,1):
    right=left.with_name(left.name.replace("_left.png","_right.png")); frames=[]
    for path,mapping in zip((left,right),maps): frames.append(cv2.remap(cv2.imread(str(path)),*mapping,cv2.INTER_LINEAR))
    view=np.hstack([cv2.resize(x,(640,496)) for x in frames])
    for y in range(40,496,60): cv2.line(view,(0,y),(1280,y),(0,255,0),1)
    cv2.imwrite(str(out/f"{n:02d}.jpg"),view)
print(f"Saved {len(paths)} previews to {out}")
