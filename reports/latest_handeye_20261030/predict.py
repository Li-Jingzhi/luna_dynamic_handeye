"""Four corrected stereo pixels + measured head angles -> diagnostic torso XYZ."""
import argparse
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from dynamic_handeye.geometry import Assets
from dynamic_handeye.empirical import predict_pixels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=str(HERE/'model.json'))
    p.add_argument('--pixels',nargs=4,type=float,required=True,metavar=('UL','VL','UR','VR'))
    p.add_argument('--head',nargs=2,type=float,required=True,metavar=('YAW','PITCH'))
    p.add_argument('--diagnostic',action='store_true')
    a=p.parse_args()
    model=json.loads(Path(a.model).read_text())
    result=predict_pixels(model,Assets(),a.pixels,a.head,diagnostic=a.diagnostic)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
