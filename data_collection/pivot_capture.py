"""Record a fixed-ball/multiple-flange-configuration diagnostic using existing drag capture."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data_collection.workflow import run_child, write_json
from dynamic_handeye.geometry import Assets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--site',required=True,help='Identifier of the physical fixed-ball fixture position')
    parser.add_argument('--sdk-root',type=Path,default=ROOT.parent/'00040701')
    parser.add_argument('--robot-ip',default='6.6.7.190')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists() or output.with_suffix('.session.json').exists():
        raise FileExistsError('Use a new output file')
    assets=Assets()
    command=[sys.executable,'-u','-m','capt_head.drag_capture','--pivot','--start',
             '--output',str(output),'--sdk-root',str(args.sdk_root.resolve()),
             '--robot-ip',args.robot_ip,'--mdh',str(ROOT/'configs/robot/luna_new.ini')]
    if not args.run:
        print(json.dumps(dict(command=command,motion=False,requires_fixed_ball_fixture=True)))
        return
    print('Keep the ball center fixed in the fixture. Vary flange attitude slowly; Ctrl+C finishes.',flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output.with_suffix('.session.json'),dict(purpose='fixed_ball_fk_consistency',site=args.site,
               measured_pf_m=[0,0,.101],assets=assets.hashes,
               operator_constraint='Ball center must remain physically fixed; this is not software-verified.'))
    code=run_child(command)
    if code:
        raise SystemExit(code)


if __name__=='__main__':
    main()
