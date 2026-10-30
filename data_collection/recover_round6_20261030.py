"""Recover this recorded round6 boundary, then continue the original remaining plan."""
import contextlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data_collection import workflow as w
from interactive_ball_calibration.capture import replay,read_rt,remaining_capture_path,arm_limits
from dynamic_handeye.geometry import Assets


def main():
    directory=ROOT/'datasets/raw/shared_fast_grid_20261029'
    output=directory/'head_05.jsonl';metadata=json.loads(output.with_suffix('.session.json').read_text())
    saved=json.loads(output.with_suffix('.completed.json').read_text());assets=Assets()
    if saved['complete'] or saved['observations']!=182 or w.digest(output)!=saved['sha256']:
        raise ValueError('This recovery requires the exact incomplete182-point round6 file')
    for filename,sha in metadata['asset_hashes'].items():
        if w.digest(Path(filename))!=sha:raise ValueError('Recovery assets changed')
    trajectory_path=Path(metadata['trajectory']);trajectory=w.validate_trajectory(trajectory_path,assets)
    rows=[json.loads(s) for s in output.read_text().splitlines() if s.strip()]
    order=remaining_capture_path(trajectory['waypoints'],rows,'forward',5)
    assert rows[-1]['waypoint_index']==905 and order[0]==906
    head=np.array([10.,-20.]);initial=json.loads((directory/'initial_feedback.json').read_text())
    backup=directory/'recovery_round6_182points'
    backup.mkdir(exist_ok=False)
    for file in (output,output.with_suffix('.completed.json')):shutil.copy2(file,backup/file.name)
    w.write_json(backup/'provenance.json',dict(reason='35s arm settle timeout; no source records removed',source_sha256=saved['sha256'],last_measurement_index=905,resume_motion_index=906))
    code=1
    try:
        with w.connected(ROOT.parent/'00040701','6.6.7.190') as (robot,sdk):
            rt=read_rt(robot)
            for value,reference in ((rt['body']['fb_pos'],initial['body']['fb_pos']),
                                    (rt['arms'][0]['fb']['fb_pos'],initial['arms'][0]['fb']['fb_pos']),
                                    (rt['head']['fb_pos'][:2],head)):
                if np.max(abs(np.array(value)-reference))>.3:raise RuntimeError('Recovery fixed pose changed')
            tail=dict(trajectory,waypoints=[trajectory['waypoints'][i] for i in order])
            w.model_check(tail,rt,head,directory/'recovery_round6_model_check.json','forward')
            try:
                w.start_receiver(w.LATEST,output,directory/'viewer.log')
                w.write_json(directory/'status.json',dict(phase='replay',round=6,total_rounds=9,head=head.tolist(),direction='forward',recovery=True,completed_positions_before_recovery=182))
                print('Recovering round6: preserve182 records, capture remaining20 positions.',flush=True)
                with (directory/'capture_05.log').open('a',buffering=1) as log,contextlib.redirect_stdout(log):
                    replay(robot,sdk,w.LATEST,assets.stereo,assets.hashes['stereo'],trajectory_path,output,
                           assets.hashes['mdh'],arm_limits(robot),head,motion_confirmed=True,
                           direct_approach=True,direction='forward',capture_stride=5,resume_existing=True)
                code=0
            finally:
                # Explicit cleanup even when the vendor link flag becomes unreliable on timeout.
                for obj in (sdk.FXObjType.OBJ_ARM1,sdk.FXObjType.OBJ_HEAD):
                    ret=robot.switch_to_idle(obj,2000)
                    if ret!=0:raise RuntimeError('Recovery IDLE failed: '+str(ret))
    finally:
        summary=w.summarize(output,1003,head,code,assets,5)
        w.write_json(output.with_suffix('.completed.json'),summary)
    if not summary['complete']:raise RuntimeError('Recovery incomplete; no later heads started')
    w.load_capture([output],assets)
    print('Round6 recovery complete; continuing heads7-9.',flush=True)
    command=[sys.executable,'-u',str(ROOT/'data_collection/workflow.py'),'--session','shared_fast_grid_20261029',
             '--trajectory',str(trajectory_path),'--plan',str(ROOT/'trajectories/shared_replay/right_20261029_1003points/capture_plan.json'),
             '--resume','--continue-low-valid','--run']
    raise SystemExit(w.run_child(command))


if __name__=='__main__':main()
