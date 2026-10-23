"""Passive observer of existing camera and head state sidecars.

Only settled-head frames are returned until camera exposure / actual encoder
time-synchronization is independently validated. No ROS/SDK commands.
"""
import json
import os
from pathlib import Path
import time
import numpy as np


def stationary(history):
    if len(history) < 8: return False
    latest = history[-1][0]
    recent = [r for r in history if 0 <= latest-r[0] <= 1_000_000_000]
    if len(recent) < 8 or recent[-1][0]-recent[0][0] < 800_000_000:
        return False
    stamps=np.asarray([r[0] for r in recent],int)
    if np.any(np.diff(stamps)<=0) or np.max(np.diff(stamps))>150_000_000:
        return False
    angles=np.asarray([r[1] for r in recent],float)
    return bool(np.max(np.ptp(angles,axis=0))<=np.radians(.15))


def observe(mapping,frame,head,now_ns,history):
    if frame.get("connected") is not True:
        raise ValueError("Camera disconnected")
    ft=int(frame["timestamp_monotonic_ns"])
    ht=int(head["timestamp_monotonic_ns"])
    if not (0<=now_ns-ft<=200_000_000 and 0<=now_ns-ht<=150_000_000):
        raise ValueError("Stale image or encoder")
    if abs(ft-ht)>100_000_000:
        raise ValueError("Camera/head feedback time mismatch")
    if head.get("head_angles_order")!=["yaw","pitch"]:
        raise ValueError("Head feedback ordering mismatch")
    q=np.asarray(head["head_angles_rad"],float)
    if q.shape!=(2,) or not np.isfinite(q).all():
        raise ValueError("Missing measured head feedback")
    if not stationary(history):
        raise ValueError("HEAD_MOVING_OR_NOT_SETTLED: exposure sync unverified")
    if frame.get("pair_dequeue_skew_ms",1e6)>10:
        raise ValueError("Stereo pair skew excessive")
    for field in ("sensor_ids", "image_size", "pixel_domain"):
        if field not in frame:
            raise ValueError("Camera frame contract missing: "+field)
    obs=frame["image_observations"]
    left=obs["left"]["yellow"]
    right=obs["right"]["yellow"]
    if len(left)!=1 or len(right)!=1:
        raise ValueError("Yellow ball absent or ambiguous")
    px=list(left[0]["pixel"])+list(right[0]["pixel"])
    out=mapping.query(px,q,sensor_ids=frame.get("sensor_ids"),
                      image_size=frame.get("image_size"),pixel_domain=frame.get("pixel_domain"))
    out.update(timestamp_monotonic_ns=ft, measurement_status="SETTLED_HEAD_RESEARCH",
               exposure_time_valid=False, training_eligible=False)
    return out


def save_latest(path, record):
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+".tmp")
    tmp.write_text(json.dumps(record,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf-8")
    os.replace(tmp,p)


def run(mapping,latest,head_state,output,duration=0.):
    from collections import deque
    hist=deque(maxlen=80)
    last_head,last_img=None,None
    start=time.monotonic()
    try:
        while duration<=0 or time.monotonic()-start<duration:
            result=dict(valid=False,frame_id="robot_base",units="m",xyz_m=None,
                        motion_enabled=False,research_only=True,training_eligible=False)
            try:
                frame=json.loads(Path(latest).read_text(encoding="utf-8"))
                head=json.loads(Path(head_state).read_text(encoding="utf-8"))
                now=time.monotonic_ns()
                stamp=int(head["timestamp_monotonic_ns"])
                q=np.asarray(head["head_angles_rad"],float)
                if last_head is not None and stamp<last_head:
                    hist.clear()
                if stamp!=last_head:
                    hist.append((stamp,q))
                    last_head=stamp
                result=observe(mapping,frame,head,now,hist)
                if frame["timestamp_monotonic_ns"]!=last_img:
                    print(json.dumps(result,ensure_ascii=False),flush=True)
                    last_img=frame["timestamp_monotonic_ns"]
            except (OSError,ValueError,TypeError,KeyError,IndexError) as exc:
                result["reason"]=str(exc)
            save_latest(output,result)
            time.sleep(.03)
    except KeyboardInterrupt:
        pass
    finally:
        save_latest(output,dict(valid=False,frame_id="robot_base",units="m",
                                xyz_m=None,reason="observer_stopped",motion_enabled=False))
