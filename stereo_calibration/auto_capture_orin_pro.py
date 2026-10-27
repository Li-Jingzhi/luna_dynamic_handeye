#!/usr/bin/env python3
"""Automatic stationary, distinct stereo checkerboard collection; Ctrl+C finishes."""
import argparse
import json
import signal
import time

import cv2
import numpy as np

from orin_pro_calibration_gui import App, ROOT, request, save_pair
from capture_settings import SETTINGS


class PoseGate:
    def __init__(self):
        self.anchor = None
        self.previous = None
        self.stable_since = None
        self.saved = []
        self.last_save = -float('inf')
        self.reason = "SHOW COMPLETE BOARD IN BOTH EYES"

    def ready(self, points, now):
        if points is None:
            self.previous = None
            self.anchor = None
            self.stable_since = None
            self.reason = "SHOW COMPLETE BOARD IN BOTH EYES / WAIT FOR LOCK"
            return False
        drift = float('inf') if self.previous is None else np.linalg.norm(points-self.previous, axis=1).mean()
        self.previous = points.copy()
        span = float('inf') if self.anchor is None else np.linalg.norm(points-self.anchor, axis=1).mean()
        if drift > 1.5 or span > 1.5:
            self.stable_since = now
            self.anchor = points.copy()
        distinct = all(np.linalg.norm(points-old, axis=1).mean() >= 6 for old in self.saved)
        stable = self.stable_since is not None and now-self.stable_since >= .5
        cooled = now-self.last_save >= .6
        self.reason = ('POSE ALREADY SAVED - MOVE / TILT BOARD' if not distinct else
                       'HOLD STILL (0.5s)' if not stable else 'SAVE COOLDOWN' if not cooled else 'READY')
        return distinct and stable and cooled

    def accepted(self, points, now):
        self.saved.append(points.copy())
        self.last_save = now
        self.stable_since = None
        self.previous = None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume")
    args=parser.parse_args()
    app = App(args.resume)
    # Collection is sequential: Ctrl+C sets a flag so an in-flight pair finishes writing.
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    gate = PoseGate()
    name = 'Luna AUTO - Ctrl+C in terminal ends capture'
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1280, 610)
    print('AUTO collection started. Show complete board in BOTH eyes and hold still.', flush=True)
    print('Change distance, position and tilt. Aim for 30-50 diverse pairs.', flush=True)
    print('Press Ctrl+C in THIS TERMINAL to end collection. Existing pairs are kept.', flush=True)
    count = len(list(app.output.glob('*_left.png')))
    # Restore duplicate filtering from saved image corners when resuming a session.
    for path in sorted(app.output.glob('*_left.png')):
        points=[]
        for eye in ('left','right'):
            image=cv2.imread(str(path.with_name(path.name.replace('_left.png',f'_{eye}.png'))),cv2.IMREAD_GRAYSCALE)
            if image is None:raise ValueError(f'Missing resume image: {path}')
            ok,c=cv2.findChessboardCornersSB(cv2.resize(image,(640,496)),app.pattern)
            if ok:points.append(c.reshape(-1,2))
        if len(points)==2:gate.saved.append(np.concatenate(points))
    pending=None
    save_ms=None
    last_error = None
    try:
        while not stopping:
            loop_started=time.perf_counter()
            if pending and pending.done():
                try:
                    count=pending.result()
                    gate.accepted(pending_points,time.monotonic())
                    save_ms=(time.perf_counter()-save_started)*1000
                    print(f'Saved pair {count:04d}; capture/transfer/validate/write={save_ms:.0f}ms',flush=True)
                except (ValueError,RuntimeError,OSError) as error:
                    print('Capture rejected:',error,flush=True)
                    gate.stable_since=None;gate.previous=None
                pending=None
            try:
                status = json.loads(request(app.client, '/status'))
                body = request(app.client, '/preview')
                image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError('Preview decode failed')
                corners = []
                for eye in range(2):
                    tile = image[:, eye*640:(eye+1)*640]
                    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                    ok, points = cv2.findChessboardCorners(gray, app.pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
                    if ok:
                        points = cv2.cornerSubPix(gray, points, (5,5), (-1,-1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, .01))
                        corners.append(points.reshape(-1,2))
                        cv2.drawChessboardCorners(tile, app.pattern, points, True)
                points = np.concatenate(corners) if len(corners) == 2 and status['locked'] else None
                ready = gate.ready(points, time.monotonic())
                text = gate.reason
                if ready and not stopping and pending is None:
                    pending_points=points.copy()
                    save_started=time.perf_counter()
                    pending=app.executor.submit(lambda:save_pair(request(app.client,'/capture'),app.output,app.pattern,app.square))
                if pending is not None:text='SAVING IN BACKGROUND - PREVIEW CONTINUES'
                header = np.zeros((105,1280,3),np.uint8)
                for i,line in enumerate([f"AUTO | CSI {SETTINGS['sensors']} | corners {app.pattern} | {app.square} mm | saved {count}",
                                          f'AE/AWB {"LOCKED" if status["locked"] else "WAITING"} | 50 Hz | Ctrl+C in terminal: finish collection',text[:130]]):
                    cv2.putText(header,line,(12,25+32*i),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),1)
                cv2.imshow(name,np.vstack([header,image]))
                (ROOT/'runtime'/'auto_capture_status.json').write_text(json.dumps(dict(saved_pairs=count,output=str(app.output),camera=status,wait_reason=text,loop_ms=round((time.perf_counter()-loop_started)*1000,1),last_save_ms=save_ms)))
                if cv2.waitKey(30)&255 in (27,ord('q')) or cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)<1:
                    stopping = True
                last_error = None
            except (OSError, RuntimeError, ValueError) as error:
                if str(error) != last_error:
                    print('Waiting:',error,flush=True);last_error=str(error)
                gate.ready(None,time.monotonic())
                if cv2.waitKey(100)&255 in (27,ord('q')):stopping=True
    finally:
        if pending:
            try:count=pending.result()
            except Exception as error:print('Final capture rejected:',error,flush=True)
        app.client.close()
        app.executor.shutdown(wait=True,cancel_futures=True)
        cv2.destroyAllWindows()
        summary=dict(saved_pairs=count,output=str(app.output),collection_finished=True)
        (app.output/'collection_summary.json').write_text(json.dumps(summary,indent=2))
        print(f'Collection finished: {count} pairs saved in {app.output}',flush=True)
        print('Fit when ready: /usr/bin/python3 portable_stereo_fisheye_calibration/calibrate.py '
              f'--images {app.output} --output stereo_calibration/output/{app.output.name}_fisheye.json --corners {app.pattern[0]} {app.pattern[1]} --square-mm {app.square}',flush=True)


if __name__ == '__main__':
    main()
