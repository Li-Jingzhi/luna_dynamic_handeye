#!/usr/bin/env python3
"""Orin stereo dataset capture: common Gst clock, dual 10ms gate, VPI/CUDA rectification."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import threading
import time

import cv2
import numpy as np
from stereo_pairing import StereoPairer


def load_calibration(path, sensors):
    c = json.loads(Path(path).read_text())
    expected = [f'{eye}:csi://{sensor}' for eye, sensor in zip(('left','right'), sensors)]
    if c.get('sensor_ids') != expected or c.get('image_size') != [2560,1984]:
        raise ValueError('Calibration sensor order/resolution mismatch')
    if c.get('pixel_domain') != 'raw' or c.get('distortion_model') != 'opencv_fisheye':
        raise ValueError('Raw fisheye calibration required')
    return c


def mapping_parameters(c):
    scale = np.diag([.5,.5,1.])
    return [dict(Kin=(scale@np.asarray(c[f'{eye}_K']))[:2],
                 X=np.c_[np.asarray(c[f'rectify_R{i}']),np.zeros(3)],
                 Kout=(scale@np.asarray(c[f'rectify_P{i}'])[:,:3])[:2],
                 coeffs=np.asarray(c[f'{eye}_D'])) for i,eye in enumerate(('left','right'),1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--output', required=True, help='new dataset directory on Orin')
    parser.add_argument('--sensors', nargs=2, type=int, default=[0,1])
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    if min(args.sensors)<0 or args.sensors[0]==args.sensors[1] or not np.isfinite(args.seconds) or args.seconds<=0:
        parser.error('invalid sensors/duration')
    config = load_calibration(args.calibration, args.sensors)
    calibration_hash = hashlib.sha256(Path(args.calibration).read_bytes()).hexdigest()
    # Hardware libraries are loaded only when capture is explicitly invoked.
    import gi
    gi.require_version('Gst','1.0')
    from gi.repository import Gst
    import vpi
    Gst.init(None)
    maps = [vpi.WarpMap.fisheye_correction(vpi.WarpGrid((1280,992)),
            mapping=vpi.FisheyeMapping.EQUIDISTANT, **parameters) for parameters in mapping_parameters(config)]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    manifest = dict(pixel_domain='ideal_pinhole', image_size=[1280,992], sensor_ids=config['sensor_ids'],
        source_calibration_sha256=calibration_hash, pairing_limit_ms=10,
        camera_control_policy='auto5s_then_lock_ae_awb_50Hz', rectification='VPI_CUDA_EQUIDISTANT_LINEAR',
        exposure_time_valid=False, clock_id='gst_system_clock_this_process',
        robot_feedback_included=False,
        note='Image acquisition only: join actual robot feedback by verified timestamps before hand-eye training.',
        rectified_P1=(np.diag([.5,.5,1.])@np.asarray(config['rectify_P1'])).tolist(),
        rectified_P2=(np.diag([.5,.5,1.])@np.asarray(config['rectify_P2'])).tolist())
    (output/'capture_manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'source_calibration.json').write_bytes(Path(args.calibration).read_bytes())
    pairer = StereoPairer(max_skew_ms=10); mutex = threading.Lock()
    running = True; capture = None; failure = []; count=0; locked=False; controls=[]
    stale=0; unlocked=0
    def stop(*_):
        nonlocal running
        running=False
    signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGTERM,stop)
    def receive(sink,eye):
        arrival=time.monotonic_ns()
        sample=sink.emit('pull-sample')
        if sample is None:return Gst.FlowReturn.ERROR
        buffer=sample.get_buffer(); pts=int(buffer.pts)
        if pts==Gst.CLOCK_TIME_NONE:return Gst.FlowReturn.OK
        ok,mapping=buffer.map(Gst.MapFlags.READ)
        if ok:
            try:
                image=np.frombuffer(mapping.data,np.uint8).reshape(992,-1)[:,:1280*3].reshape(992,1280,3).copy()
            finally: buffer.unmap(mapping)
            try:
                with mutex:pairer.push(eye,(image,pts,arrival,int(capture.get_base_time())+pts))
            except ValueError as error:
                failure.append(str(error));return Gst.FlowReturn.ERROR
        return Gst.FlowReturn.OK
    descriptions=[]
    for eye,sensor in enumerate(args.sensors):
        descriptions.append(f'nvarguscamerasrc name=cam{eye} sensor-id={sensor} sensor-mode=0 aeantibanding=2 aelock=false awblock=false ! '
            'video/x-raw(memory:NVMM),width=2560,height=1984,format=NV12,framerate=30/1 ! '
            'nvvidconv ! video/x-raw,width=1280,height=992,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! '
            f'appsink name=sink{eye} emit-signals=true sync=false drop=true max-buffers=1')
    capture=Gst.parse_launch(' '.join(descriptions));capture.use_clock(Gst.SystemClock.obtain())
    cameras=[capture.get_by_name(f'cam{eye}') for eye in range(2)]
    for eye in range(2):capture.get_by_name(f'sink{eye}').connect('new-sample',receive,eye)
    started=time.monotonic()
    capture.set_state(Gst.State.PLAYING)
    try:
        with (output/'frames.jsonl').open('x') as records:
            while running and time.monotonic()-started<args.seconds:
                error=capture.get_bus().pop_filtered(Gst.MessageType.ERROR)
                if error:raise RuntimeError(str(error.parse_error()))
                if failure:raise RuntimeError(failure[0])
                with mutex:
                    pair=pairer.take_latest()
                    received=list(pairer.stats['received'])
                if not locked and time.monotonic()-started>=5 and min(received)>0:
                    for camera in cameras:
                        camera.set_property('aelock',True);camera.set_property('awblock',True)
                    locked=all(c.get_property('aelock') and c.get_property('awblock') for c in cameras)
                    controls=[]
                    for camera in cameras:
                        values={}
                        for key in ('aelock','awblock','aeantibanding','exposuretimerange','gainrange','ispdigitalgainrange','wbmode','tnr-mode','tnr-strength','ee-mode','ee-strength','saturation'):
                            value=camera.get_property(key)
                            values[key]=value if isinstance(value,(str,bool,float,int)) else str(value)
                        controls.append(values)
                if pair is None:time.sleep(.002);continue
                if not locked:unlocked+=1;continue
                if max(time.monotonic_ns()-f[2] for f in pair)>=200_000_000:stale+=1;continue
                source_skew=abs(pair[0][3]-pair[1][3])/1e6
                arrival_skew=abs(pair[0][2]-pair[1][2])/1e6
                if max(source_skew,arrival_skew)>10:raise AssertionError('Pair exceeded dual 10ms gate')
                with vpi.Backend.CUDA:
                    corrected=[np.asarray(vpi.asimage(pair[i][0]).remap(maps[i],interp=vpi.Interp.LINEAR).cpu()).copy() for i in range(2)]
                count+=1; files={}
                for i,eye in enumerate(('left','right')):
                    for kind,image in (('uncorrected',pair[i][0]),('rectified',corrected[i])):
                        name=f'{count:06d}_{eye}_{kind}.png'
                        if not cv2.imwrite(str(output/name),image):raise OSError('PNG write failed')
                        files[f'{eye}_{kind}']=name
                records.write(json.dumps(dict(pair_id=count,files=files,source_pts_ns=[f[1] for f in pair],
                    arrival_monotonic_ns=[f[2] for f in pair],source_clock_ns=[f[3] for f in pair],
                    pair_source_clock_skew_ms=source_skew,pair_dequeue_skew_ms=arrival_skew,
                    exposure_locked=locked,exposure_time_valid=False,source_calibration_sha256=calibration_hash))+'\n')
                records.flush()
                if args.preview:
                    cv2.imshow('CUDA rectified stereo',np.hstack([cv2.resize(im,(640,496)) for im in corrected]))
                    if cv2.waitKey(1)&255 in (27,ord('q')):running=False
    finally:
        capture.set_state(Gst.State.NULL)
        if args.preview:cv2.destroyAllWindows()
        summary=dict(saved_pairs=count,pairing_stats=pairer.stats,stale_drops=stale,unlocked_drops=unlocked,
                     controls_readback=controls,elapsed_s=time.monotonic()-started)
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
