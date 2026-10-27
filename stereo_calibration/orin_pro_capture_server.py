#!/usr/bin/env python3
"""Raw full resolution CSI capture, with locked imaging and SSH-only HTTP access."""
import argparse
import io
import json
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import gi
import numpy as np
from stereo_pairing import StereoPairer

gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)
frames = [None, None]
pairer = StereoPairer(max_skew_ms=10)
sensors = [0, 1]
eye_names = ['left', 'right']
mutex = threading.Lock()
pipelines = []
cameras = []
locked = False
failure = None
started = time.monotonic()


def receive(sink, eye):
    arrival = time.monotonic_ns()
    sample = sink.emit('pull-sample')
    if sample is None:
        return Gst.FlowReturn.ERROR
    buffer = sample.get_buffer()
    ok, mapping = buffer.map(Gst.MapFlags.READ)
    if ok:
        caps = sample.get_caps().get_structure(0)
        width, height = caps.get_value('width'), caps.get_value('height')
        image = np.frombuffer(mapping.data, np.uint8).reshape(height, -1)[:, :width*3].reshape(height, width, 3).copy()
        buffer.unmap(mapping)
        with mutex:
            # PTS is the source pipeline timestamp, not a verified exposure time.
            pts = int(buffer.pts)
            if pts == Gst.CLOCK_TIME_NONE: return Gst.FlowReturn.OK
            frames[eye] = (image, pts, arrival, int(pipelines[0].get_base_time())+pts)
            pairer.push(eye, frames[eye])
    return Gst.FlowReturn.OK


def state():
    return dict(service='luna_stereo_direct_csi_v2', locked=locked, error=failure, sensors=sensors, image_size=[2560, 1984],
                controls=[dict(aelock=bool(c.get_property('aelock')),
                               awblock=bool(c.get_property('awblock')),
                               aeantibanding=int(c.get_property('aeantibanding'))) for c in cameras],
                frames=[None if f is None else dict(pts_ns=f[1], age_ms=(time.monotonic_ns()-f[2])/1e6) for f in frames],
                pairing_stats=pairer.stats,
                timestamp_basis='common_gst_clock_and_arrival_not_verified_exposure')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, body, kind='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/status':
            self.reply(200, json.dumps(state()).encode())
            return
        if self.path not in ('/preview', '/capture'):
            self.reply(404, b'{}')
            return
        with mutex:
            pair = list(frames) if self.path == '/preview' else pairer.take_latest()
        if pair is None or any(f is None or time.monotonic_ns()-f[2] > 200_000_000 for f in pair):
            self.reply(503, b'{"error":"waiting for fresh stereo frames"}')
            return
        if self.path == '/preview':
            view = np.hstack([cv2.resize(f[0], (640, 496)) for f in pair])
            self.reply(200, cv2.imencode('.jpg', view, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(), 'image/jpeg')
            return
        skew = (max(f[2] for f in pair)-min(f[2] for f in pair))/1e6
        source_skew = abs(pair[0][3]-pair[1][3])/1e6
        if not locked or skew > 10 or source_skew > 10:
            self.reply(409, json.dumps(dict(error='controls unlocked or source/arrival skew >10ms', skew_ms=skew)).encode())
            return
        metadata = state()
        metadata['pair_skew_ms'] = skew
        metadata['pair_source_clock_skew_ms'] = source_skew
        metadata['source_clock_ns'] = [f[3] for f in pair]
        metadata['exposure_time_valid'] = False
        metadata['stamp_ns'] = [f[1] for f in pair]
        metadata['dequeue_ns'] = [f[2] for f in pair]
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w', compression=zipfile.ZIP_STORED) as archive:
            for eye, f in zip(eye_names, pair):
                ok, png = cv2.imencode('.png', f[0])
                if not ok:
                    raise RuntimeError('PNG encoding failed')
                archive.writestr(eye+'.png', png.tobytes())
            archive.writestr('metadata.json', json.dumps(metadata))
        self.reply(200, data.getvalue(), 'application/zip')


def monitor():
    global locked, failure
    while True:
        for p in pipelines:
            message = p.get_bus().pop_filtered(Gst.MessageType.ERROR)
            if message:
                failure = str(message.parse_error())
                print(failure, flush=True)
        with mutex:
            ready = all(f is not None for f in frames)
        if not locked and ready and time.monotonic()-started >= 5:
            for camera in cameras:
                camera.set_property('aelock', True)
                camera.set_property('awblock', True)
            locked = all(c.get_property('aelock') and c.get_property('awblock') for c in cameras)
            print('AE/AWB locked, 50Hz antibanding', flush=True)
        time.sleep(.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensors", nargs=2, type=int, default=[0,1])
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    sensors = args.sensors
    if min(sensors) < 0 or sensors[0] == sensors[1]:
        parser.error("two distinct nonnegative CSI IDs required")
    descriptions = []
    for eye, sensor in enumerate(sensors):
        descriptions.append(
            f'nvarguscamerasrc name=cam{eye} sensor-id={sensor} sensor-mode=0 aeantibanding=2 aelock=false awblock=false ! '
            'video/x-raw(memory:NVMM),width=2560,height=1984,format=NV12,framerate=30/1 ! '
            'nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! '
            f'appsink name=sink{eye} emit-signals=true sync=false drop=true max-buffers=1')
    capture = Gst.parse_launch(' '.join(descriptions))
    capture.use_clock(Gst.SystemClock.obtain())
    pipelines.append(capture)
    for eye in range(2):
        capture.get_by_name(f'sink{eye}').connect('new-sample', receive, eye)
        cameras.append(capture.get_by_name(f'cam{eye}'))
    for p in pipelines:
        p.set_state(Gst.State.PLAYING)
    threading.Thread(target=monitor, daemon=True).start()
    try:
        ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
    finally:
        for p in pipelines:
            p.set_state(Gst.State.NULL)
