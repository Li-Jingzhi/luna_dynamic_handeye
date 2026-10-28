#!/usr/bin/env python3
"""Display rectified SSH/JPEG or RTP/H.264 previews; frames are not measurements."""
import argparse
import json
import time
import threading
from pathlib import Path

import cv2
import numpy as np

from capture_settings import SETTINGS, connect
from orin_pro_calibration_gui import request


class LatestReceiver:
    """Keep blocking video reads off the GUI thread and retain only the latest frame."""
    def __init__(self, capture):
        self.capture = capture
        self.latest = None
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        count = 0
        try:
            while not self.stop.is_set():
                ok, image = self.capture.read()
                if not ok:
                    self.error = 'UDP video frame unavailable'
                    break
                count += 1
                self.latest = (count, time.monotonic(), image)
        finally:
            self.capture.release()


def make_maps(calibration, size=(640, 496)):
    if calibration.get('distortion_model') != 'opencv_fisheye':
        raise ValueError('OpenCV fisheye calibration required')
    original = np.asarray(calibration['image_size'], dtype=float)
    scale = np.asarray(size) / original
    if not np.allclose(scale[0], scale[1]):
        raise ValueError('Preview and calibration aspect ratios differ')
    maps = []
    for eye, index in (('left', 1), ('right', 2)):
        K = np.array(calibration[f'{eye}_K'], dtype=float)
        P = np.array(calibration[f'rectify_P{index}'], dtype=float)[:, :3]
        K[:2] *= scale[0]
        P[:2] *= scale[0]
        maps.append(cv2.fisheye.initUndistortRectifyMap(
            K, np.array(calibration[f'{eye}_D'], dtype=float),
            np.array(calibration[f'rectify_R{index}'], dtype=float), P, size, cv2.CV_16SC2))
    return maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--snapshot', help='Update a JPEG and adjacent JSON status for verification')
    parser.add_argument('--udp', action='store_true', help='Receive RTP/H.264 on UDP ports 5100/5102 instead of SSH/JPEG')
    args = parser.parse_args()
    path = Path(args.calibration).resolve()
    calibration = json.loads(path.read_text())
    expected = [f'{eye}:csi://{sensor}' for eye, sensor in zip(('left', 'right'), SETTINGS['sensors'])]
    if calibration['sensor_ids'] != expected:
        raise ValueError('Calibration sensor order differs from capture configuration')
    maps = make_maps(calibration)
    client = None if args.udp else connect()
    captures = []
    receivers = []
    name = 'Luna V2.0 NEW CALIBRATION - RECTIFIED LEFT | RIGHT - Q close'
    frames = 0
    started = time.monotonic()
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1280, 550)
    try:
        if args.udp:
            for port in (5100, 5102):
                pipeline = (f'udpsrc port={port} buffer-size=4194304 caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000" '
                            '! rtpjitterbuffer latency=60 drop-on-latency=true ! rtph264depay ! h264parse '
                            '! openh264dec ! videoconvert ! video/x-raw,format=BGR '
                            '! appsink sync=false drop=true max-buffers=1')
                capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                captures.append(capture)
                if not capture.isOpened():
                    raise RuntimeError(f'Cannot open UDP receiver {port}')
                receivers.append(LatestReceiver(capture))
        else:
            status = json.loads(request(client, '/status'))
            if status.get('sensors') != SETTINGS['sensors'] or status.get('error'):
                raise RuntimeError(f'Camera service mismatch: {status}')
        previous = None
        started = time.monotonic()
        while True:
            if args.udp:
                latest = [receiver.latest for receiver in receivers]
                if any(receiver.error for receiver in receivers):
                    raise RuntimeError('UDP decoder stopped')
                serials = tuple(frame[0] if frame else 0 for frame in latest)
                if not all(latest) or serials == previous:
                    if cv2.waitKey(5) & 255 in (27, ord('q')):
                        break
                    continue
                previous = serials
                raw = np.hstack([frame[2] for frame in latest])
            else:
                raw = cv2.imdecode(np.frombuffer(request(client, '/preview'), np.uint8), cv2.IMREAD_COLOR)
            if raw is None or raw.shape != (496, 1280, 3):
                raise ValueError('Unexpected preview image size')
            eyes = [cv2.remap(image, *mapping, cv2.INTER_LINEAR)
                    for image, mapping in zip(np.hsplit(raw, 2), maps)]
            view = np.hstack(eyes)
            for y in range(40, 496, 60):
                cv2.line(view, (0, y), (1279, y), (0, 255, 0), 1)
            frames += 1
            fps = frames / (time.monotonic() - started)
            header = np.zeros((54, 1280, 3), np.uint8)
            transport = 'H264/UDP' if args.udp else 'JPEG/SSH'
            if args.udp:
                age_ms = [1000 * (time.monotonic() - frame[1]) for frame in latest]
                transport += f' | age {age_ms[0]:.0f}/{age_ms[1]:.0f}ms'
            cv2.putText(header, f'{transport} | PREVIEW ONLY | {fps:.1f} fps | Q close',
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .56, (0, 255, 255), 1)
            cv2.putText(header, f'LEFT CSI{SETTINGS["sensors"][0]}                         NEW LOCAL CALIBRATION                         RIGHT CSI{SETTINGS["sensors"][1]}',
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1)
            display = np.vstack([header, view])
            cv2.imshow(name, display)
            if args.snapshot and frames % 10 == 0:
                snapshot = Path(args.snapshot)
                if not cv2.imwrite(str(snapshot), display):
                    raise OSError(f'Cannot write {snapshot}')
                snapshot.with_suffix('.json').write_text(json.dumps(dict(
                    calibration=str(path), transport=transport, frames=frames, preview_fps=fps,
                    received_frames=list(previous) if args.udp else None,
                    updated_unix=time.time())))
            key = cv2.waitKey(1) & 255
            if key in (27, ord('q')) or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if client:
            client.close()
        for receiver in receivers:
            receiver.stop.set()
        for receiver in receivers:
            receiver.thread.join(timeout=1)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
