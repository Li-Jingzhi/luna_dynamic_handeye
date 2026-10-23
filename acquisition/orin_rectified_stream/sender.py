#!/usr/bin/env python3
"""Orin SH5 formal measurement and independent low-rate preview publisher."""
import argparse
from collections import deque
import json
from pathlib import Path
import signal
import socket
import sys
import threading
import time

import cv2
import gi
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capt_head.four_ball_pose import estimate_pose9
from capt_head.measurement_contract import DEFAULT_CONTRACT, load_contract
from capt_head.subpixel_ball import stereo_measure
from capt_head.camera_controls import controls, pipeline_properties, validate_profile

gi.require_version("Gst", "1.0")
from gi.repository import Gst


def capture_pipeline(sensor, profile=None):
    return Gst.parse_launch(
        f"nvarguscamerasrc name=argus_{sensor} sensor-id={sensor} sensor-mode=0 "
        f"{pipeline_properties(profile, sensor)} ! "
        "video/x-raw(memory:NVMM),width=2560,height=1984,framerate=30/1,format=NV12 ! "
        "nvvidconv ! video/x-raw,format=BGRx,width=1280,height=992 ! "
        "videoconvert ! video/x-raw,format=BGR,width=1280,height=992 ! "
        "appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true")


def preview_pipeline(host, port, fps):
    return Gst.parse_launch(
        "appsrc name=src is-live=true format=time do-timestamp=true block=false "
        f"caps=video/x-raw,format=BGR,width=640,height=496,framerate={fps}/1 ! "
        "queue leaky=downstream max-size-buffers=1 ! videoconvert ! video/x-raw,format=I420 ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
        "nvv4l2h264enc bitrate=2000000 insert-sps-pps=true iframeinterval=12 ! "
        "h264parse config-interval=1 ! rtph264pay pt=96 config-interval=1 ! "
        f"udpsink host={host} port={port} sync=false async=false")


def run(args):
    config, calibration = load_contract(args.contract)
    if args.camera_profile:
        profile = json.loads(Path(args.camera_profile).read_text())
        config["camera_controls"] = validate_profile(profile) if profile is not None else None
    pose_config = pose_stereo = None
    if args.four_ball_config:
        pose_path = Path(args.four_ball_config).resolve()
        pose_config = json.loads(pose_path.read_text(encoding="utf-8"))
        if (pose_config.get("schema") != "luna_four_ball_orientation_v1" or
                pose_config.get("image_size") != config["image_size"] or
                pose_config.get("pixel_domain") != config["pixel_domain"]):
            raise ValueError("Four-ball orientation config disagrees with measurement contract")
        stereo_path = (pose_path.parent / pose_config["stereo_model"]).resolve()
        pose_stereo = json.loads(stereo_path.read_text(encoding="utf-8"))
    if args.check:
        print(f"Contract OK: {config['image_size']} measurement, {config['preview_size']} preview; "
              f"calibration {config['calibration_sha256']}")
        return
    import vpi
    Gst.init(None)
    scale = config["image_size"][0] / calibration["image_size"][0]
    S = np.diag([scale, scale, 1.])
    maps = [vpi.WarpMap.fisheye_correction(
        vpi.WarpGrid(tuple(config["image_size"])),
        Kin=(S @ np.asarray(calibration[f"{eye}_K"]))[:2],
        X=np.c_[np.asarray(calibration[f"rectify_R{i}"]), np.zeros(3)],
        Kout=(S @ np.asarray(calibration[f"rectify_P{i}"])[:, :3])[:2],
        coeffs=np.asarray(calibration[f"{eye}_D"]), mapping=vpi.FisheyeMapping.EQUIDISTANT)
        for i, eye in enumerate(("left", "right"), 1)]
    cameras = [capture_pipeline(i, config.get("camera_controls")) for i in range(2)]
    encoders = [preview_pipeline(args.host, port, config["preview_fps"]) for port in args.ports]
    sources = [pipe.get_by_name("src") for pipe in encoders]
    preview_enabled = [True, True]
    latest = [None, None]
    preview_latest = [None, None]
    pose_input = [None]
    pose_result = [None]
    serials = [0, 0]
    lock = threading.Lock()
    running = threading.Event()
    running.set()
    exposure_locked = threading.Event()
    locked_at = [0.]
    control_readback = []
    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else None
    evidence_run = time.time_ns()
    if evidence_dir:
        evidence_dir.mkdir(parents=True, exist_ok=True)
    metadata = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def receive(sink, eye):
        sample = sink.emit("pull-sample")
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            frame = np.ndarray((992, 1280, 3), np.uint8, buffer=mapped.data).copy()
        finally:
            buffer.unmap(mapped)
        with lock:
            serials[eye] += 1
            latest[eye] = (frame, serials[eye], time.monotonic_ns(), int(buffer.pts))
        return Gst.FlowReturn.OK

    for eye, pipe in enumerate(cameras):
        pipe.get_by_name("sink").connect("new-sample", receive, eye)
    for pipe in cameras + encoders:
        if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Camera or preview pipeline failed to start")

    def measure():
        previous = [None, None]
        processed = [0, 0]
        times = deque(maxlen=60)
        luma = [deque(maxlen=config["brightness_window_frames"]) for _ in range(2)]
        while running.is_set():
            with lock:
                pair = list(latest) if all(latest) and all(latest[i][1] != processed[i] for i in range(2)) else None
            if pair is None:
                time.sleep(.001)
                continue
            processed[:] = [item[1] for item in pair]
            with vpi.Backend.CUDA:
                rectified = [np.asarray(vpi.asimage(item[0]).remap(maps[i], interp=vpi.Interp.LINEAR).cpu())
                             for i, item in enumerate(pair)]
            with lock:
                preview_latest[:] = [(rectified[i], pair[i][1]) for i in range(2)]
            observation = stereo_measure(rectified, config, previous)
            if observation:
                previous[:] = [observation[eye]["center"] for eye in ("left", "right")]
                if pose_config is not None:
                    with lock:
                        pose_input[0] = ([rectified[0], rectified[1]], processed.copy(),
                                         observation, time.monotonic_ns())
            for i, frame in enumerate(rectified):
                if exposure_locked.is_set():
                    luma[i].append(float(cv2.mean(cv2.cvtColor(frame[::8, ::8], cv2.COLOR_BGR2GRAY))[0]))
            stable = all(len(values) == values.maxlen and
                         np.percentile(np.abs(np.asarray(values)-np.median(values)), 95)
                         <= config["max_luma_p95_delta"] for values in luma)
            times.append(time.monotonic())
            fps = (len(times)-1)/(times[-1]-times[0]) if len(times) > 1 else 0.
            skew_ms = abs(pair[0][2]-pair[1][2])/1e6
            quality = ("PASS" if observation and exposure_locked.is_set() and stable and
                       time.monotonic()-locked_at[0] >= config["post_lock_settle_s"] and
                       skew_ms <= config["max_pair_skew_ms"] else "REJECT")
            reasons = []
            if not observation:
                reasons.append("yellow_pair_not_detected")
            if not exposure_locked.is_set():
                reasons.append("imaging_controls_not_locked")
            if not stable:
                reasons.append("brightness_not_stable")
            if time.monotonic()-locked_at[0] < config["post_lock_settle_s"]:
                reasons.append("post_lock_settling")
            if skew_ms > config["max_pair_skew_ms"]:
                reasons.append("pair_dequeue_skew")
            record = {
                "schema": "luna_yellow_measurement_v1", "quality": quality,
                "rejection_reasons": reasons,
                "frame_number": processed.copy(),
                "timestamp_basis": "orin_host_dequeue_not_exposure",
                "source_timestamp_ns": [item[3] for item in pair],
                "capture_dequeue_timestamp_ns": [item[2] for item in pair],
                "exposure_time_valid": False, "pair_capture_skew_ms": skew_ms,
                "processing_latency_ms": (time.monotonic_ns()-max(item[2] for item in pair))/1e6,
                "measurement_fps": fps,
                "image_size": config["image_size"], "preview_size": config["preview_size"],
                "pixel_domain": config["pixel_domain"], "sensor_ids": config["sensor_ids"],
                "measurement_backend": config["measurement_backend"],
                "source_calibration_sha256": config["calibration_sha256"],
                "contract_sha256": config["contract_sha256"],
                "exposure_locked": exposure_locked.is_set(), "awb_locked": exposure_locked.is_set(),
                "antibanding_hz": config["antibanding_hz"], "brightness_stable": stable,
                "mean_luma": [float(np.mean(values)) if values else None for values in luma],
                "camera_controls": config.get("camera_controls"),
                "camera_profile_override": bool(args.camera_profile),
            }
            if observation:
                record.update(observation)
                record["quality"] = quality
                if evidence_dir and processed[0] % args.evidence_every == 0:
                    evidence = []
                    for i, eye in enumerate(("left", "right")):
                        detection = observation[eye]
                        x, y = np.rint(detection["center"]).astype(int)
                        radius = int(np.ceil(detection["ellipse"]["major_px"]/2))+12
                        x0, y0 = max(0, x-radius), max(0, y-radius)
                        patch = rectified[i][y0:y+radius+1, x0:x+radius+1]
                        filename = f"{evidence_run}_{eye}_{processed[i]}.png"
                        if not cv2.imwrite(str(evidence_dir/filename), patch):
                            raise OSError("Cannot save measurement ROI evidence")
                        evidence.append(dict(file=filename, origin=[int(x0), int(y0)],
                                             size=[int(patch.shape[1]), int(patch.shape[0])],
                                             pixel_domain=config["pixel_domain"],
                                             frame_number=processed[i]))
                    record["image_evidence"] = evidence
            record["camera_controls_readback"] = control_readback
            if pose_config is not None:
                with lock:
                    current_pose = None if pose_result[0] is None else dict(pose_result[0])
                if current_pose is not None:
                    age_ms = (time.monotonic_ns()-current_pose["timestamp_monotonic_ns"])/1e6
                    current_pose["age_ms"] = float(age_ms)
                    current_pose["fresh"] = bool(age_ms <= pose_config["max_pose_age_ms"])
                    record["four_ball_pose"] = current_pose
            metadata.sendto(json.dumps(record, allow_nan=False).encode(), (args.host, args.metadata_port))

    def pose_measure():
        if pose_config is None:
            return
        last_serial = None
        period = 1. / float(pose_config["target_pose_fps"])
        while running.is_set():
            started = time.monotonic()
            with lock:
                item = pose_input[0]
            if item is not None and item[1] != last_serial:
                frames, frame_number, yellow, stamp_ns = item
                try:
                    result = estimate_pose9(frames, yellow, pose_stereo, config, pose_config)
                except Exception as exc:
                    result = {"valid": False, "reason": f"four_ball_exception:{type(exc).__name__}"}
                result["frame_number"] = frame_number
                result["timestamp_monotonic_ns"] = int(stamp_ns)
                result["pose_fps_target"] = float(pose_config["target_pose_fps"])
                with lock:
                    pose_result[0] = result
                last_serial = frame_number
            time.sleep(max(0., period-(time.monotonic()-started)))

    def preview():
        sent = [0, 0]
        period = 1/config["preview_fps"]
        while running.is_set():
            started = time.monotonic()
            with lock:
                frames = list(preview_latest)
            for i, item in enumerate(frames):
                if not preview_enabled[i] or item is None or item[1] == sent[i]:
                    continue
                image = cv2.resize(item[0], tuple(config["preview_size"]), interpolation=cv2.INTER_AREA)
                payload = image.tobytes()
                buffer = Gst.Buffer.new_allocate(None, len(payload), None)
                buffer.fill(0, payload)
                if sources[i].emit("push-buffer", buffer) != Gst.FlowReturn.OK:
                    preview_enabled[i] = False
                    continue
                sent[i] = item[1]
            time.sleep(max(0., period-(time.monotonic()-started)))

    workers = [threading.Thread(target=measure, daemon=True),
               threading.Thread(target=preview, daemon=True)]
    if pose_config is not None:
        workers.append(threading.Thread(target=pose_measure, daemon=True))
    for worker in workers:
        worker.start()
    started = time.monotonic()
    signal.signal(signal.SIGINT, lambda *_: running.clear())
    signal.signal(signal.SIGTERM, lambda *_: running.clear())
    try:
        while running.is_set():
            if not workers[0].is_alive():
                raise RuntimeError("Formal measurement worker stopped")
            for pipe in cameras:
                message = pipe.get_bus().pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
                if message:
                    raise RuntimeError(str(message.parse_error()) if message.type == Gst.MessageType.ERROR else "Pipeline ended")
            for i, pipe in enumerate(encoders):
                message = pipe.get_bus().pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
                if message:
                    preview_enabled[i] = False
                    pipe.set_state(Gst.State.NULL)
            if not exposure_locked.is_set() and time.monotonic()-started >= config["ae_awb_lock_delay_s"]:
                locked = []
                for i, pipe in enumerate(cameras):
                    camera = pipe.get_by_name(f"argus_{i}")
                    camera.set_property("aelock", True)
                    camera.set_property("awblock", True)
                    locked.append(bool(camera.get_property("aelock") and camera.get_property("awblock") and
                                       int(camera.get_property("aeantibanding")) == 2))
                if all(locked):
                    if config.get("camera_controls"):
                        control_readback[:] = [{key: pipe.get_by_name(f"argus_{i}").get_property(key)
                                               for key in controls(config["camera_controls"]["eyes"][i])}
                                              for i, pipe in enumerate(cameras)]
                    locked_at[0] = time.monotonic()
                    exposure_locked.set()
            time.sleep(.01)
    finally:
        running.clear()
        for worker in workers:
            worker.join(timeout=2)
        metadata.close()
        for pipe in cameras + encoders:
            pipe.set_state(Gst.State.NULL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--camera-profile", help="interactive preview profile; does not change the formal contract")
    parser.add_argument("--host", default="6.6.7.101")
    parser.add_argument("--ports", type=int, nargs=2, default=(5000, 5002))
    parser.add_argument("--metadata-port", type=int, default=5004)
    parser.add_argument("--evidence-dir", help="save lossless rectified measurement ROIs")
    parser.add_argument("--evidence-every", type=int, default=10)
    parser.add_argument("--four-ball-config",
                        help="optional low-priority red/blue/green orientation config")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.evidence_every < 1:
        parser.error("--evidence-every must be positive")
    run(args)


if __name__ == "__main__":
    main()
