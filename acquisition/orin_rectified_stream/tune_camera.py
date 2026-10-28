#!/usr/bin/env python3
"""Measure fixed Argus profiles with the formal detector, then test restarts."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capt_head.camera_controls import controls, roi_metrics, score_window
from capt_head.measurement_contract import load_contract
from capt_head.subpixel_ball import stereo_measure
from interactive_ball_calibration.geometry import stereo_sphere_center
from orin_rectified_stream.sender import capture_pipeline, preview_pipeline, Gst


def profile(exposure_ms, gain, wb=3, isp=0, digital=1., ee=None):
    ee = isp if ee is None else ee
    eye = dict(exposure_ns=int(exposure_ms*1e6), analog_gain=float(gain), digital_gain=float(digital),
               wbmode=wb, tnr_mode=isp, ee_mode=ee, saturation=1.,
               tnr_strength=.5 if isp else 0., ee_strength=.5 if ee else 0.)
    return dict(mode="fixed", eyes=[dict(eye), dict(eye)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed-centers", nargs=4, required=True, type=float)
    parser.add_argument("--host", default="6.6.7.101")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config, calibration = load_contract(args.contract)
    stereo = json.loads((Path(args.contract).parent/"orin_6_6_7_100_20261014/luna_rectified_stereo.json").read_text())
    import vpi
    Gst.init(None)
    scale = config["image_size"][0]/calibration["image_size"][0]
    S = np.diag([scale, scale, 1.])
    maps = [vpi.WarpMap.fisheye_correction(
        vpi.WarpGrid(tuple(config["image_size"])),
        Kin=(S@np.array(calibration[f"{eye}_K"]))[:2],
        X=np.c_[np.array(calibration[f"rectify_R{i}"]), np.zeros(3)],
        Kout=(S@np.array(calibration[f"rectify_P{i}"])[:, :3])[:2],
        coeffs=np.array(calibration[f"{eye}_D"]), mapping=vpi.FisheyeMapping.EQUIDISTANT)
        for i, eye in enumerate(("left", "right"), 1)]
    latest = [None, None]
    serial = [0, 0]
    lock = threading.Lock()
    cameras = []
    encoders = [preview_pipeline(args.host, p, 6) for p in (5000, 5002)]
    for pipe in encoders:
        pipe.set_state(Gst.State.PLAYING)

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
            serial[eye] += 1
            latest[eye] = frame, serial[eye], time.monotonic_ns(), int(buffer.pts)
        return Gst.FlowReturn.OK

    def start(settings):
        nonlocal cameras
        latest[:] = [None, None]
        cameras = [capture_pipeline(i, settings) for i in range(2)]
        for eye, pipe in enumerate(cameras):
            pipe.get_by_name("sink").connect("new-sample", receive, eye)
            if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Camera tuning pipeline failed")

    def stop():
        for pipe in cameras:
            pipe.set_state(Gst.State.NULL)

    def collect(name, settings, settle=1.5, count=60):
        readback = ([{k: pipe.get_by_name(f"argus_{i}").get_property(k)
                      for k in controls(settings["eyes"][i])} for i, pipe in enumerate(cameras)]
                    if settings else None)
        time.sleep(settle)
        for i, pipe in enumerate(cameras):
            pipe.get_by_name(f"argus_{i}").set_property("aelock", True)
            pipe.get_by_name(f"argus_{i}").set_property("awblock", True)
        time.sleep(.5)
        used = [0, 0]
        previous = np.array(args.seed_centers).reshape(2, 2).tolist()
        records, attempted, preview_at = [], 0, 0.
        seed_metrics = None
        deadline = time.monotonic()+12
        while attempted < count and time.monotonic() < deadline:
            for pipe in cameras:
                message = pipe.get_bus().pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
                if message:
                    raise RuntimeError(str(message.parse_error()) if message.type == Gst.MessageType.ERROR else "EOS")
            with lock:
                pair = list(latest) if all(latest) and all(latest[i][1] != used[i] for i in range(2)) else None
            if pair is None:
                time.sleep(.001)
                continue
            used = [x[1] for x in pair]
            with vpi.Backend.CUDA:
                frames = [np.asarray(vpi.asimage(x[0]).remap(maps[i], interp=vpi.Interp.LINEAR).cpu()) for i, x in enumerate(pair)]
            attempted += 1
            if attempted == 1:
                seed_metrics = []
                for i, eye in enumerate(("left", "right")):
                    cv2.imwrite(str(args.output_dir/f"{name}_{eye}_full.png"), frames[i])
                    seed_detection = dict(center=np.array(args.seed_centers).reshape(2, 2)[i].tolist(),
                                          ellipse=dict(major_px=45., minor_px=45.))
                    seed_metrics.append(roi_metrics(frames[i], seed_detection))
            if time.monotonic()-preview_at >= 1/6:
                for i in range(2):
                    payload = cv2.resize(frames[i], (640, 496)).tobytes()
                    buf = Gst.Buffer.new_allocate(None, len(payload), None)
                    buf.fill(0, payload)
                    encoders[i].get_by_name("src").emit("push-buffer", buf)
                preview_at = time.monotonic()
            observation = stereo_measure(frames, config, previous, stereo=stereo)
            if not observation:
                continue
            observation["pair_dequeue_skew_ms"] = abs(pair[0][2]-pair[1][2])/1e6
            previous = [observation[e]["center"] for e in ("left", "right")]
            observation["roi"] = [roi_metrics(frames[i], observation[e]) for i, e in enumerate(("left", "right"))]
            sphere = stereo_sphere_center(observation["left"], observation["right"], stereo)
            observation["sphere_center_camera_m"] = sphere["center_left_camera_m"]
            observation["frame_number"] = used.copy()
            observation["timestamp_monotonic_ns"] = time.monotonic_ns()
            records.append(observation)
            if len(records) in (1, 45):
                for i, e in enumerate(("left", "right")):
                    x, y = np.rint(observation[e]["center"]).astype(int)
                    patch = frames[i][max(0, y-56):y+57, max(0, x-56):x+57]
                    cv2.imwrite(str(args.output_dir/f"{name}_{len(records)}_{e}.png"), patch)
        metrics = score_window(records, attempted)
        if records:
            points = np.array([r["sphere_center_camera_m"] for r in records])
            median = np.median(points, axis=0)
            metrics.update(sphere_center_median_camera_m=median.tolist(),
                           pair_skew_p95_ms=float(np.percentile([r["pair_dequeue_skew_ms"] for r in records], 95)),
                           timing_valid_fraction=float(np.mean([r["pair_dequeue_skew_ms"] <= config["max_pair_skew_ms"] for r in records])),
                           sphere_center_jitter_p95_mm=float(np.percentile(np.linalg.norm(points-median, axis=1)*1000, 95)))
        result = dict(name=name, profile=settings, property_readback=readback, metrics=metrics,
                      seed_roi_diagnostic=seed_metrics,
                      sensor_actual_exposure_metadata_available=False, records=records)
        with (args.output_dir/f"{name}.json").open("x") as out:
            json.dump(result, out, allow_nan=False)
        print(name, json.dumps(metrics), flush=True)
        return result

    results = []
    restarts = []
    try:
        start(None)
        time.sleep(5)
        for i, pipe in enumerate(cameras):
            source = pipe.get_by_name(f"argus_{i}")
            source.set_property("aelock", True)
            source.set_property("awblock", True)
        results.append(collect("baseline_auto_then_lock", None))
        coarse = []
        for wb in (2, 3, 4, 5, 6, 7, 8, 0):
            for exposure, gain in ((10, 1), (5, 2)):
                settings = profile(exposure, gain, wb=wb, isp=1)
                stop()
                time.sleep(1)
                start(settings)
                trial = collect(f"coarse_wb{wb}_e{exposure}_g{gain}", settings, settle=3)
                results.append(trial)
                if trial["metrics"]["eligible"]:
                    coarse.append(trial)
        if not coarse:
            raise RuntimeError("Fixed-profile probes have no eligible yellow detection; inspect saved full images")
        coarse_best = min(coarse, key=lambda r: r["metrics"]["score"])["profile"]["eyes"][0]
        for exposure in (2.5, 5, 10):
            for gain in (1, 2, 4):
                settings = profile(exposure, gain, wb=coarse_best["wbmode"], isp=1,
                                   digital=coarse_best["digital_gain"])
                if any(r["profile"] == settings for r in results):
                    continue
                stop()
                time.sleep(1)
                start(settings)
                results.append(collect(f"grid_e{exposure}_g{gain}", settings, settle=3))
        eligible = sorted([r for r in results if r["profile"] and r["metrics"]["eligible"]],
                          key=lambda r: r["metrics"]["score"])
        if not eligible:
            raise RuntimeError("No tested fixed profile satisfies visibility/brightness requirements")
        for n, seed in enumerate(eligible[:3]):
            eye = seed["profile"]["eyes"][0]
            for wb, isp in ((0, 0), (5, 0), (3, 1)):
                settings = profile(eye["exposure_ns"]/1e6, eye["analog_gain"], wb, isp, eye["digital_gain"])
                if any(r["profile"] == settings for r in results):
                    continue
                stop()
                time.sleep(1)
                start(settings)
                results.append(collect(f"refine{n}_wb{wb}_isp{isp}", settings, settle=3))
        eligible = sorted([r for r in results if r["profile"] and r["metrics"]["eligible"]],
                          key=lambda r: r["metrics"]["score"])
        winner = eligible[0]
        for iteration in range(3):
            stop()
            time.sleep(1)
            start(winner["profile"])
            restarts.append(collect(f"restart{iteration}", winner["profile"], settle=5, count=90))
        points = np.array([r["metrics"]["sphere_center_median_camera_m"] for r in restarts])
        repeatability = float(max(np.linalg.norm(a-b)*1000 for a in points for b in points))
        report = dict(schema="luna_fixed_camera_profile_selection_v1", chosen_profile=winner["profile"],
                      selected_trial=winner["name"], selected_metrics=winner["metrics"],
                      tested_profiles=[dict(name=r["name"], profile=r["profile"], metrics=r["metrics"]) for r in results],
                      restart_trials=[dict(name=r["name"], metrics=r["metrics"]) for r in restarts],
                      restart_sphere_center_max_difference_mm=repeatability,
                      calibration_sha256=config["calibration_sha256"],
                      qualification="Best among tested profiles at this fixed ball pose; confirm across selected workspace points.")
        with (args.output_dir/"selection.json").open("x") as out:
            json.dump(report, out, indent=2, allow_nan=False)
        print("SELECTION", json.dumps(report), flush=True)
    finally:
        stop()
        for pipe in encoders:
            pipe.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
