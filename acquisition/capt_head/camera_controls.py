"""Fixed Argus controls shared by tuning and the formal measurement service."""
import math

import cv2
import numpy as np


def validate_profile(profile):
    if profile.get("mode") not in ("fixed", "fixed_exposure_auto_wb") or len(profile.get("eyes", [])) != 2:
        raise ValueError("Fixed camera profile requires two eyes")
    for eye in profile["eyes"]:
        for key, low, high in (("exposure_ns", 40000, 10000000), ("analog_gain", 1, 4),
                               ("digital_gain", 1, 8), ("saturation", 0, 2),
                               ("tnr_strength", 0, 1), ("ee_strength", 0, 1)):
            value = eye[key]
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Invalid camera {key}: {value}")
        allowed_wb = (1,) if profile["mode"] == "fixed_exposure_auto_wb" else (0, 2, 3, 4, 5, 6, 7, 8)
        if eye["wbmode"] not in allowed_wb:
            raise ValueError("White balance must use an explicit fixed preset")
        if any(eye[k] not in (0, 1, 2) for k in ("tnr_mode", "ee_mode")):
            raise ValueError("Invalid ISP mode")
    return profile


def controls(eye):
    return {"exposuretimerange": f"{int(eye['exposure_ns'])} {int(eye['exposure_ns'])}",
            "gainrange": f"{eye['analog_gain']:g} {eye['analog_gain']:g}",
            "ispdigitalgainrange": f"{eye['digital_gain']:g} {eye['digital_gain']:g}",
            "wbmode": eye["wbmode"], "tnr-mode": eye["tnr_mode"],
            "ee-mode": eye["ee_mode"], "tnr-strength": eye["tnr_strength"],
            "ee-strength": eye["ee_strength"], "saturation": eye["saturation"]}


def verify_readback(profile, readback):
    """Check configured Argus properties, not unavailable sensor exposure metadata."""
    validate_profile(profile)
    if len(readback) != 2:
        raise ValueError("Both camera control readbacks required")
    for eye, actual in zip(profile['eyes'], readback):
        for key, expected in controls(eye).items():
            want = np.asarray(str(expected).split(), dtype=float)
            got = np.asarray(str(actual[key]).split() if isinstance(expected, str)
                             else [float(actual[key])], dtype=float)
            if want.shape != got.shape or not np.allclose(want, got, rtol=1e-5, atol=1e-6):
                raise ValueError(f"Camera control readback mismatch: {key}: {actual[key]} != {expected}")


def pipeline_properties(profile, sensor):
    if profile is None:
        return "aeantibanding=2 aelock=false awblock=false"
    validate_profile(profile)
    values = controls(profile["eyes"][sensor])
    return "aeantibanding=2 aelock=false awblock=false " + " ".join(
        f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in values.items())


def roi_metrics(image, detection):
    ellipse = detection["ellipse"]
    cx, cy = np.array(detection["center"])
    radius = .35*min(ellipse["major_px"], ellipse["minor_px"])
    x0, y0 = max(0, int(cx-radius)), max(0, int(cy-radius))
    x1, y1 = min(image.shape[1], int(cx+radius)+1), min(image.shape[0], int(cy+radius)+1)
    patch = image[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (xx-cx)**2+(yy-cy)**2 <= radius**2
    pixels = patch[mask]
    if not len(pixels):
        raise ValueError("Empty yellow ROI")
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[mask]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)[mask]
    return dict(mean_luma=float(gray.mean()), hsv_median=np.median(hsv, axis=0).tolist(),
                clipped_any_channel_fraction=float(np.mean(pixels.max(axis=1) >= 250)),
                clipped_all_channels_fraction=float(np.mean(pixels.min(axis=1) >= 250)),
                dark_fraction=float(np.mean(gray < 30)))


def score_window(records, attempted):
    if len(records) < 20:
        return dict(eligible=False, score=None, valid_frames=len(records), attempted=attempted)
    disparity = np.array([r["disparity_px"] for r in records])
    jitter = float(np.percentile(abs(disparity-np.median(disparity)), 95))
    rms = float(np.mean([r[e]["ellipse_rms_px"] for r in records for e in ("left", "right")]))
    coverage = min(r[e]["edge_coverage_deg"] for r in records for e in ("left", "right"))
    luma = np.array([[r["roi"][i]["mean_luma"] for i in range(2)] for r in records])
    clip = float(np.mean([r["roi"][i]["clipped_any_channel_fraction"] for r in records for i in range(2)]))
    valid_fraction = len(records)/max(attempted, 1)
    eligible = (valid_fraction >= .9 and clip <= .1 and
                np.all((luma.mean(0) >= 45) & (luma.mean(0) <= 220)))
    return dict(eligible=bool(eligible), valid_frames=len(records), attempted=attempted,
                valid_fraction=valid_fraction, disparity_jitter_p95_px=jitter,
                ellipse_rms_mean_px=rms, minimum_coverage_deg=coverage,
                mean_luma=luma.mean(0).tolist(), luma_std=luma.std(0).tolist(),
                clipped_any_channel_fraction=clip,
                score=float(jitter+.2*rms+.05*clip+.0005*np.abs(luma.mean(0)-140).mean()))
