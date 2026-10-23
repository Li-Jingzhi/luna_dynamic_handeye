"""Current 1280x992 stereo four-colour orientation for visual 9D calibration.

Yellow position stays on the validated formal yellow pipeline.  Red/blue/green are
used only to recover a rigid orientation basis.  No legacy 28 mm / 94 mm marker
geometry is assumed: the pose uses colour directions and explicit quality gates.
"""
from __future__ import annotations

import itertools

import cv2
import numpy as np

from capt_head.subpixel_ball import _fit_ball
from interactive_ball_calibration.geometry import stereo_sphere_center, triangulate


COLORS = ("red", "blue", "green")


def _mask(frame, ranges):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    result = np.zeros(frame.shape[:2], np.uint8)
    for low, high in ranges:
        result |= cv2.inRange(hsv, np.asarray(low, np.uint8), np.asarray(high, np.uint8))
    return cv2.morphologyEx(result, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))



def _colour_diagnostics(frame, ranges):
    mask = _mask(frame, ranges)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    items = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        x, y, w, h = cv2.boundingRect(contour)
        items.append({"area_px": float(cv2.contourArea(contour)),
                      "bbox": [int(x), int(y), int(w), int(h)]})
    return {"mask_pixels": int(np.count_nonzero(mask)), "largest": items}


def _hue_peaks(frame, top=6):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    select = (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 45)
    hue = hsv[:, :, 0][select]
    if hue.size == 0:
        return []
    hist = np.bincount(hue, minlength=180)
    order = np.argsort(hist)[::-1]
    chosen = []
    for index in order:
        if hist[index] <= 0:
            break
        if any(min(abs(int(index)-old), 180-abs(int(index)-old)) <= 3
               for old, _ in chosen):
            continue
        chosen.append((int(index), int(hist[index])))
        if len(chosen) >= top:
            break
    return [{"h": hue_value, "pixels": count} for hue_value, count in chosen]


def color_candidates(frame, ranges, detector_config, pose_config, anchor, limit=5):
    """Return coloured subpixel ellipse candidates near the formal yellow origin."""
    if tuple(frame.shape[:2][::-1]) != tuple(detector_config["image_size"]):
        raise ValueError("Four-ball orientation requires the formal measurement image")
    anchor = np.asarray(anchor, float)
    radius = int(pose_config["marker_search_radius_px"])
    x, y = np.rint(anchor).astype(int)
    x0, y0 = max(0, x-radius), max(0, y-radius)
    x1, y1 = min(frame.shape[1], x+radius), min(frame.shape[0], y+radius)
    image = frame[y0:y1, x0:x1]
    mask = _mask(image, ranges)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fit_config = dict(detector_config)
    fit_config.update(
        candidate_area_px=pose_config["color_candidate_area_px"],
        min_edge_count=pose_config["color_min_edge_count"],
        min_edge_coverage_deg=pose_config["color_min_edge_coverage_deg"],
        max_ellipse_rms_px=pose_config["color_max_ellipse_rms_px"],
        max_ellipse_axis_ratio=pose_config["color_max_ellipse_axis_ratio"],
    )
    found = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = cv2.contourArea(contour)
        bx, by, w, h = cv2.boundingRect(contour)
        if (not fit_config["candidate_area_px"][0] <= area <= fit_config["candidate_area_px"][1] or
                not .60 <= w/max(h, 1) <= 1.67):
            continue
        pad = 4
        cx, cy = max(0, bx-pad), max(0, by-pad)
        crop = image[cy:min(image.shape[0], by+h+pad),
                     cx:min(image.shape[1], bx+w+pad)]
        local = contour - np.array([[[cx, cy]]], np.int32)
        fit = _fit_ball(crop, local, fit_config)
        if fit is None:
            continue
        fit["center"] = (np.asarray(fit["center"]) + [x0+cx, y0+cy]).tolist()
        fit["ellipse"]["cx"], fit["ellipse"]["cy"] = fit["center"]
        fit["area_px"] = float(area)
        found.append(fit)
    return found[:limit]


def _stereo_pairs(left, right, stereo, pose_config, diameter_m=None):
    pairs = []
    for left_detection, right_detection in itertools.product(left, right):
        l = np.asarray(left_detection["center"], float)
        r = np.asarray(right_detection["center"], float)
        dv = float(l[1]-r[1])
        disparity = float(l[0]-r[0])
        if (abs(dv) > pose_config["max_epipolar_dv_px"] or
                not pose_config["disparity_range_px"][0] <= disparity <=
                    pose_config["disparity_range_px"][1]):
            continue
        try:
            if diameter_m is not None:
                recovered = stereo_sphere_center(
                    left_detection, right_detection, stereo, diameter_m)
                center = np.asarray(recovered["center_left_camera_m"], float)
                corrected = recovered["corrected_stereo_pixels"]
            else:
                center, reprojection = triangulate(np.r_[l, r], stereo)
                corrected = np.r_[l, r].tolist()
                recovered = {"stereo_reprojection_error_px": reprojection}
        except ValueError:
            continue
        score = (abs(dv) +
                 float(left_detection["ellipse_rms_px"]) +
                 float(right_detection["ellipse_rms_px"]) +
                 2.*float(recovered["stereo_reprojection_error_px"]))
        pairs.append({
            "center_m": center,
            "left": left_detection,
            "right": right_detection,
            "corrected_stereo_pixels": corrected,
            "pair_score": float(score),
            "epipolar_dv_px": dv,
            "disparity_px": disparity,
        })
    return sorted(pairs, key=lambda item: item["pair_score"])[:pose_config["max_pairs_per_color"]]


def pose_from_centers(yellow, red, blue, green, pose_config):
    """Build a right-handed marker frame from colour directions."""
    yellow, red, blue, green = [np.asarray(value, float)
                                for value in (yellow, red, blue, green)]
    if any(value.shape != (3,) or not np.isfinite(value).all()
           for value in (yellow, red, blue, green)):
        raise ValueError("Expected four finite 3-D colour centres")
    vx, vy, vg = red-yellow, blue-yellow, green-yellow
    lengths = np.asarray([np.linalg.norm(vx), np.linalg.norm(vy), np.linalg.norm(vg)])
    if (np.any(lengths < pose_config["min_axis_length_m"]) or
            np.any(lengths > pose_config["max_axis_length_m"])):
        raise ValueError("Four-ball axis length outside configured range")
    ratio = float(lengths.max()/lengths.min())
    if ratio > pose_config["max_axis_length_ratio"]:
        raise ValueError("Four-ball axis lengths are inconsistent")
    xr, yr, gr = vx/lengths[0], vy/lengths[1], vg/lengths[2]
    raw_dot = np.array([np.dot(xr, yr), np.dot(xr, gr), np.dot(yr, gr)])
    if np.max(np.abs(raw_dot)) > pose_config["max_abs_axis_dot"]:
        raise ValueError("Four-ball colour directions are not sufficiently orthogonal")
    x = xr
    y = yr-x*np.dot(x, yr)
    ny = np.linalg.norm(y)
    if ny < 1e-6:
        raise ValueError("Red/blue directions are degenerate")
    y /= ny
    z = np.cross(x, y)
    z /= np.linalg.norm(z)
    green_alignment = float(np.dot(z, gr))
    if green_alignment < pose_config["min_green_alignment_cos"]:
        raise ValueError("Green direction disagrees with red-cross-blue handedness")
    # Re-orthogonalize y after z selection.
    y = np.cross(z, x)
    R = np.column_stack((x, y, z))
    if np.linalg.det(R) < .999:
        raise ValueError("Recovered colour frame is not right-handed")
    score = (float(np.sum(np.abs(raw_dot))) +
             2.*(ratio-1.) + 2.*(1.-green_alignment))
    return {
        "visual_xyz_camera_m": yellow.tolist(),
        "visual_rotation6d_camera": np.r_[R[:, 0], R[:, 1]].tolist(),
        "rotation_matrix_camera": R.tolist(),
        "axis_lengths_m": lengths.tolist(),
        "raw_axis_dot": raw_dot.tolist(),
        "green_alignment_cos": green_alignment,
        "geometry_score": score,
    }


def estimate_pose9(frames, yellow_observation, stereo, detector_config, pose_config):
    """Estimate visual XYZ+Rotation6D from current rectified stereo frames."""
    if len(frames) != 2:
        raise ValueError("Expected left/right measurement frames")
    if not yellow_observation:
        return {"valid": False, "reason": "yellow_missing"}
    try:
        yellow = stereo_sphere_center(
            yellow_observation["left"], yellow_observation["right"], stereo,
            float(pose_config["yellow_diameter_m"]))
        yellow_center = np.asarray(yellow["center_left_camera_m"], float)
    except (KeyError, TypeError, ValueError):
        return {"valid": False, "reason": "yellow_center_invalid"}

    candidates = {
        color: [color_candidates(frame, pose_config["hsv_ranges"][color],
                                 detector_config, pose_config,
                                 yellow_observation[eye]["center"])
                for eye, frame in zip(("left", "right"), frames)]
        for color in COLORS
    }
    pairs = {}
    for color in COLORS:
        diameter = pose_config.get("ball_diameter_m", {}).get(color)
        pairs[color] = _stereo_pairs(candidates[color][0], candidates[color][1],
                                     stereo, pose_config, diameter)
        if not pairs[color]:
            return {"valid": False, "reason": f"{color}_stereo_missing",
                    "candidate_counts": {key: [len(value[0]), len(value[1])]
                                         for key, value in candidates.items()},
                    "colour_diagnostics": {
                        key: [_colour_diagnostics(
                            frame[max(0, int(yellow_observation[eye]["center"][1])-pose_config["marker_search_radius_px"]):
                                  min(frame.shape[0], int(yellow_observation[eye]["center"][1])+pose_config["marker_search_radius_px"]),
                                  max(0, int(yellow_observation[eye]["center"][0])-pose_config["marker_search_radius_px"]):
                                  min(frame.shape[1], int(yellow_observation[eye]["center"][0])+pose_config["marker_search_radius_px"])],
                            pose_config["hsv_ranges"][key])
                              for eye, frame in zip(("left", "right"), frames)]
                        for key in COLORS},
                    "hue_peaks": [_hue_peaks(
                        frame[max(0, int(yellow_observation[eye]["center"][1])-pose_config["marker_search_radius_px"]):
                              min(frame.shape[0], int(yellow_observation[eye]["center"][1])+pose_config["marker_search_radius_px"]),
                              max(0, int(yellow_observation[eye]["center"][0])-pose_config["marker_search_radius_px"]):
                              min(frame.shape[1], int(yellow_observation[eye]["center"][0])+pose_config["marker_search_radius_px"])])
                        for eye, frame in zip(("left", "right"), frames)]}

    hypotheses = []
    for red, blue, green in itertools.product(pairs["red"], pairs["blue"], pairs["green"]):
        try:
            pose = pose_from_centers(yellow_center, red["center_m"], blue["center_m"],
                                     green["center_m"], pose_config)
        except ValueError:
            continue
        pair_score = red["pair_score"] + blue["pair_score"] + green["pair_score"]
        pose["score"] = float(pose["geometry_score"] + pose_config["pair_score_weight"]*pair_score)
        pose["balls"] = {
            "yellow": {"center_m": yellow_center.tolist(),
                       "stereo_pixels": yellow["corrected_stereo_pixels"]},
            "red": {"center_m": red["center_m"].tolist(),
                    "stereo_pixels": red["corrected_stereo_pixels"]},
            "blue": {"center_m": blue["center_m"].tolist(),
                     "stereo_pixels": blue["corrected_stereo_pixels"]},
            "green": {"center_m": green["center_m"].tolist(),
                      "stereo_pixels": green["corrected_stereo_pixels"]},
        }
        hypotheses.append(pose)

    if not hypotheses:
        return {"valid": False, "reason": "four_ball_geometry_rejected"}
    hypotheses.sort(key=lambda item: item["score"])
    if (len(hypotheses) > 1 and
            hypotheses[1]["score"]-hypotheses[0]["score"] <
            pose_config["ambiguity_score_margin"]):
        return {"valid": False, "reason": "four_ball_ambiguous",
                "best_scores": [hypotheses[0]["score"], hypotheses[1]["score"]]}
    result = hypotheses[0]
    result.update(valid=True, reason="four_ball_pose_valid",
                  method="stereo_colour_direction_pose_v1")
    return result
