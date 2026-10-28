"""HSV acquisition followed by original-image subpixel edge and robust ellipse fitting."""
import cv2
import numpy as np


def _edge_points(edge_image, contour, seed):
    points = contour.reshape(-1, 2).astype(np.float32)
    direction = points - np.asarray(seed, np.float32)
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1)
    gx = cv2.Sobel(edge_image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(edge_image, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    offsets = np.arange(-2., 2.01, .5, dtype=np.float32)
    samples = []
    for offset in offsets:
        xy = points + offset * direction
        samples.append(cv2.remap(gradient, xy[:, :1], xy[:, 1:2],
                                 cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[:, 0])
    samples = np.stack(samples, axis=1)
    peaks = np.argmax(samples[:, 1:-1], axis=1) + 1
    rows = np.arange(len(points))
    before, middle, after = (samples[rows, peaks-1], samples[rows, peaks],
                             samples[rows, peaks+1])
    curvature = before - 2*middle + after
    sub = np.divide(.25*(before-after), curvature,
                    out=np.zeros_like(curvature), where=np.abs(curvature) > 1e-5)
    sub = np.clip(sub, -.5, .5)
    refined = points + (offsets[peaks]+sub)[:, None]*direction
    return refined[middle >= max(float(np.percentile(middle, 20)), 5.)]


def _ellipse_residual(points, fit):
    (cx, cy), (a, b), angle = fit
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), np.sin(theta)],
                         [-np.sin(theta), np.cos(theta)]])
    local = (points-np.array([cx, cy])) @ rotation.T
    radial = np.sqrt((local[:, 0]/max(a/2, 1))**2 +
                     (local[:, 1]/max(b/2, 1))**2)
    return (radial-1) * (a+b)/4


def _fit_ball(image, contour, config):
    seed = cv2.moments(contour)
    if seed["m00"] <= 0:
        return None
    center = (seed["m10"]/seed["m00"], seed["m01"]/seed["m00"])
    chroma = image.max(axis=2)-image.min(axis=2)
    points = _edge_points(chroma, contour, center)
    if len(points) < config["min_edge_count"]:
        return None
    for _ in range(3):
        fit = cv2.fitEllipse(points.reshape(-1, 1, 2))
        residual = np.abs(_ellipse_residual(points, fit))
        keep = residual <= max(.8, 2.5*np.median(residual))
        if keep.sum() < config["min_edge_count"] or keep.all():
            break
        points = points[keep]
    fit = cv2.fitEllipse(points.reshape(-1, 1, 2))
    residual = _ellipse_residual(points, fit)
    angles = np.arctan2(points[:, 1]-fit[0][1], points[:, 0]-fit[0][0])
    coverage = float(np.count_nonzero(np.histogram(angles, bins=36, range=(-np.pi, np.pi))[0]) * 10)
    major, minor = sorted(fit[1], reverse=True)
    major_angle = (fit[2]+90) % 180 if fit[1][0] < fit[1][1] else fit[2] % 180
    rms = float(np.sqrt(np.mean(residual**2)))
    if (rms > config["max_ellipse_rms_px"] or
            major/max(minor, 1) > config["max_ellipse_axis_ratio"] or
            coverage < config["min_edge_coverage_deg"]):
        return None
    return {"center": [float(fit[0][0]), float(fit[0][1])],
            "ellipse": {"cx": float(fit[0][0]), "cy": float(fit[0][1]),
                        "major_px": float(major), "minor_px": float(minor),
                        "angle_deg": float(major_angle)},
            "edge_count": len(points), "ellipse_rms_px": rms,
            "edge_coverage_deg": coverage}


def candidates(frame, config, previous=None):
    """Return qualified ball candidates in measurement pixels; preview is never input."""
    if tuple(frame.shape[:2][::-1]) != tuple(config["image_size"]):
        raise ValueError("Yellow detector requires full measurement-resolution image")
    origin = np.zeros(2, int)
    image = frame
    if previous is not None:
        x, y = np.rint(previous).astype(int)
        radius = config["tracking_radius_px"]
        x0, y0 = max(0, x-radius), max(0, y-radius)
        image = frame[y0:min(frame.shape[0], y+radius),
                      x0:min(frame.shape[1], x+radius)]
        origin = np.array([x0, y0])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, tuple(config["hsv_lower"]), tuple(config["hsv_upper"]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    result = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if (not config["candidate_area_px"][0] <= area <= config["candidate_area_px"][1] or
                not .72 <= w/max(h, 1) <= 1.39):
            continue
        pad = 4
        crop_x, crop_y = max(0, x-pad), max(0, y-pad)
        crop = image[crop_y:min(image.shape[0], y+h+pad),
                     crop_x:min(image.shape[1], x+w+pad)]
        local_contour = contour - np.array([[[crop_x, crop_y]]], np.int32)
        found = _fit_ball(crop, local_contour, config)
        if found is None:
            continue
        found["center"] = (np.asarray(found["center"])+origin+[crop_x, crop_y]).tolist()
        found["ellipse"]["cx"], found["ellipse"]["cy"] = found["center"]
        if previous is not None and np.linalg.norm(np.asarray(found["center"])-previous) > config["max_center_step_px"]:
            continue
        result.append(found)
    return result


def stereo_measure(frames, config, previous=None, stereo=None):
    from interactive_ball_calibration.geometry import stereo_sphere_center
    if stereo is None:
        raise ValueError("Sphere-corrected stereo model required")
    previous = previous or [None, None]
    options = [candidates(frame, config, old) for frame, old in zip(frames, previous)]
    pairs = []
    for left in options[0]:
        for right in options[1]:
            disparity = left["center"][0]-right["center"][0]
            dv = left["center"][1]-right["center"][1]
            if config["disparity_range_px"][0] <= disparity <= config["disparity_range_px"][1]:
                score = (left["ellipse_rms_px"]+right["ellipse_rms_px"]+abs(dv) if
                         any(old is None for old in previous) else
                         sum(np.linalg.norm(np.asarray(candidate["center"])-old)
                             for candidate, old in zip((left, right), previous)))
                pairs.append((score, left, right, disparity, dv))
    if not pairs:
        return None
    _, left, right, disparity, dv = min(pairs, key=lambda item: item[0])
    try:
        sphere = stereo_sphere_center(left, right, stereo)
    except ValueError:
        return None
    uv = sphere["corrected_stereo_pixels"]
    corrected_dv = uv[1]-uv[3]
    return {"left": left, "right": right, "raw_stereo_pixels": left["center"]+right["center"],
            "raw_epipolar_dv_px": float(dv), "raw_disparity_px": float(disparity),
            "corrected_stereo_pixels": uv, "center_left_camera_m": sphere["center_left_camera_m"],
            "disparity_px": float(uv[0]-uv[2]), "epipolar_dv_px": float(corrected_dv),
            "quality": "PASS" if abs(corrected_dv) <= config["max_epipolar_dv_px"] else "REJECT"}
