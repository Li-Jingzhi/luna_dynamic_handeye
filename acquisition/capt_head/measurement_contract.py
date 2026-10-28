"""Single camera and yellow measurement contract shared by Orin and PC."""
import hashlib
import json
from pathlib import Path
import math
from .camera_controls import validate_profile


DEFAULT_CONTRACT = Path(__file__).resolve().parents[1] / "configs/yellow_measurement_v1.json"


def load_contract(path=DEFAULT_CONTRACT):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "luna_yellow_contract_v1":
        raise ValueError("Unsupported yellow measurement contract")
    calibration_path = (path.parent / config["calibration"]).resolve()
    calibration_bytes = calibration_path.read_bytes()
    calibration = json.loads(calibration_bytes)
    if (calibration.get("schema") != "luna_raw_fisheye_stereo_v1" or
            tuple(calibration["image_size"]) != (2560, 1984) or
            config["image_size"] != [1280, 992] or
            config["preview_size"] != [640, 496] or
            config["pixel_domain"] != "ideal_pinhole" or
            config["antibanding_hz"] != 50):
        raise ValueError("Camera contract disagrees with current SH5 calibration")
    config["calibration_sha256"] = hashlib.sha256(calibration_bytes).hexdigest()
    config["contract_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if config.get("camera_controls") is not None:
        validate_profile(config["camera_controls"])
    return config, calibration


def valid_measurement(record, config):
    if (record.get("schema") != "luna_yellow_measurement_v1" or
            record.get("quality") != "PASS" or
            record.get("image_size") != config["image_size"] or
            record.get("pixel_domain") != config["pixel_domain"] or
            record.get("sensor_ids") != config["sensor_ids"] or
            record.get("measurement_backend") != config["measurement_backend"] or
            record.get("source_calibration_sha256") != config["calibration_sha256"] or
            record.get("contract_sha256") != config["contract_sha256"] or
            record.get("exposure_locked") is not True or
            record.get("awb_locked") is not True or
            record.get("brightness_stable") is not True or
            record.get("antibanding_hz") != config["antibanding_hz"]):
        raise ValueError("Measurement does not satisfy the current Orin camera contract")
    try:
        fps = float(record["measurement_fps"])
        skew = float(record["pair_capture_skew_ms"])
        dv = float(record["epipolar_dv_px"])
        disparity = float(record["disparity_px"])
        raw_centers = [record[eye]["center"] for eye in ("left", "right")]
        uv = record["corrected_stereo_pixels"]
        if len(uv) != 4:
            raise ValueError("Four corrected pixels required")
        centers = [uv[:2], uv[2:]]
        source_skew = float(record["pair_source_clock_skew_ms"])
        if (record.get("pairing_policy") != "shared_pipeline_source_and_arrival_10ms" or
                record.get("epipolar_check_domain") != "sphere_corrected_projection" or
                not math.isfinite(source_skew) or not 0 <= source_skew <= config["max_pair_skew_ms"]):
            raise ValueError("Shared-clock source pairing failed")
        values = [fps, skew, dv, disparity, *centers[0], *centers[1], *raw_centers[0], *raw_centers[1]]
        for eye in ("left", "right"):
            item = record[eye]
            ellipse = item["ellipse"]
            values.extend((item["ellipse_rms_px"], item["edge_coverage_deg"],
                           ellipse["major_px"], ellipse["minor_px"]))
            if (item["edge_count"] < config["min_edge_count"] or
                    item["ellipse_rms_px"] > config["max_ellipse_rms_px"] or
                    item["edge_coverage_deg"] < config["min_edge_coverage_deg"] or
                    ellipse["minor_px"] <= 0 or
                    ellipse["major_px"]/ellipse["minor_px"] > config["max_ellipse_axis_ratio"] or
                    abs(ellipse["cx"]-item["center"][0]) > .01 or
                    abs(ellipse["cy"]-item["center"][1]) > .01):
                raise ValueError("Subpixel ellipse quality outside contract")
        if (not all(math.isfinite(float(value)) for value in values) or
                fps < 0 or skew < 0 or skew > config["max_pair_skew_ms"] or
                abs(dv) > config["max_epipolar_dv_px"] or
                not config["disparity_range_px"][0] <= disparity <= config["disparity_range_px"][1] or
                abs(disparity-(centers[0][0]-centers[1][0])) > .01 or
                abs(dv-(centers[0][1]-centers[1][1])) > .01 or
                any(not 0 <= center[0] < config["image_size"][0] or
                    not 0 <= center[1] < config["image_size"][1] for center in centers)):
            raise ValueError("Measurement geometry or timing outside contract")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Incomplete Orin measurement") from exc
    return record
