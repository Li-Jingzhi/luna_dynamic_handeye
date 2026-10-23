"""Strict ingestion of passive stationary Luna yellow-ball observations."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SPLITS = ("train", "validation", "test")


def load_rows(paths):
    rows = []
    for filename in paths:
        p = Path(filename)
        if not p.is_file():
            raise FileNotFoundError(p)
        for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"{p}:{line_no}: expected JSON object")
                obj["_source_file"] = str(p)
                rows.append(obj)
    if not rows:
        raise ValueError("No observations")
    return rows


def valid_rows(rows, *, strict=True):
    accepted = []
    rejected = []
    for i, r in enumerate(rows):
        try:
            if r.get("calibration_sample_valid") is not True:
                raise ValueError("Not a stationarity-approved calibration sample")
            if r.get("split") not in SPLITS:
                raise ValueError("Missing train/validation/test split")
            if r.get("frame_id") != "robot_base" or r.get("units") != "m":
                raise ValueError("Expected robot_base XYZ in metres")
            if r.get("pixel_domain") not in ("raw", "ideal_pinhole"):
                raise ValueError("Unrecognized pixel domain")
            if not r.get("group_id"):
                raise ValueError("Distinct stationarity group_id required")
            x = np.asarray(r["xyz_base_m"], float)
            pix = np.asarray(r["stereo_pixels"], float)
            angles = np.asarray(r["head_angles_rad"], float)
            ref = np.asarray(r["reference_head_angles_rad"], float)
            if x.shape != (3,) or pix.shape != (4,) or angles.shape != (2,) or ref.shape != (2,):
                raise ValueError("Invalid XYZ/pixel/head vector sizes")
            if not all(np.isfinite(z).all() for z in (x, pix, angles, ref)):
                raise ValueError("Nonfinite observation")
            size = r.get("image_size")
            if not isinstance(size, list) or len(size) != 2 or min(size) <= 0:
                raise ValueError("Missing image_size")
            w, h = size
            if np.any(pix[[0,2]] < 0) or np.any(pix[[0,2]] >= w) or np.any(pix[[1,3]] < 0) or np.any(pix[[1,3]] >= h):
                raise ValueError("Pixels outside image")
            if not isinstance(r.get("sensor_ids"), list) or len(r["sensor_ids"]) != 2:
                raise ValueError("Missing ordered stereo sensor IDs")
            accepted.append(r)
        except (KeyError, ValueError, TypeError) as exc:
            rejected.append(f"row {i}: {exc}")
    if strict and rejected:
        raise ValueError("Rejected sample(s): "+"; ".join(rejected[:8]))
    if not accepted:
        raise ValueError("No stationarity-approved data")
    contracts = {
        json.dumps((r["sensor_ids"], r["image_size"], r["pixel_domain"],
                    r["reference_head_angles_rad"], r.get("yellow_center_flange_m"),
                    r.get("yellow_offset_calibration_id")), sort_keys=True)
        for r in accepted
    }
    if len(contracts) != 1:
        raise ValueError("Mismatched camera, reference head, pixel domain or marker offset")
    group_splits = {}
    for r in accepted:
        group = str(r["group_id"])
        split = r["split"]
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"Calibration group leakage between splits: {group}")
        group_splits[group] = split
    return accepted, rejected


def camera_contract(rows, lens):
    ref = rows[0]
    if lens is not None:
        if lens.get("image_size") != ref["image_size"] or lens.get("sensor_ids") != ref["sensor_ids"]:
            raise ValueError("Lens belongs to different image size/sensor order")
    if ref["pixel_domain"] == "ideal_pinhole" and lens is not None:
        raise ValueError("Pixels are already undistorted")
    return {
        "sensor_ids": ref["sensor_ids"],
        "image_size": ref["image_size"],
        "input_pixel_domain": ref["pixel_domain"],
        "reference_head_angles_rad": ref["reference_head_angles_rad"],
        "yellow_offset_calibration_id": ref.get("yellow_offset_calibration_id"),
    }


def load_lens(path):
    if not path:
        return None
    lens = json.loads(Path(path).read_text(encoding="utf-8"))
    if lens.get("model") not in ("pinhole", "fisheye"):
        raise ValueError("Lens requires pinhole/fisheye projection model")
    return lens


def write_new(path, obj):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # exclusive creation, never clobber a real calibration artifact
    with target.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def prepare_multisession(rows, reference_head_deg):
    """Normalize batch-local *reference metadata* to global q0, never measured q.

    Original collection assigns each stationary batch its own 'reference head'
    from that batch. This field is a bookkeeping reference, not a 3D transform.
    All recorded base XYZ, stereo pixels, measured head angles, tags and raw
    source data are unchanged.
    """
    reference = np.radians(np.asarray(reference_head_deg,float))
    if reference.shape!=(2,) or not np.isfinite(reference).all():
        raise ValueError("Global reference yaw/pitch must be two finite degrees")
    prepared=[]
    for i,r in enumerate(rows):
        if r.get("calibration_sample_valid") is not True:
            continue
        measured=np.asarray(r["head_angles_rad"],float)
        local=np.asarray(r["reference_head_angles_rad"],float)
        if measured.shape!=(2,) or local.shape!=(2,) or not np.isfinite(measured).all() or not np.isfinite(local).all():
            raise ValueError(f"Invalid source head metadata at row {i}")
        if np.max(np.abs(measured-local))>np.radians(.5):
            raise ValueError(f"Source head group mismatches actual SDK angles at row {i}")
        s=dict(r)
        s["original_batch_reference_head_angles_rad"]=s["reference_head_angles_rad"]
        s["reference_head_angles_rad"]=reference.tolist()
        s["canonical_reference_provenance"]="metadata_only; measured head/pixels/base XYZ unchanged"
        prepared.append(s)
    if not prepared:
        raise ValueError("No valid stationary samples; dynamic continuous frames are excluded")
    valid_rows(prepared, strict=True)
    q0train=[r for r in prepared if r["split"]=="train" and
             np.max(np.abs(np.asarray(r["head_angles_rad"])-reference))<=np.radians(.25)]
    if len(q0train)<30:
        raise ValueError("Canonical reference head has fewer than 30 stationary train observations")
    return prepared


def write_jsonl_new(path, rows):
    target=Path(path)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n")
