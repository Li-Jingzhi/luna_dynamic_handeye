"""Offline heldout evaluation and fixed-ball, multihead invariance metrics."""
from collections import Counter, defaultdict
import numpy as np


def evaluate(mapping, rows, *, heldout_only=True):
    errors, records, rejected = [], [], Counter()
    by_point = defaultdict(list)
    for r in rows:
        if heldout_only and r.get("split") not in ("validation", "test"):
            rejected["not_heldout"] += 1
            continue
        try:
            if r.get("calibration_sample_valid") is not True:
                raise ValueError("not stationary calibration")
            ans = mapping.query(r["stereo_pixels"], r["head_angles_rad"],
                                sensor_ids=r["sensor_ids"], image_size=r["image_size"],
                                pixel_domain=r["pixel_domain"])
            X = np.asarray(ans["xyz_m"],float)
            truth = np.asarray(r["xyz_base_m"],float)
            delta_mm = (X-truth)*1000
            errors.append(float(np.linalg.norm(delta_mm)))
            records.append(delta_mm)
            if r.get("point_id"):
                by_point[str(r["point_id"])].append((X,truth,np.asarray(r["head_angles_rad"],float)))
        except (ValueError, KeyError, TypeError) as exc:
            rejected[str(exc)] += 1
    report = dict(total_rows=len(rows), accepted=len(errors), rejected=dict(rejected),
                  heldout_only=heldout_only,
                  reference_labels="recorded robot FK/MDH; not independent metrology ground truth",
                  independently_verified=False)
    if errors:
        e=np.asarray(errors)
        d=np.asarray(records)
        report["absolute_vs_recorded_FK"] = dict(
            xyz_rms_mm=float(np.sqrt(np.mean(e*e))),
            xyz_p95_mm=float(np.percentile(e,95)), xyz_mean_mm=float(e.mean()),
            axis_bias_mm=d.mean(0).tolist(), axis_std_mm=d.std(0).tolist())
    report["fixed_point_invariance"]={}
    for name, samples in by_point.items():
        points, labels, angles=map(np.asarray,zip(*samples))
        nhead=len(np.unique(np.round(np.degrees(angles),1),axis=0))
        if nhead < 3: continue
        movement=np.max(np.linalg.norm(labels-np.median(labels,axis=0),axis=1))
        if movement > .001:
            report["fixed_point_invariance"][name]=dict(valid=False, reason="Reference marker moved >1 mm")
            continue
        residual=(points-points.mean(0))*1000
        dist=np.linalg.norm(residual,axis=1)
        report["fixed_point_invariance"][name]=dict(
            valid=True, head_pose_count=nhead, samples=len(points),
            drift_rms_mm=float(np.sqrt(np.mean(dist*dist))),
            drift_p95_mm=float(np.percentile(dist,95)), axis_std_mm=residual.std(0).tolist())
    target = 1.0
    measured = report.get("absolute_vs_recorded_FK", {}).get("xyz_p95_mm")
    report["acceptance_target"] = dict(
        xyz_p95_mm_lt=target, passed=bool(measured is not None and measured < target),
        note="Must pass on heldout poses; FK agreement is not independent metrology truth")
    return report
