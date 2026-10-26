"""Run with python3 -m dynamic_handeye; defaults never access hardware."""
import argparse
import json
from pathlib import Path
from .geometry import Assets, ROOT, digest, torso_flange, fixed_head_root, fixed_geometry_hashes
from .dataset import load_capture, write_json
from .calibration import evaluate, predict
from .runtime import predict_stopped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('inspect', help='Verify source hashes, Luna assets and head-chain definitions')
    imp = sub.add_parser('import', help='Validate original stationary capture JSONL')
    imp.add_argument('--left-ball-offset', type=float, nargs=3, metavar=('X','Y','Z'), help='Measured left flange-to-ball offset in metres')
    imp.add_argument('--capture', nargs='+', required=True); imp.add_argument('--output', required=True)
    fk = sub.add_parser('fk', help='Old V2.0 arm flange poses in S_MODEL, no hardware access')
    fk.add_argument('--left', type=float, nargs=7, required=True)
    fk.add_argument('--right', type=float, nargs=7, required=True)
    fit = sub.add_parser('fit', help='Fit six camera parameters with fixed nominal head geometry; never auto-approve')
    fit.add_argument('--dataset', required=True); fit.add_argument('--output', required=True)
    fit.add_argument('--chain', choices=['luna_v2_hybrid'], default='luna_v2_hybrid')
    query = sub.add_parser('predict', help='Stopped-head inference; diagnostic output requires explicit flag')
    query.add_argument('--model', required=True); query.add_argument('--pixels', type=float, nargs=4, required=True)
    query.add_argument('--head', type=float, nargs=2, required=True)
    query.add_argument('--head-speed', type=float, nargs=2, required=True)
    query.add_argument('--head-window-range', type=float, nargs=2, required=True)
    query.add_argument('--diagnostic', action='store_true')
    cap = sub.add_parser('capture', help='Check crossed plan; --run explicitly enables existing SDK motion')
    cap.add_argument('--plan', default=str(ROOT/'configs/capture_plan.json'))
    cap.add_argument('--path', required=True); cap.add_argument('--output', required=True)
    cap.add_argument('--sdk-root', default=str(ROOT.parent/'00040701'))
    cap.add_argument('--latest', default=str(ROOT/'acquisition/runtime/new_sh5_yellow/latest_pose.json'))
    cap.add_argument('--robot-ip', default='6.6.7.190'); cap.add_argument('--stop-after', type=int, default=12)
    cap.add_argument('--run', action='store_true')
    args = parser.parse_args()
    assets = Assets()
    if args.command == 'inspect':
        sources = json.loads((ROOT/'sources.json').read_text())['files']
        bad = [r['destination'] for r in sources if digest(ROOT/r['destination']) != r['sha256']]
        if bad:
            raise ValueError('Copied sources changed: '+str(bad))
        output = dict(robot='gento_luna', source_files_verified=len(sources), assets=assets.hashes,
                      sensor_ids=assets.stereo['sensor_ids'], image_size=assets.stereo['image_size'],
                      baseline_mm=abs(assets.stereo['right_from_left_T_m'][0])*1000,
                      arm_mdh_rows=assets.rows('R.A1', 7).tolist(), head_mdh_rows=assets.head_rows.tolist(),
                      authoritative_chain='luna_v2_hybrid', mjcf_role='nominal_head_center_and_arm_mounts',
                      head_mdh_role='legacy_provenance_only_not_used_by_hybrid',
                      camera_mount_in_mjcf=False, hardware_access=False, parameter_count=6,
                      fixed_head_root_T=fixed_head_root(assets).tolist(),
                      fixed_geometry_hashes=fixed_geometry_hashes(assets),
                      fixed_head_geometry_status='nominal_not_independently_measured')
    elif args.command == 'fk':
        output = dict(robot_version='luna_v2_0', frame_id='S_MODEL', length_unit='m',
                      left_flange_T=torso_flange(assets, args.left, 'left')[0].tolist(),
                      right_flange_T=torso_flange(assets, args.right, 'right')[0].tolist())
    elif args.command == 'import':
        dataset = load_capture(args.capture, assets, left_ball_offset=args.left_ball_offset); write_json(args.output, dataset)
        output = dict(output=args.output, valid_rows=len(dataset['rows']), rejected_rows=sum(dataset['rejected'].values()))
    elif args.command == 'fit':
        dataset = json.loads(Path(args.dataset).read_text())
        if dataset.get('schema') != 'luna_dynamic_dataset_v1' or dataset.get('assets') != assets.hashes:
            raise ValueError('Dataset uses different robot/calibration assets')
        if dataset.get('robot_version', 'luna_v2_0') != 'luna_v2_0':
            raise ValueError('Dataset belongs to another Luna version')
        out = Path(args.output)
        if out.exists():
            raise FileExistsError(out)
        model, report = evaluate(dataset, assets, chain=args.chain)
        out.mkdir(parents=True)
        write_json(out/'model.json', model); write_json(out/'validation.json', report)
        output = dict(output=str(out), approved=False, fit_metrics=model['fit_metrics'],
                      trajectory_holdout=report['trajectory_holdout'].get('metrics', report['trajectory_holdout']),
                      head_coverage_qualified=model['head_coverage_qualified'], parameter_count=model['parameter_count'])
    elif args.command == 'capture':
        from .capture import run_capture
        output = run_capture(args, assets)
    else:
        model = json.loads(Path(args.model).read_text())
        if model.get('assets') != assets.hashes or model.get('robot') != 'gento_luna':
            raise ValueError('Wrong robot or calibration assets')
        if args.diagnostic:
            import numpy as np
            camera = assets.triangulate(args.pixels)
            xyz = predict(model, assets, camera[None, :], np.array([args.head]))[0]
            output = dict(xyz_m=xyz.tolist(), frame_id='S_MODEL', mode='offline_diagnostic', approved=False)
        else:
            output = predict_stopped(model, assets, args.pixels, args.head,
                                     head_velocity_deg_s=args.head_speed, head_window_range_deg=args.head_window_range)
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
