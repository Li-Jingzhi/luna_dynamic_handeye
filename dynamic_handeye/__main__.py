"""Run with python3 -m dynamic_handeye; defaults never access hardware."""
import argparse
import json
from pathlib import Path
from .geometry import Assets, ROOT, digest
from .dataset import load_capture, write_json
from .calibration import evaluate, predict
from .runtime import predict_stopped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('inspect', help='Verify source hashes, Luna assets and head-chain definitions')
    imp = sub.add_parser('import', help='Validate original stationary capture JSONL')
    imp.add_argument('--capture', nargs='+', required=True); imp.add_argument('--output', required=True)
    fit = sub.add_parser('fit', help='Fit and validate a diagnostic model, never auto-approve')
    fit.add_argument('--dataset', required=True); fit.add_argument('--output', required=True)
    fit.add_argument('--chain', choices=['controller_mdh'], default='controller_mdh')
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
                      authoritative_chain='controller_mdh', mjcf_role='structural_reference_only',
                      camera_mount_in_mjcf=False, hardware_access=False)
    elif args.command == 'import':
        dataset = load_capture(args.capture, assets); write_json(args.output, dataset)
        output = dict(output=args.output, valid_rows=len(dataset['rows']), rejected_rows=sum(dataset['rejected'].values()))
    elif args.command == 'fit':
        dataset = json.loads(Path(args.dataset).read_text())
        if dataset.get('schema') != 'luna_dynamic_dataset_v1' or dataset.get('assets') != assets.hashes:
            raise ValueError('Dataset uses different robot/calibration assets')
        out = Path(args.output)
        if out.exists():
            raise FileExistsError(out)
        model, report = evaluate(dataset, assets, chain=args.chain)
        out.mkdir(parents=True)
        write_json(out/'model.json', model); write_json(out/'validation.json', report)
        output = dict(output=str(out), approved=False, fit_metrics=model['fit_metrics'],
                      trajectory_holdout=report['trajectory_holdout'].get('metrics', report['trajectory_holdout']),
                      head_coverage_qualified=model['head_coverage_qualified'])
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
