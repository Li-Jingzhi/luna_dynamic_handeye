"""Read completed fixed-head capture records for the preview overlay."""
import json
from pathlib import Path


def capture_indices(count, stride=1):
    """Select measurement stops while preserving original indices and both endpoints."""
    if type(count) is not int or count < 1 or type(stride) is not int or stride < 1:
        raise ValueError('Waypoint count and capture stride must be positive integers')
    return sorted(set(range(0, count, stride)) | {count - 1})


def read_progress(filename):
    path = Path(filename)
    if not path.exists():
        return 'CAPTURE: waiting for first saved point'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return 'CAPTURE: waiting for first saved point'
    trajectory = json.loads(Path(rows[0]['taught_path']).read_text())
    stride = rows[0].get('capture_stride',1)
    selected = capture_indices(len(trajectory['waypoints']),stride)
    total = len(selected)
    indices = [int(r['waypoint_index']) for r in rows]
    if (len(set(indices)) != len(indices) or any(i not in selected for i in indices)
            or any(r.get('capture_stride',1) != stride for r in rows)):
        raise ValueError('Invalid or repeated capture waypoint')
    valid = sum(r.get('capture_valid') is True for r in rows)
    state_path=path.parent/'status.json'
    label='CAPTURE'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if state.get('phase')=='stopped':
            reason=state.get('reason','unknown')
            label='STOPPED: valid ratio <80%' if reason.startswith('Less than 80%') else 'STOPPED: '+reason[:65]
        elif state.get('phase')=='complete':label='ALL HEADS COMPLETE'
        elif state.get('phase')=='paused':label=f'PAUSED after head {state.get("completed_rounds","?")}/9'
    return (f'{label} saved {len(rows)}/{total} ({100*len(rows)/total:.1f}%) | '
            f'valid {valid} | rejected {len(rows)-valid} | last waypoint {indices[-1]+1}')
