"""Read completed fixed-head capture records for the preview overlay."""
import json
from pathlib import Path


def read_progress(filename):
    path = Path(filename)
    if not path.exists():
        return 'CAPTURE: waiting for first saved point'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return 'CAPTURE: waiting for first saved point'
    trajectory = json.loads(Path(rows[0]['taught_path']).read_text())
    total = len(trajectory['waypoints'])
    indices = [int(r['waypoint_index']) for r in rows]
    if len(set(indices)) != len(indices) or any(i < 0 or i >= total for i in indices):
        raise ValueError('Invalid or repeated capture waypoint')
    valid = sum(r.get('capture_valid') is True for r in rows)
    return (f'CAPTURE saved {len(rows)}/{total} ({100*len(rows)/total:.1f}%) | '
            f'valid {valid} | rejected {len(rows)-valid} | last waypoint {indices[-1]+1}')
