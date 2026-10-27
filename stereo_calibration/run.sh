#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${LUNA_CALIB_PYTHON:-$ROOT/../.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON=python3; fi
case "${1:-help}" in
  calibrate|calibrate_pinhole|start_capture|capture_realtime|rectify_preview|evaluate)
    ENTRY="$1"; shift
    exec "$PYTHON" "$ROOT/$ENTRY.py" "$@" ;;
  *) echo "Usage: bash $0 {calibrate|calibrate_pinhole|start_capture|capture_realtime|rectify_preview|evaluate} [options]"
     exit 2 ;;
esac
