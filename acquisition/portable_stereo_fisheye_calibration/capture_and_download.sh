#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ORIN_IP="${ORIN_IP:-6.6.7.100}"
ORIN_USER="${ORIN_USER:-nvidia}"
LOCAL_IFACE="${LOCAL_IFACE:-eno0}"
COLS="${BOARD_COLS:-11}"; ROWS="${BOARD_ROWS:-8}"; SQUARE="${SQUARE_MM:-35}"
SECONDS="${CAPTURE_SECONDS:-1800}"; TAG="$(date +%Y%m%d_%H%M%S)"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-$ROOT/data/raw_chessboard_$TAG}"
REMOTE_OUTPUT="/home/$ORIN_USER/raw_chessboard_$TAG"
SSH=(ssh -Y -o BindInterface="$LOCAL_IFACE" "$ORIN_USER@$ORIN_IP")
SCP=(scp -o BindInterface="$LOCAL_IFACE")
"${SCP[@]}" "$ROOT/capture_orin.py" "$ORIN_USER@$ORIN_IP:/tmp/portable_capture_orin.py"
echo "Board: ${COLS}x${ROWS} INNER corners, ${SQUARE} mm. Press q in the Orin preview or Ctrl+C to finish."
"${SSH[@]}" "source /opt/ros/humble/setup.bash; source /opt/kernelmind/apex/install/setup.bash; source /home/nvidia/apex_gento_devel/install/setup.bash; python3 /tmp/portable_capture_orin.py --output '$REMOTE_OUTPUT' --corners '$COLS' '$ROWS' --square-mm '$SQUARE' --seconds '$SECONDS' --preview"
mkdir -p "$(dirname "$LOCAL_OUTPUT")"
"${SCP[@]}" -r "$ORIN_USER@$ORIN_IP:$REMOTE_OUTPUT" "$LOCAL_OUTPUT"
echo "Downloaded to $LOCAL_OUTPUT"
echo "Run: python3 calibrate.py --images '$LOCAL_OUTPUT' --output calibration.json --corners $COLS $ROWS --square-mm $SQUARE"
