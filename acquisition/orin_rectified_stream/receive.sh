#!/usr/bin/env bash
set -euo pipefail
gst-launch-1.0 -e compositor name=mix sink_0::xpos=0 sink_1::xpos=640 \
  ! video/x-raw,width=1280,height=496 \
  ! videoconvert ! xvimagesink sync=false force-aspect-ratio=true \
  udpsrc port=5000 caps='application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000' \
  ! rtph264depay ! h264parse ! openh264dec ! videoconvert ! videoscale \
  ! video/x-raw,width=640,height=496 ! queue leaky=downstream ! mix.sink_0 \
  udpsrc port=5002 caps='application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000' \
  ! rtph264depay ! h264parse ! openh264dec ! videoconvert ! videoscale \
  ! video/x-raw,width=640,height=496 ! queue leaky=downstream ! mix.sink_1
