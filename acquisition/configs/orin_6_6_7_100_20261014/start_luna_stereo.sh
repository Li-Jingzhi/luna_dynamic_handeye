#!/bin/bash
source /opt/ros/humble/setup.bash
source /opt/kernelmind/apex/install/setup.bash
source /home/nvidia/apex_gento_devel/install/setup.bash
export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ORIN_IFACE="${LUNA_ORIN_IFACE:-eth0}"
LOCAL_PEER="${LUNA_LOCAL_IP:-6.6.7.101}"
CAMERA_CONFIG="${LUNA_REMOTE_CAMERA_CONFIG:-/home/nvidia/luna_stereo_undistorted.yaml}"
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"$ORIN_IFACE\"/></Interfaces></General><Discovery><Peers><Peer address=\"$LOCAL_PEER\"/></Peers><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>256</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>"
exec /usr/bin/python3 /opt/ros/humble/bin/ros2 launch gmsl_quadcam quad_csi_quickview.launch.py "params_file:=$CAMERA_CONFIG"
