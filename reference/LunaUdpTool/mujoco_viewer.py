"""
MuJoCo viewer with UDP server for Luna robot joint control.
Receives 20 float32 values (80 bytes) via UDP port 9999:
  [0:6]   = Leg_J1 ~ Leg_J6
  [6:13]  = Arm_R1 ~ Arm_R7
  [13:20] = Arm_L1 ~ Arm_L7
"""
import mujoco
import mujoco.viewer
import socket
import threading
import numpy as np
import struct
import time
import os

# Joint names matching the order in the data packet
JOINT_NAMES = [
    "Gento Luna_Leg_J1_Joint",
    "Gento Luna_Leg_J2_Joint",
    "Gento Luna_Leg_J3_Joint",
    "Gento Luna_Leg_J4_Joint",
    "Gento Luna_Leg_J5_Joint",
    "Gento Luna_Leg_J6_Joint",
    "Arm_R1_Joint",
    "Arm_R2_Joint",
    "Arm_R3_Joint",
    "Arm_R4_Joint",
    "Arm_R5_Joint",
    "Arm_R6_Joint",
    "Arm_R7_Joint",
    "Arm_L1_Joint",
    "Arm_L2_Joint",
    "Arm_L3_Joint",
    "Arm_L4_Joint",
    "Arm_L5_Joint",
    "Arm_L6_Joint",
    "Arm_L7_Joint",
]

NUM_JOINTS = len(JOINT_NAMES)
UDP_PORT = 9999


def main():
    xml_path = os.path.join(os.path.dirname(__file__),
                            "model", "gento_luna", "gento_luna_contact_filtered.xml")
    if not os.path.exists(xml_path):
        print(f"ERROR: Model file not found: {xml_path}")
        return

    print(f"Loading model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Map joint names to qpos addresses
    qpos_addrs = []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            print(f"ERROR: Joint not found: {name}")
            return
        qpos_addrs.append(model.jnt_qposadr[jid])

    # Thread-safe storage for latest received angles
    latest_angles = np.zeros(NUM_JOINTS, dtype=np.float32)
    lock = threading.Lock()
    running = [True]

    def udp_receiver():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", UDP_PORT))
        sock.settimeout(0.5)
        while running[0]:
            try:
                data_bytes, addr = sock.recvfrom(4096)
                if len(data_bytes) >= NUM_JOINTS * 4:
                    angles = struct.unpack(f"{NUM_JOINTS}f", data_bytes[:NUM_JOINTS * 4])
                    with lock:
                        latest_angles[:] = angles
            except socket.timeout:
                continue
            except Exception as e:
                if running[0]:
                    print(f"UDP error: {e}")
        sock.close()

    thread = threading.Thread(target=udp_receiver, daemon=True)
    thread.start()

    print(f"UDP server listening on 0.0.0.0:{UDP_PORT}")
    print(f"Controlling {NUM_JOINTS} joints: Leg(6) + RightArm(7) + LeftArm(7)")
    print("Run demo.cpp to send joint angles.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            with lock:
                angles = latest_angles.copy()

            for i, addr in enumerate(qpos_addrs):
                data.qpos[addr] = float(angles[i])

            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)

    running[0] = False
    print("Viewer closed.")


if __name__ == "__main__":
    main()