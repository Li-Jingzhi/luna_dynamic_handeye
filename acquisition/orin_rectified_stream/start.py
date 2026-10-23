#!/usr/bin/env python3
"""Validate locally or deploy the unified Orin measurement and preview service."""
import argparse
import os
import socket
import shlex
import subprocess
import time
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parents[1]
SENDER = ROOT / "orin_rectified_stream/sender.py"
FULL_RECEIVER = ROOT / "orin_rectified_stream/viewer.py"
CALIBRATION = ROOT / "configs/orin_6_6_7_100_20261014/luna_raw_fisheye_stereo.json"
CONTRACT = ROOT / "configs/yellow_measurement_v1.json"
DETECTOR = ROOT / "capt_head/subpixel_ball.py"
FOUR_BALL = ROOT / "capt_head/four_ball_pose.py"
FOUR_BALL_CONFIG = ROOT / "configs/four_ball_orientation_v1.json"
RECTIFIED_STEREO = ROOT / "configs/orin_6_6_7_100_20261014/luna_rectified_stereo.json"
INTERACTIVE_GEOMETRY = ROOT / "interactive_ball_calibration/geometry.py"
CONTRACT_CODE = ROOT / "capt_head/measurement_contract.py"
CAMERA_CONTROLS_CODE = ROOT / "capt_head/camera_controls.py"
REMOTE_ROOT = "/home/nvidia/luna_measurement/head_2dof_capt"
REMOTE_SENDER = REMOTE_ROOT + "/orin_rectified_stream/sender.py"
REMOTE_CALIBRATION = REMOTE_ROOT + "/configs/orin_6_6_7_100_20261014/luna_raw_fisheye_stereo.json"
REMOTE_CONTRACT = REMOTE_ROOT + "/configs/yellow_measurement_v1.json"
REMOTE_FOUR_BALL_CONFIG = REMOTE_ROOT + "/configs/four_ball_orientation_v1.json"
REMOTE_RECTIFIED_STEREO = REMOTE_ROOT + "/configs/orin_6_6_7_100_20261014/luna_rectified_stereo.json"
REMOTE_LOG = "/home/nvidia/luna_rectified_sender.log"


def local_ipv4(interface):
    result = subprocess.run(["ip", "-4", "-o", "addr", "show", "dev", interface],
                            text=True, capture_output=True, check=True)
    fields = result.stdout.split()
    if len(fields) < 4:
        raise RuntimeError(f"本机网卡 {interface} 没有 IPv4 地址")
    return fields[3].split("/", 1)[0]


def remote(client, command, timeout=70):
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    code = stdout.channel.recv_exit_status()
    output = stdout.read().decode(errors="replace")
    error = stderr.read().decode(errors="replace")
    if code:
        raise RuntimeError((error or output or f"远程命令退出码 {code}").strip())
    return output


def connect(host, user, password, local_ip):
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.settimeout(8)
    connection.bind((local_ip, 0))
    connection.connect((host, 22))
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=8, sock=connection)
    return client, connection


def stop_all(client):
    remote(client, "pkill -INT -f '[o]rin_rectified_stream/sender.py' || true; "
                   "pkill -INT -f '[l]una_rectified_sender.py' || true; "
                   "pkill -INT -f '[r]os2 launch gmsl_quadcam quad_csi_quickview' || true")
    time.sleep(3)
    remote(client, "pkill -KILL -f '[o]rin_rectified_stream/sender.py' || true; "
                   "pkill -KILL -f '[l]una_rectified_sender.py' || true; "
                   "pkill -KILL -f '[r]os2 launch gmsl_quadcam quad_csi_quickview' || true; "
                   "pkill -KILL -f '[g]msl_pipeline_container' || true; "
                   "pkill -KILL -f '[w]ebrtc_signal_server.py' || true")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--evidence-dir", help="Orin directory for lossless measurement ROI evidence")
    parser.add_argument("--four-ball", action="store_true",
                        help="enable low-priority visual Rotation6D estimation on Orin")
    args = parser.parse_args()
    interface = os.getenv("LUNA_LOCAL_IFACE", "eno0")
    host = os.getenv("LUNA_ORIN_IP", "6.6.7.100")
    user = os.getenv("LUNA_ORIN_USER", "nvidia")
    password = os.getenv("LUNA_ORIN_PASSWORD")
    if args.check:
        check = ["python3", str(SENDER), "--contract", str(CONTRACT), "--check"]
        if args.four_ball:
            check += ["--four-ball-config", str(FOUR_BALL_CONFIG)]
        subprocess.run(check, check=True)
        subprocess.run(["python3", "-m", "py_compile", str(DETECTOR), str(FOUR_BALL),
                        str(INTERACTIVE_GEOMETRY), str(FULL_RECEIVER)], check=True)
        print("本机测量算法、统一参数和接收器检查通过；未连接 Orin")
        return
    local_ip = local_ipv4(interface)

    client, connection = connect(host, user, password, local_ip)
    try:
        if args.stop:
            stop_all(client)
            print("Orin非ROS校正视频已停止")
            return
        with client.open_sftp() as sftp:
            remote(client, f"mkdir -p {REMOTE_ROOT}/orin_rectified_stream {REMOTE_ROOT}/capt_head "
                           f"{REMOTE_ROOT}/interactive_ball_calibration "
                           f"{REMOTE_ROOT}/configs/orin_6_6_7_100_20261014")
            sftp.put(str(SENDER), REMOTE_SENDER)
            sftp.put(str(DETECTOR), REMOTE_ROOT + "/capt_head/subpixel_ball.py")
            sftp.put(str(FOUR_BALL), REMOTE_ROOT + "/capt_head/four_ball_pose.py")
            sftp.put(str(CONTRACT_CODE), REMOTE_ROOT + "/capt_head/measurement_contract.py")
            sftp.put(str(CAMERA_CONTROLS_CODE), REMOTE_ROOT + "/capt_head/camera_controls.py")
            sftp.put(str(INTERACTIVE_GEOMETRY),
                     REMOTE_ROOT + "/interactive_ball_calibration/geometry.py")
            sftp.put(str(CONTRACT), REMOTE_CONTRACT)
            sftp.put(str(FOUR_BALL_CONFIG), REMOTE_FOUR_BALL_CONFIG)
            sftp.put(str(CALIBRATION), REMOTE_CALIBRATION)
            sftp.put(str(RECTIFIED_STEREO), REMOTE_RECTIFIED_STEREO)
            sftp.chmod(REMOTE_SENDER, 0o755)
        running = remote(client, "pgrep -af '[o]rin_rectified_stream/sender.py' || true").strip()
        if args.restart or not running:
            print("停止ROS2/Apex相机后端并初始化GMSL……", flush=True)
            stop_all(client)
            remote(client, "sudo -n systemctl restart nvargus-daemon")
            remote(client, "cd /home/nvidia/cam_geac && "
                           "sudo -n ./demo/init_demo 3 > /home/nvidia/luna_camera_init_live.log 2>&1",
                   timeout=55)
            pose_option = (f" --four-ball-config {REMOTE_FOUR_BALL_CONFIG}"
                           if args.four_ball else "")
            evidence_option = (f" --evidence-dir {shlex.quote(args.evidence_dir)}"
                               if args.evidence_dir else "")
            command = (f"nohup python3 {REMOTE_SENDER} --contract {REMOTE_CONTRACT}"
                       f"{pose_option}{evidence_option} --host {local_ip} "
                       f"> {REMOTE_LOG} 2>&1 < /dev/null &")
            client.exec_command(command)
            print("等待Orin输出两路校正视频……", flush=True)
            for attempt in range(35):
                time.sleep(1)
                log = remote(client, f"tail -20 {REMOTE_LOG} 2>/dev/null || true")
                running = remote(client, "pgrep -af '[o]rin_rectified_stream/sender.py' || true").strip()
                if "Traceback" in log or not running:
                    raise RuntimeError("Orin校正视频提前退出：\n" + log)
                if attempt >= 5:
                    print(log.strip() or "Orin sender process started", flush=True)
                    break
            else:
                raise RuntimeError("等待校正视频超时；查看 " + REMOTE_LOG)
        else:
            print("复用Orin现有非ROS校正视频进程。", flush=True)
    finally:
        client.close()
        connection.close()
    print("打开本机校正双目画面；关闭窗口或按Ctrl+C只退出本机接收。", flush=True)
    subprocess.run(["python3", str(FULL_RECEIVER), "--contract", str(CONTRACT),
                    "--latest", str(ROOT / "runtime/new_sh5_yellow/latest_pose.json")], check=False)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError, paramiko.SSHException) as error:
        raise SystemExit(f"启动失败：{error}")
