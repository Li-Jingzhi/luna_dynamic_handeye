#!/usr/bin/env python3
"""Desktop controls for the existing Orin sender; no robot commands."""
import json
import os
import shlex
import sys
import threading
import time
from pathlib import Path

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capt_head.camera_controls import validate_profile
from orin_rectified_stream.start import connect, remote, REMOTE_ROOT, REMOTE_SENDER, REMOTE_CONTRACT


def make_profile(values, wb):
    eye = dict(values, wbmode=wb)
    eye["exposure_ns"] = round(eye.pop("exposure_ms") * 1e6)
    for key in ("tnr_mode", "ee_mode"):
        eye[key] = round(eye[key])
    return validate_profile(dict(mode="fixed_exposure_auto_wb" if wb == 1 else "fixed",
                                 eyes=[dict(eye), dict(eye)]))


def restart_sender(client, profile):
    """Release both Argus sessions before reacquiring either camera."""
    path = REMOTE_ROOT + "/runtime_manual_camera.json"
    with client.open_sftp() as sftp:
        with sftp.file(path, "w") as stream:
            stream.write(json.dumps(profile))
    remote(client, "pkill -INT -f '[o]rin_rectified_stream/sender.py' || true")
    for _ in range(10):
        if not remote(client, "pgrep -f '[o]rin_rectified_stream/sender.py' || true").strip():
            break
        time.sleep(.5)
    else:
        remote(client, "pkill -KILL -f '[o]rin_rectified_stream/sender.py' || true")
    # Argus can retain a dead CaptureSession even after the process exits.
    remote(client, "sudo -n systemctl restart nvargus-daemon")
    time.sleep(2)
    remote(client, "cd /home/nvidia/cam_geac && sudo -n ./demo/init_demo 3 "
                   "> /home/nvidia/luna_manual_camera_init.log 2>&1")
    command = (f"nohup python3 -u {shlex.quote(REMOTE_SENDER)} --contract {shlex.quote(REMOTE_CONTRACT)} "
               f"--camera-profile {shlex.quote(path)} --host 6.6.7.101 "
               "> /home/nvidia/luna_rectified_sender.log 2>&1 < /dev/null &")
    remote(client, command)
    for _ in range(20):
        time.sleep(1)
        log = remote(client, "cat /home/nvidia/luna_rectified_sender.log")
        running = remote(client, "pgrep -f '[o]rin_rectified_stream/sender.py' || true").strip()
        if not running or any(error in log for error in ("Traceback", "Failed to create CaptureSession", "ERROR")):
            raise RuntimeError(log[-2400:] or "发送器已退出，请恢复基线")
        if log.count("Producer has connected") >= 2 and _ >= 7:
            return
    raise RuntimeError("等待两路相机启动超时；请恢复基线。" + log[-1000:])


class Panel(Gtk.Window):
    def __init__(self):
        super().__init__(title="双目相机手动调节 — 左右同步，不控制机械臂")
        self.set_default_size(650, 600)
        self.connect("destroy", Gtk.main_quit)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin=16)
        self.add(box)
        box.pack_start(Gtk.Label(label="拖动仅修改草稿；点击应用会重置相机，约需 40–60 秒。\n先降低曝光/增益，再调白平衡。自动白平衡不能保证每次启动颜色相同。"), False, False, 0)
        self.scales = {}
        specs = [("exposure_ms", "曝光 ms", .04, 10, .01, 5),
                 ("analog_gain", "模拟增益", 1, 4, .05, 1),
                 ("digital_gain", "数字增益", 1, 8, .05, 1),
                 ("saturation", "饱和度", 0, 2, .05, 1),
                 ("tnr_mode", "降噪模式 0关/1快/2高", 0, 2, 1, 1),
                 ("tnr_strength", "降噪强度", 0, 1, .05, .2),
                 ("ee_mode", "锐化模式 0关/1快/2高", 0, 2, 1, 0),
                 ("ee_strength", "锐化强度", 0, 1, .05, 0)]
        for key, label, low, high, step, value in specs:
            row = Gtk.Box(spacing=10)
            row.pack_start(Gtk.Label(label=label, width_chars=23, xalign=0), False, False, 0)
            scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, low, high, step)
            scale.set_value(value)
            scale.set_hexpand(True)
            scale.set_digits(0 if key.endswith("mode") else 2)
            row.pack_start(scale, True, True, 0)
            self.scales[key] = scale
            box.pack_start(row, False, False, 0)
        self.wb = Gtk.ComboBoxText()
        for key, name in [(1,"自动白平衡（重启不保证一致）"), (0,"关闭"), (2,"白炽灯"),
                          (3,"荧光灯"), (4,"暖荧光灯"), (5,"日光"), (6,"阴天"), (7,"黄昏"), (8,"阴影")]:
            self.wb.append(str(key), name)
        self.wb.set_active_id("1")
        box.pack_start(self.wb, False, False, 0)
        buttons = Gtk.Box(spacing=8)
        self.buttons = buttons
        for label, action in [("应用预览", self.apply), ("恢复原较暗自动基线", self.restore), ("保存当前草稿", self.save)]:
            button = Gtk.Button(label=label)
            button.connect("clicked", action)
            buttons.pack_start(button, True, True, 0)
        box.pack_start(buttons, False, False, 0)
        self.status = Gtk.Label(label="滑块是草稿值，不代表当前相机设置；点击应用后生效。", wrap=True)
        box.pack_start(self.status, False, False, 0)
        box.pack_start(Gtk.Label(label="调参期间暂停数据采集。保存仅写草稿，不覆盖正式标定配置。\n关闭本面板不会停止视频。正式锁参前需检查黄球过曝与重启一致性。", wrap=True), False, False, 0)

    def profile(self):
        return make_profile({k: s.get_value() for k, s in self.scales.items()}, int(self.wb.get_active_id()))

    def save(self, _):
        path = ROOT / "runtime/manual_camera_profile.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(self.profile(), ensure_ascii=False, indent=2) + "\n")
        self.status.set_text(f"已保存草稿：{path}；尚未修改正式开机配置。")

    def restore(self, _):
        self.restart(None)

    def apply(self, _):
        self.restart(self.profile())

    def restart(self, profile):
        self.buttons.set_sensitive(False)
        self.status.set_text("正在应用，请等待新画面；此过程不移动机械臂……")
        def worker():
            client = connection = None
            try:
                password = os.environ.get("LUNA_ORIN_PASSWORD")
                client, connection = connect("6.6.7.100", "nvidia", password, "6.6.7.101")
                restart_sender(client, profile)
                message = "已应用。请看双目画面确认效果；若过亮先减曝光/增益。" if profile else "已恢复原自动曝光/白平衡基线。"
            except Exception as exc:
                message = f"应用失败（未移动机械臂）：{exc}"
            finally:
                if client:
                    client.close()
                if connection:
                    connection.close()
            GLib.idle_add(self.finished, message)
        threading.Thread(target=worker, daemon=True).start()

    def finished(self, message):
        self.status.set_text(message)
        self.buttons.set_sensitive(True)
        return False


if __name__ == "__main__":
    Panel().show_all()
    Gtk.main()
