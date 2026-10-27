#!/usr/bin/env python3
"""Deploy/start the direct-CSI capture server and launch the local board UI."""
import argparse
import getpass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(Path(__file__).with_name('capture_config.json')))
    p.add_argument('--auto', action='store_true')
    p.add_argument('--resume', help='existing automatic capture directory')
    p.add_argument('--password', action='store_true', help='prompt for SSH password; otherwise use SSH keys/agent')
    a = p.parse_args()
    if a.resume and not a.auto: p.error('--resume requires --auto')
    os.environ['LUNA_CAPTURE_CONFIG'] = str(Path(a.config).resolve())
    if a.password: os.environ['LUNA_CAPTURE_PASSWORD'] = getpass.getpass('Orin SSH password: ')
    from capture_settings import SETTINGS, connect
    from orin_pro_calibration_gui import request
    root = Path(__file__).resolve().parent
    client = connect()
    try:
        # Reuse only our service with matching CSI settings. Never kill other camera applications.
        try:
            status = json.loads(request(client, '/status'))
        except Exception:
            status = None
        if status is None:
            with client.open_sftp() as sftp:
                remote = sftp.normalize('.') + '/luna_stereo_calibration'
                try: sftp.stat(remote)
                except IOError: sftp.mkdir(remote)
                sftp.put(str(root/'orin_pro_capture_server.py'), remote+'/capture_server.py')
                sftp.put(str(root/'stereo_pairing.py'), remote+'/stereo_pairing.py')
            command = ['python3', '-u', remote+'/capture_server.py', '--sensors',
                       *map(str, SETTINGS['sensors']), '--port', str(SETTINGS['port'])]
            shell = 'nohup ' + shlex.join(command) + ' > ' + shlex.quote(remote+'/capture_server.log') + ' 2>&1 < /dev/null &'
            _, out, _ = client.exec_command(shell)
            out.channel.recv_exit_status()
            for _ in range(30):
                try:
                    status = json.loads(request(client, '/status'))
                    break
                except Exception: time.sleep(.5)
            if status is None:
                raise RuntimeError(f'Capture server unavailable; inspect {remote}/capture_server.log on Orin')
        if status.get('service') != 'luna_stereo_direct_csi_v2' or status.get('sensors') != SETTINGS['sensors']:
            raise RuntimeError('Port belongs to a different capture service/config; use a free port or stop that service explicitly')
        if status.get('error'): raise RuntimeError(status['error'])
    finally:
        client.close()
    entry = 'auto_capture_orin_pro.py' if a.auto else 'orin_pro_calibration_gui.py'
    command = [sys.executable, str(root/entry)]
    if a.resume: command += ['--resume', a.resume]
    process = subprocess.Popen(command, env=os.environ.copy())
    try: code = process.wait()
    except KeyboardInterrupt: code = process.wait()
    if code: raise SystemExit(code)


if __name__ == '__main__': main()
