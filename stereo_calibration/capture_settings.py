"""Local capture configuration; no dependency on the source project."""
import json
import os
from pathlib import Path

import paramiko


def load_settings(path):
    config = json.loads(Path(path).read_text())
    if not config.get('host') or not config.get('username'):
        raise ValueError('host and username are required')
    ids = config['sensors']; board = config['board_inner_corners']
    if len(ids) != 2 or any(type(i) is not int or i < 0 for i in ids) or ids[0] == ids[1]:
        raise ValueError('sensors must contain distinct left/right CSI IDs')
    if len(board) != 2 or any(type(i) is not int or i < 3 for i in board):
        raise ValueError('invalid inner corner counts')
    import math
    if not math.isfinite(config['square_mm']) or config['square_mm'] <= 0:
        raise ValueError('invalid measured square length')
    if type(config['port']) is not int or not 1 <= config['port'] <= 65535:
        raise ValueError('invalid port')
    return config


SETTINGS = load_settings(os.environ.get('LUNA_CAPTURE_CONFIG', Path(__file__).with_name('capture_config.json')))


def connect():
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SETTINGS['host'], username=SETTINGS['username'],
                   password=os.environ.get('LUNA_CAPTURE_PASSWORD'), timeout=10)
    return client
