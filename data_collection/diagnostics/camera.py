"""SSH-localhost transport; no network connection during import or offline checks."""
import io
import json
import os
from pathlib import Path
import shlex
import time
import zipfile
import cv2
import numpy as np
from .core import ROOT,digest,validate_metadata


def connect(host,user):
    import paramiko
    client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host,username=user,password=os.environ.get('LUNA_DIAG_PASSWORD'),timeout=10)
    return client


def request(client,port,path):
    channel=client.get_transport().open_channel('direct-tcpip',('127.0.0.1',port),('127.0.0.1',0));channel.settimeout(30)
    try:
        channel.sendall(f'GET {path} HTTP/1.0\r\nHost: localhost\r\n\r\n'.encode());data=b''
        while b'\r\n\r\n' not in data:
            chunk=channel.recv(65536)
            if not chunk:raise OSError('Truncated camera response')
            data+=chunk
        header,body=data.split(b'\r\n\r\n',1)
        length=next(int(line.split(b':',1)[1]) for line in header.split(b'\r\n') if line.lower().startswith(b'content-length:'))
        if length>64_000_000:raise ValueError('Camera response too large')
        while len(body)<length:
            chunk=channel.recv(min(1_048_576,length-len(body)))
            if not chunk:raise OSError('Truncated camera body')
            body+=chunk
        if header.split()[1]!=b'200':raise ValueError(body.decode(errors='replace'))
        return body
    finally:channel.close()


def expected_assets():
    return dict(source_calibration_sha256=digest(ROOT/'configs/camera/luna_raw_fisheye_stereo.json'),
        stereo_model_sha256=digest(ROOT/'configs/camera/luna_rectified_stereo.json'),
        controls_sha256=digest(ROOT/'configs/camera/luna_v2_fixed_controls.json'))


class Camera:
    def __init__(self,host,user,port):self.client=connect(host,user);self.port=port;self.expected=expected_assets()
    def close(self):self.client.close()
    def barrier(self):return json.loads(request(self.client,self.port,'/barrier'))
    def capture(self,barrier):
        body=request(self.client,self.port,f"/capture?after_ns={barrier['after_ns']}")
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            metadata=json.loads(z.read('metadata.json'))
            if metadata['process_id']!=barrier['process_id']:raise ValueError('Camera restarted during capture')
            identity=validate_metadata(metadata,self.expected,barrier['after_ns'])
            blobs={name:z.read(name) for name in ('left_input.png','right_input.png','left_rectified.png','right_rectified.png')}
        images=[cv2.imdecode(np.frombuffer(blobs[f'{eye}_rectified.png'],np.uint8),cv2.IMREAD_COLOR) for eye in ('left','right')]
        if any(x is None or x.shape!=(992,1280,3) for x in images):raise ValueError('Wrong measurement images')
        return metadata,identity,blobs,images


def deployment(host,user,port,run=False):
    files={
        'camera_server.py':Path(__file__).with_name('camera_server.py'),
        'camera_controls.py':ROOT/'acquisition/capt_head/camera_controls.py',
        'stereo_pairing.py':ROOT/'stereo_calibration/stereo_pairing.py',
        'raw.json':ROOT/'configs/camera/luna_raw_fisheye_stereo.json',
        'stereo.json':ROOT/'configs/camera/luna_rectified_stereo.json',
        'controls.json':ROOT/'configs/camera/luna_v2_fixed_controls.json'}
    result=dict(host=host,user=user,port=port,files={k:digest(v) for k,v in files.items()},hardware_started=False)
    if not run:return result
    client=connect(host,user)
    try:
        try:status=json.loads(request(client,port,'/status'))
        except Exception:status=None
        if status is not None:
            for k,v in expected_assets().items():
                if status.get(k)!=v:raise ValueError('Existing camera service uses different assets; stop it explicitly')
            if status.get('server_code_sha256')!=digest(files['camera_server.py']):raise ValueError('Existing server code differs; stop it explicitly')
        else:
            with client.open_sftp() as sftp:
                remote=sftp.normalize('.')+'/luna_auto_diagnostics'
                try:sftp.stat(remote)
                except OSError:sftp.mkdir(remote)
                for name,path in files.items():sftp.put(str(path),remote+'/'+name)
            command=['python3','-u',remote+'/camera_server.py','--raw',remote+'/raw.json','--stereo',remote+'/stereo.json',
                     '--controls',remote+'/controls.json','--port',str(port),'--run']
            _,out,_=client.exec_command('nohup '+shlex.join(command)+' > '+shlex.quote(remote+'/camera.log')+' 2>&1 < /dev/null &')
            out.channel.recv_exit_status()
            for _ in range(30):
                try:status=json.loads(request(client,port,'/status'));break
                except Exception:time.sleep(.5)
            if status is None:raise RuntimeError('Camera unavailable; inspect Orin ~/luna_auto_diagnostics/camera.log')
        if status.get('error'):raise RuntimeError('Camera fault: '+str(status['error']))
        result.update(hardware_started=True,status=status)
        return result
    finally:client.close()
