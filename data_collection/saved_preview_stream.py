"""Small JPEG stream of recorded evidence; never opens a camera or reads the entire log."""
import argparse
import json
from pathlib import Path
import struct
import sys
import threading
import time
import cv2
import numpy as np


def tail_record(path,limit=262144):
    with Path(path).open('rb') as stream:
        stream.seek(0,2);end=stream.tell();stream.seek(max(0,end-limit));lines=stream.read(limit).splitlines()
    for line in reversed(lines):
        try:return json.loads(line)
        except (ValueError,UnicodeError):continue
    raise ValueError('No complete recent record')


def serve(directory):
    last=None
    while True:
        try:
            files=list(directory.glob('*/frames.jsonl'))
            if not files:time.sleep(.2);continue
            file=max(files,key=lambda p:p.stat().st_mtime)
            row=tail_record(file);key=(str(file),row['id'])
            if key==last:time.sleep(.2);continue
            previews=[]
            for eye in ('left','right'):
                path=file.parent/row['image_files'][f'{eye}_rectified.png']['path']
                image=cv2.imread(str(path))
                if image is None:raise ValueError('Image not yet available')
                image=cv2.resize(image,(640,496))
                center=row['measurement'][eye]['center']
                cv2.drawMarker(image,tuple(np.rint(np.array(center)*.5).astype(int)),(0,255,0),cv2.MARKER_CROSS,18,1)
                previews.append(image)
            footer=np.zeros((60,1280,3),np.uint8);status=file.parent/'status.json';progress='first window in progress'
            if status.exists():
                s=json.loads(status.read_text());progress=f"{s.get('saved',s.get('attempted',0))}/{s['planned']}"
            cv2.putText(footer,f'{file.parent.name}: {progress} | recorded detection frame | gamma1.6 | Q close',(10,22),0,.6,(0,255,255),1)
            cv2.putText(footer,'Evidence preview only; does not consume camera frames or change acquisition.',(10,47),0,.55,(0,255,255),1)
            frame=np.vstack((np.hstack(previews),footer));ok,jpeg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
            if not ok:raise ValueError('JPEG encoding failed')
            data=jpeg.tobytes();sys.stdout.buffer.write(struct.pack('!I',len(data))+data);sys.stdout.buffer.flush();last=key
        except (OSError,ValueError,KeyError):time.sleep(.2)


def client(directory,host,user):
    from data_collection.diagnostics.camera import connect
    connection=connect(host,user);remote_script='/home/nvidia/luna_auto_diagnostics/saved_preview_stream.py'
    with connection.open_sftp() as sftp:sftp.put(__file__,remote_script)
    import shlex
    channel=connection.get_transport().open_session();channel.exec_command('python3 -u '+shlex.quote(remote_script)+' --serve --directory '+shlex.quote(str(directory)))
    channel.settimeout(5);stop=threading.Event();latest=[None,0.,0];error=[None]
    def read_exact(count):
        data=bytearray()
        while len(data)<count and not stop.is_set():
            try:chunk=channel.recv(count-len(data))
            except TimeoutError:continue
            if not chunk:raise OSError('Preview connection closed')
            data.extend(chunk)
        return bytes(data)
    def receive():
        while not stop.is_set():
            try:
                header=read_exact(4)
                if len(header)!=4:break
                size=struct.unpack('!I',header)[0]
                if not 0<size<4_000_000:raise ValueError('Invalid preview frame size')
                data=read_exact(size);image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                if image is None:raise ValueError('Invalid preview JPEG')
                latest[:]=[image,time.monotonic(),size];error[0]=None
            except TimeoutError:continue
            except (OSError,ValueError) as exc:error[0]=str(exc);break
    worker=threading.Thread(target=receive,daemon=True);worker.start()
    name='Luna lightweight diagnostic evidence preview - Q close'
    shown=None
    try:
        while not stop.is_set():
            if latest[0] is not None:
                age=time.monotonic()-latest[1];key=(latest[1],int(age))
                if shown!=key:
                    display=latest[0].copy()
                    cv2.putText(display,f'Last saved frame {age:.1f}s ago | {latest[2]/1024:.0f}KB',(10,28),0,.65,(0,255,255),1)
                    cv2.imshow(name,display);shown=key
            if cv2.waitKey(100)&255==ord('q'):break
            if error[0]:raise OSError(error[0])
    finally:
        stop.set();channel.close();connection.close();cv2.destroyAllWindows()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--serve',action='store_true');p.add_argument('--host',default='6.6.7.100');p.add_argument('--user',default='nvidia');args=p.parse_args()
    if args.serve:serve(args.directory)
    else:client(args.directory,args.host,args.user)


if __name__=='__main__':main()
