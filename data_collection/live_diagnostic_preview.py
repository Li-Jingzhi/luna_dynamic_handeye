"""Preview live camera cache without consuming measurement pairs or reading evidence files."""
import argparse
import threading
import time
import cv2
import numpy as np
from data_collection.diagnostics.camera import connect,request


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--host',default='6.6.7.100');args=parser.parse_args()
    connection=connect(args.host,'nvidia');stop=threading.Event();latest=[None,0.,0];stats=[0,time.monotonic(),0]
    def receive():
        while not stop.is_set():
            started=time.monotonic()
            try:
                blob=request(connection,8768,'/preview.jpg');image=cv2.imdecode(np.frombuffer(blob,np.uint8),cv2.IMREAD_COLOR)
                if image is None:raise ValueError('Invalid live preview JPEG')
                latest[:]=[image,time.monotonic(),len(blob)];stats[0]+=1;stats[2]+=len(blob)
                if time.monotonic()-stats[1]>=5:
                    elapsed=time.monotonic()-stats[1]
                    print(f'LIVE preview {stats[0]/elapsed:.1f}fps, {stats[2]/max(stats[0],1)/1024:.0f}KB/frame',flush=True)
                    stats[:]=[0,time.monotonic(),0]
            except (ValueError,OSError):pass
            stop.wait(max(.01,.125-(time.monotonic()-started)))
    threading.Thread(target=receive,daemon=True).start();shown=None
    try:
        while True:
            if latest[0] is not None and shown!=latest[1]:
                image=latest[0].copy();cv2.putText(image,'Live stereo | Q close | recording runs separately',(10,48),0,.6,(0,255,255),1)
                cv2.imshow('Luna LIVE diagnostic stereo - Q close',image);shown=latest[1]
            if cv2.waitKey(50)&255==ord('q'):break
    finally:stop.set();connection.close();cv2.destroyAllWindows()


if __name__=='__main__':main()
