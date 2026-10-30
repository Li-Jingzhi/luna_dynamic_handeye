"""Display the exclusive diagnostic camera; brightness enhancement is display-only."""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from .camera import Camera
from .core import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='6.6.7.100');p.add_argument('--user',default='nvidia')
    p.add_argument('--port',type=int,default=8768)
    p.add_argument('--session',type=Path,required=True);args=p.parse_args()
    sys.path.insert(0,str(ROOT/'acquisition'))
    from acquisition.capt_head.subpixel_ball import stereo_measure
    from dynamic_handeye.geometry import Assets
    assets=Assets();camera=Camera(args.host,args.user,args.port);bright=True
    lut=np.rint(255*(np.arange(256)/255)**(1/1.6)).astype(np.uint8)
    assert lut[0]==0 and lut[-1]==255 and np.all(lut.astype(int)>=np.arange(256))
    name='Luna diagnostic stereo - B brightness - Q close'
    try:
        while True:
            try:
                metadata,identity,blobs,images=camera.capture(camera.barrier())
                measurement=stereo_measure(images,assets.contract,stereo=assets.stereo)
                previews=[cv2.resize(image,(640,496)) for image in images]
                if bright:previews=[cv2.LUT(image,lut) for image in previews]
                if measurement:
                    for eye,frame in zip(('left','right'),previews):
                        center=tuple(np.rint(np.array(measurement[eye]['center'])*.5).astype(int))
                        cv2.drawMarker(frame,center,(0,255,0),cv2.MARKER_CROSS,18,1)
                display=np.hstack(previews);footer=np.zeros((60,1280,3),np.uint8)
                state=args.session/'optical/status.json';progress='waiting'
                if state.exists():
                    s=json.loads(state.read_text());progress=f"optical {s['saved']}/{s['planned']}"
                events=args.session/'optical/events.jsonl'
                if events.exists() and 'session_abort' in events.read_text().splitlines()[-1]:progress+=' STOPPED'
                cv2.putText(footer,'LEFT | RIGHT  Fixed exposure 5ms gain1  '+('DISPLAY gamma1.6 ONLY' if bright else 'Original display'),(10,22),0,.6,(0,255,255),1)
                cv2.putText(footer,progress+' | detection '+(measurement['quality'] if measurement else 'REJECT')+' | saved images unchanged',(10,47),0,.6,(0,255,255),1)
                cv2.imshow(name,np.vstack((display,footer)))
            except (OSError,ValueError,KeyError,json.JSONDecodeError):time.sleep(.05)
            key=cv2.waitKey(1)&255
            if key==ord('q'):break
            if key==ord('b'):bright=not bright
    finally:camera.close();cv2.destroyAllWindows()


if __name__=='__main__':main()
