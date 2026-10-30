"""Display the exclusive diagnostic camera; brightness enhancement is display-only."""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from data_collection.diagnostics.camera import Camera
from data_collection.diagnostics.core import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='6.6.7.100');p.add_argument('--user',default='nvidia')
    p.add_argument('--port',type=int,default=8768)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--saved-images',action='store_true',help='Read collector evidence without competing for camera frames')
    args=p.parse_args()
    sys.path.insert(0,str(ROOT/'acquisition'))
    from acquisition.capt_head.subpixel_ball import stereo_measure
    from dynamic_handeye.geometry import Assets
    assets=Assets();camera=None if args.saved_images else Camera(args.host,args.user,args.port);bright=not args.saved_images
    lut=np.rint(255*(np.arange(256)/255)**(1/1.6)).astype(np.uint8)
    assert lut[0]==0 and lut[-1]==255 and np.all(lut.astype(int)>=np.arange(256))
    name='Luna diagnostic stereo - B brightness - Q close'
    try:
        while True:
            try:
                if args.saved_images:
                    files=list(args.session.glob('*/camera_evidence.jsonl'))
                    if not files:time.sleep(.1);continue
                    file=max(files,key=lambda f:f.stat().st_mtime)
                    lines=file.read_text().splitlines();record=json.loads(lines[-1])
                    images=[cv2.imread(str(file.parent/record['image_files'][f'{eye}_rectified.png']['path'])) for eye in ('left','right')]
                    if any(image is None for image in images):time.sleep(.1);continue
                else:metadata,identity,blobs,images=camera.capture(camera.barrier())
                measurement=stereo_measure(images,assets.contract,stereo=assets.stereo)
                previews=[cv2.resize(image,(640,496)) for image in images]
                if bright:previews=[cv2.LUT(image,lut) for image in previews]
                if measurement:
                    for eye,frame in zip(('left','right'),previews):
                        center=tuple(np.rint(np.array(measurement[eye]['center'])*.5).astype(int))
                        cv2.drawMarker(frame,center,(0,255,0),cv2.MARKER_CROSS,18,1)
                display=np.hstack(previews);footer=np.zeros((60,1280,3),np.uint8)
                folder=file.parent if args.saved_images else args.session/'optical'
                state=folder/'status.json';progress='waiting'
                if state.exists():
                    s=json.loads(state.read_text());progress=f"{folder.name} {s.get('saved',s.get('attempted',0))}/{s['planned']}"
                events=folder/'events.jsonl'
                if events.exists() and 'session_abort' in events.read_text().splitlines()[-1]:progress+=' STOPPED'
                mode='DISPLAY gamma1.6 ONLY' if bright else ('SAVED DETECTION INPUT gamma1.6' if args.saved_images else 'Original display')
                cv2.putText(footer,'LEFT | RIGHT  Fixed exposure 5ms gain1  '+mode,(10,22),0,.6,(0,255,255),1)
                cv2.putText(footer,progress+' | detection '+(measurement['quality'] if measurement else 'REJECT')+' | saved images unchanged',(10,47),0,.6,(0,255,255),1)
                cv2.imshow(name,np.vstack((display,footer)))
            except (OSError,ValueError,KeyError,json.JSONDecodeError):time.sleep(.05)
            key=cv2.waitKey(1)&255
            if key==ord('q'):break
            if key==ord('b'):bright=not bright
    finally:
        if camera is not None:camera.close()
        cv2.destroyAllWindows()


if __name__=='__main__':main()
