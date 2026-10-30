#!/usr/bin/env python3
"""Orin-only opt-in camera service: fixed ISP, common clock, dual10ms, VPI/CUDA."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
import uuid
import zipfile
import cv2
import numpy as np

try:
    from camera_controls import pipeline_properties,controls,verify_readback
    from stereo_pairing import StereoPairer
except ModuleNotFoundError:
    from acquisition.capt_head.camera_controls import pipeline_properties,controls,verify_readback
    from stereo_calibration.stereo_pairing import StereoPairer

SERVICE='luna_diagnostic_camera_v1'
EVIDENCE_ROOT=Path('/home/nvidia/luna_diagnostic_data')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence_folder(relative):
    path=EVIDENCE_ROOT/relative
    if not relative or not path.resolve().is_relative_to(EVIDENCE_ROOT.resolve()) or path.resolve()==EVIDENCE_ROOT.resolve():
        raise ValueError('Evidence directory outside diagnostic storage')
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',required=True);p.add_argument('--stereo',required=True);p.add_argument('--controls',required=True)
    p.add_argument('--port',type=int,default=8768);p.add_argument('--run',action='store_true')
    args=p.parse_args();raw=json.loads(Path(args.raw).read_text());stereo=json.loads(Path(args.stereo).read_text());profile=json.loads(Path(args.controls).read_text())
    expected=dict(source_calibration_sha256=sha(args.raw),stereo_model_sha256=sha(args.stereo),controls_sha256=sha(args.controls))
    if stereo['source_calibration_sha256']!=expected['source_calibration_sha256'] or raw['sensor_ids']!=stereo['sensor_ids']:
        raise ValueError('Camera assets mismatch')
    if raw['image_size']!=[2560,1984] or stereo['image_size']!=[1280,992]:raise ValueError('Wrong camera resolution')
    if profile['mode']!='fixed':raise ValueError('Diagnostic camera requires fixed exposure/gain/WB')
    pipeline_properties(profile,0);pipeline_properties(profile,1)
    if not args.run:
        print(json.dumps(dict(service=SERVICE,hardware_started=False,**expected)));return
    import gi
    gi.require_version('Gst','1.0')
    from gi.repository import Gst
    import vpi
    Gst.init(None);pairer=StereoPairer(max_skew_ms=10);lock=threading.Lock();gpu_lock=threading.Lock()
    serial=[0,0];fault=[];started=time.monotonic();process_id=uuid.uuid4().hex
    preview_frames=[None,None];preview=[None,0.];preview_stop=threading.Event()
    scale=np.diag([.5,.5,1.])
    maps=[vpi.WarpMap.fisheye_correction(vpi.WarpGrid((1280,992)),Kin=(scale@np.array(raw[f'{eye}_K']))[:2],
        X=np.c_[raw[f'rectify_R{i}'],np.zeros(3)],Kout=(scale@np.array(raw[f'rectify_P{i}'])[:,:3])[:2],
        coeffs=np.array(raw[f'{eye}_D']),mapping=vpi.FisheyeMapping.EQUIDISTANT) for i,eye in enumerate(('left','right'),1)]
    descriptions=[f'nvarguscamerasrc name=cam{i} sensor-id={i} sensor-mode=0 {pipeline_properties(profile,i)} ! '
        'video/x-raw(memory:NVMM),width=2560,height=1984,framerate=30/1,format=NV12 ! '
        'nvvidconv ! video/x-raw,width=1280,height=992,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! '
        f'appsink name=sink{i} emit-signals=true sync=false drop=true max-buffers=1' for i in range(2)]
    capture=Gst.parse_launch(' '.join(descriptions));capture.use_clock(Gst.SystemClock.obtain())
    cameras=[capture.get_by_name(f'cam{i}') for i in range(2)]
    def receive(sink,eye):
        arrival=time.monotonic_ns();sample=sink.emit('pull-sample')
        if sample is None:return Gst.FlowReturn.ERROR
        buffer=sample.get_buffer();pts=int(buffer.pts)
        if pts==Gst.CLOCK_TIME_NONE:return Gst.FlowReturn.OK
        ok,mapped=buffer.map(Gst.MapFlags.READ)
        if not ok:return Gst.FlowReturn.ERROR
        try:image=np.frombuffer(mapped.data,np.uint8).reshape(992,-1)[:,:1280*3].reshape(992,1280,3).copy()
        finally:buffer.unmap(mapped)
        try:
            with lock:
                serial[eye]+=1;item=(image,pts,arrival,int(capture.get_base_time())+pts,serial[eye])
                preview_frames[eye]=item;pairer.push(eye,item)
        except ValueError as error:fault.append(str(error));return Gst.FlowReturn.ERROR
        return Gst.FlowReturn.OK
    for i in range(2):capture.get_by_name(f'sink{i}').connect('new-sample',receive,i)
    if capture.set_state(Gst.State.PLAYING)==Gst.StateChangeReturn.FAILURE:raise RuntimeError('Argus camera startup failed')
    ready=False;readback=[]
    def state():
        nonlocal ready,readback
        error=capture.get_bus().pop_filtered(Gst.MessageType.ERROR)
        if error:fault.append(str(error.parse_error()))
        if not ready and time.monotonic()-started>=5 and min(serial)>0:
            for camera in cameras:camera.set_property('aelock',True);camera.set_property('awblock',True)
            readback=[{k:camera.get_property(k) for k in controls(eye)} for camera,eye in zip(cameras,profile['eyes'])]
            verify_readback(profile,readback)
            ready=all(c.get_property('aelock') and c.get_property('awblock') for c in cameras)
        return dict(service=SERVICE,process_id=process_id,image_size=[1280,992],sensor_ids=stereo['sensor_ids'],
            controls_verified=ready and not fault,controls_readback=readback,camera_controls=profile,exposure_time_valid=False,
            server_code_sha256=sha(__file__),timestamp_basis='common_gst_and_arrival_not_sensor_exposure',
            error=fault[-1] if fault else None,**expected)
    def live_preview():
        lut=np.rint(255*(np.arange(256)/255)**(1/1.6)).astype(np.uint8)
        while not preview_stop.wait(.125):
            with lock:pair=list(preview_frames)
            if any(frame is None for frame in pair) or max(time.monotonic_ns()-f[2] for f in pair)>500_000_000:continue
            try:
                with gpu_lock,vpi.Backend.CUDA:
                    images=[np.asarray(vpi.asimage(pair[i][0]).remap(maps[i],interp=vpi.Interp.LINEAR).cpu()).copy() for i in range(2)]
                image=np.hstack([cv2.LUT(cv2.resize(frame,(640,496)),lut) for frame in images])
                cv2.putText(image,'LIVE PREVIEW gamma1.6 | preview frames are not measurement samples',(10,24),0,.6,(0,255,255),1)
                ok,jpeg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,75])
                if ok:preview[:]=[jpeg.tobytes(),time.monotonic()]
            except Exception as error:fault.append('Preview failed: '+str(error));return
    threading.Thread(target=live_preview,daemon=True).start()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def reply(self,code,body,kind='application/json'):
            self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            try:
                route=urlparse(self.path);metadata=state()
                if route.path=='/status':self.reply(200,json.dumps(metadata).encode());return
                if route.path=='/barrier':self.reply(200,json.dumps(dict(process_id=process_id,after_ns=time.monotonic_ns())).encode());return
                if route.path=='/preview.jpg':
                    if preview[0] is None or time.monotonic()-preview[1]>1:raise ValueError('No fresh preview')
                    self.reply(200,preview[0],'image/jpeg');return
                if route.path!='/capture':self.reply(404,b'{}');return
                query=parse_qs(route.query);after=int(query.get('after_ns',['0'])[0])
                gamma=float(query.get('gamma',['1'])[0]);save=query.get('save',['1'])[0]=='1'
                if not 1<=gamma<=2:raise ValueError('Invalid gamma')
                relative=query.get('store',[''])[0];folder=evidence_folder(relative) if relative else None
                with lock:pair=pairer.take_latest()
                if not metadata['controls_verified'] or pair is None:raise ValueError('Waiting for fixed controls and paired frames')
                if min(f[2] for f in pair)<=after or max(time.monotonic_ns()-f[2] for f in pair)>200_000_000:raise ValueError('Frame outside stationary/freshness window')
                if abs(pair[0][2]-pair[1][2])>10_000_000 or abs(pair[0][3]-pair[1][3])>10_000_000:raise ValueError('Dual pairing gate failed')
                with gpu_lock,vpi.Backend.CUDA:
                    rectified=[np.asarray(vpi.asimage(pair[i][0]).remap(maps[i],interp=vpi.Interp.LINEAR).cpu()).copy() for i in range(2)]
                metadata.update(source_pts_ns=[f[1] for f in pair],source_clock_ns=[f[3] for f in pair],
                    arrival_monotonic_ns=[f[2] for f in pair],frame_number=[f[4] for f in pair],
                    pairing_policy='shared_pipeline_source_and_arrival_10ms',input_domain='ISP_BGR_scaled_uncorrected',pixel_domain='ideal_pinhole',
                    rectification='VPI_CUDA_EQUIDISTANT_LINEAR')
                lut=np.rint(255*(np.arange(256)/255)**(1/gamma)).astype(np.uint8)
                detection=[cv2.LUT(image,lut) if gamma!=1 else image for image in rectified]
                evidence_id=uuid.uuid4().hex;stored={};encoded={}
                for i,eye in enumerate(('left','right')):
                    values={f'{eye}_rectified.png':detection[i]}
                    if save:
                        values.update({f'{eye}_input.png':pair[i][0],f'{eye}_rectified_unenhanced.png':rectified[i]})
                    for name,image in values.items():
                        ok,png=cv2.imencode('.png',image)
                        if not ok:raise OSError('PNG encoding failed')
                        blob=png.tobytes()
                        if save and folder is not None:
                            dest=folder/'images'/evidence_id/name;dest.parent.mkdir(parents=True,exist_ok=True)
                            with dest.open('xb') as out:out.write(blob)
                            stored[name]=dict(path=str(dest.relative_to(folder)),sha256=hashlib.sha256(blob).hexdigest())
                        if name.endswith('_rectified.png') or (save and folder is None):encoded[name]=blob
                metadata.update(image_processing=dict(gamma=gamma,detection_images='left_rectified.png,right_rectified.png',sensor_exposure_unchanged=True),
                    evidence_id=evidence_id,stored_evidence=stored,evidence_session=relative)
                data=io.BytesIO()
                with zipfile.ZipFile(data,'w',compression=zipfile.ZIP_STORED) as archive:
                    for name,blob in encoded.items():archive.writestr(name,blob)
                    archive.writestr('metadata.json',json.dumps(metadata))
                self.reply(200,data.getvalue(),'application/zip')
            except (ValueError,OSError) as error:self.reply(503,json.dumps(dict(error=str(error))).encode())
    try:ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
    finally:preview_stop.set();capture.set_state(Gst.State.NULL)


if __name__=='__main__':main()
