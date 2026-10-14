"""SIM detector using the existing leased analysis API, with durable completion retry.

v1.2 (비전 · 10-14): worker 가 주어지면 같은 IngestWorker.observe 로 임무마다 후보 · 추적 · 색 · 상태를 이어 간다
  · candidate_id = 임무 안에서 같은 사람이면 같은 id (v1.1 은 프레임마다 새 id)
  · geo_result: 기본은 v1.1 그대로 (PENDING · 사유) · VISION_GEO_CLAIMS=1 이면 pose 로 계산한 주장 (공식 승격은 백엔드)
  · SIM_RICH_FIELDS=1 이면 appearance · state · track_id 도 보냄 — **백엔드 SIM Observation 계약에 칸이 생긴 뒤에만** (지금은 extra=forbid)
"""
import hashlib
import json
import os
from uuid import UUID,uuid5
import cv2
import numpy as np

def save(path,value):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def observations(job,frame,data,detector,worker=None):
    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if image is None or (image.shape[1],image.shape[0])!=(frame['width'],frame['height']):raise ValueError('Frame decode/size mismatch')
    if worker is not None:return rich_observations(job,frame,image,worker)
    boxes,scores,_=detector.detect(image)
    result=[];ns=UUID(job['analysis_run_id'])
    for i in np.argsort(-scores)[:128]:
        x1,y1,x2,y2=map(float,boxes[i]);w,h=frame['width'],frame['height']
        x1,y1=max(0,x1),max(0,y1);x2,y2=min(w,x2),min(h,y2)
        if x1>=x2 or y1>=y2:continue
        index=len(result)
        result.append({'observation_id':str(uuid5(ns,'observation/'+str(index))),
            'candidate_id':str(uuid5(ns,'candidate/'+str(index))),'detection_index':index,
            'bbox':{'convention':'PIXEL_XYXY','source_width':w,'source_height':h,'x1':x1,'y1':y1,'x2':x2,'y2':y2},
            'confidence':float(scores[i]),'geo_result':{'status':'PENDING','reason':'SIM_PHYSICAL_VALIDATION_REQUIRED' if frame.get('pose') else 'POSE_MISSING'}})
    return result

def rich_observations(job,frame,image,worker):
    """v1.2 — 임무 단위 기억 (후보 · 추적 · 색 · 상태) · SIM 계약 칸에 맞춰 줄인다"""
    from datetime import datetime
    import time as _t
    t=datetime.fromisoformat(frame['capture_at'].replace('Z','+00:00')).timestamp() if frame.get('capture_at') else _t.time()
    obs=worker.observe(str(frame.get('mission_id') or job.get('mission_id') or 'sim'),image,t,frame.get('pose'),query=worker.query_for(frame),upload_crops=False)
    ns=UUID(job['analysis_run_id']);w,h=frame['width'],frame['height'];rich=os.environ.get('SIM_RICH_FIELDS','0')=='1'
    out=[]
    for o in obs:
        b=o['bbox'];g=o['geo']
        geo=g if worker.geo_claims else {'status':'PENDING','reason':'SIM_PHYSICAL_VALIDATION_REQUIRED' if frame.get('pose') else 'POSE_MISSING'}
        row={'observation_id':str(uuid5(ns,'observation/'+str(len(out)))),'candidate_id':o['candidate_id'],'detection_index':len(out),
             'bbox':{'convention':'PIXEL_XYXY','source_width':w,'source_height':h,'x1':b['x1'],'y1':b['y1'],'x2':b['x2'],'y2':b['y2']},
             'confidence':o['confidence'],'geo_result':geo}
        if rich:row.update(appearance=o['appearance'],state=o['state'],track_id=o['track_id'])
        out.append(row)
    return out

def poll_once(api,token,detector,active,root,stats=None,worker=None):
    headers={'Authorization':'Bearer '+token()};pending=root/'sim-completion.json'
    if pending.exists():
        value=json.loads(pending.read_text())
        r=api.post('/internal/v1/analysis-runs/'+value['rid']+'/completion',json=value['body'],headers=headers)
        if r.status_code==409:
            os.replace(pending,root/('sim-expired-'+value['body']['claim_id']+'.json'))
        else:
            r.raise_for_status();pending.unlink()
            if stats is not None:
                stats['frames']+=1;stats['observations']+=len(value['body']['observations'])
        return 1
    r=api.post('/internal/v1/analysis-runs/claim',params={'model_config_id':active['model_config_id']},headers=headers)
    r.raise_for_status();job=r.json()
    if job.get('job') is None and 'analysis_run_id' not in job:return 0
    rid=job['analysis_run_id'];base='/internal/v1/analysis-runs/'+rid
    try:
        f=api.get(base+'/frame',headers=headers);f.raise_for_status();frame=f.json()
        data=api.get(base+'/input',headers=headers);data.raise_for_status()
        if hashlib.sha256(data.content).hexdigest()!=job['input_hash'] or data.headers.get('X-Content-SHA256')!=job['input_hash']:raise ValueError('Input hash mismatch')
        body={'claim_id':job['claim_id'],'attempt_no':job['attempt_no'],'observations':observations(job,frame,data.content,detector,worker)}
        save(pending,{'rid':rid,'body':body})
    except Exception:
        api.post(base+'/attempt-failures',headers=headers,json={'claim_id':job['claim_id'],'attempt_no':job['attempt_no'],'reason':'INFERENCE_OR_INPUT_FAILED'}).raise_for_status()
        raise
    return 1
