"""SIM detector using the existing leased analysis API, with durable completion retry."""
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

def observations(job,frame,data,detector):
    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if image is None or (image.shape[1],image.shape[0])!=(frame['width'],frame['height']):raise ValueError('Frame decode/size mismatch')
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

def poll_once(api,token,detector,active,root,stats=None):
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
        body={'claim_id':job['claim_id'],'attempt_no':job['attempt_no'],'observations':observations(job,frame,data.content,detector)}
        save(pending,{'rid':rid,'body':body})
    except Exception:
        api.post(base+'/attempt-failures',headers=headers,json={'claim_id':job['claim_id'],'attempt_no':job['attempt_no'],'reason':'INFERENCE_OR_INPUT_FAILED'}).raise_for_status()
        raise
    return 1
