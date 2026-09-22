#!/bin/bash
# chain_auto.sh — D 완주 후 평가·층화를 전부 자동 진행 (2026-09-22)
# · 완료 판정은 **결과 파일**로만 한다 (pgrep -f 는 자기 셸을 잡는다 · CLAUDE.md §4-17)
# · 시각은 KST 로 찍는다 (서버는 UTC · CLAUDE.md §4-18)
# · **판정 임계값을 스크립트에 넣는다** (09-21 중복률 26.4% 를 측정만 하고 지나친 실수 방지)
cd /home/se/JupyterLAB/Capstone/drone_yolo
P=/home/se/miniconda3/envs/drone/bin/python
R=metrics/AUTO_RESULT.md
log(){ echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*"; }
say(){ echo "$*" >> $R; }

: > $R
say "# 자동 실행 결과 — $(TZ=Asia/Seoul date '+%F %H:%M KST')"
say ""

# ── ① D 학습 완주 대기 (30에폭 행이 results.csv 에 찍히면 끝)
log "① D(vd_11m) 완주 대기"
for i in $(seq 1 240); do
  [ -f runs_person/vd_11m/results.csv ] && [ "$(awk -F, 'END{print $1}' runs_person/vd_11m/results.csv)" = "30" ] && break
  sleep 60
done
E=$(awk -F, 'END{print $1}' runs_person/vd_11m/results.csv 2>/dev/null)
log "D 상태: ${E:-없음} 에폭"
say "## D — 항공 사전학습 (vd_11m)"
say ""
say "\`\`\`"
awk -F, 'NR>1{printf "%-4s mAP50 %.4f  mAP50-95 %.4f\n",$1,$8,$9}' runs_person/vd_11m/results.csv 2>/dev/null | tail -5 >> $R
say "\`\`\`"
say ""
W=runs_person/vd_11m/weights/best.pt
[ -s "$W" ] || { log "가중치 없음 — 기준선으로 대체"; W=runs_person/fov_11m_1280_all/weights/best.pt; }

# ── ② D 판정 (기준선 대비)
log "② D 판정 — 가림 40칸 · 타일"
$P eval_fov.py  --weights $W --tag fovbudget > logs/auto_fov.log  2>&1
$P eval_tile.py --weights $W                 > logs/auto_tile.log 2>&1
$P - <<'PYX' >> $R 2>&1
import csv,glob,os
def avg(f,c="recall@0.15"):
    rs=list(csv.DictReader(open(f,encoding="utf-8")))
    v=[float(r[c]) for r in rs if r.get(c) not in ("","nan",None)]
    return (sum(v)/len(v), len(v)) if v else (float('nan'),0)
print("### 가림 지형 40칸 평균 재현율@0.15\n")
print("| 모델 | 값 | 기준선 대비 |")
print("|---|---:|---:|")
base,_=avg("metrics/eval_fovbudget_fov_11m_1280_all.csv")
print(f"| 기준선 fov_11m | {base:.4f} | — |")
for f in sorted(glob.glob("metrics/eval_fov*vd_11m*.csv")):
    a,n=avg(f); print(f"| vd_11m | {a:.4f} | **{(a-base)*100:+.1f} %p** |")
    print(f"\n판정: {'✅ 개선' if a>base else '❌ 실패'} (성공 조건: 기준선 초과)\n")
PYX

# ── ③ 마운트각 태그 검증 (Okutama 두 부류)
log "③ 마운트각 태그 검증"
say "## 마운트각 태그 검증 (Okutama 비스듬 vs 수직)"
say ""
say "\`\`\`"
$P eval_tilt_groups.py --weights $W >> $R 2>&1
say "\`\`\`"
say ""

# ── ④ 고도별 층화 (NOMAD a10~a90)
log "④ 고도별 층화"
say "## 고도별 층화"
say ""
say "\`\`\`"
$P eval_altitude.py --weights $W --imgsz 1280 >> $R 2>&1
say "\`\`\`"
say ""

# ── ⑤ Unicamp 표본 분석 (다운로드 끝났으면)
log "⑤ Unicamp 크기 분포"
say "## Unicamp-UAV 크기 분포"
say ""
say "\`\`\`"
$P - <<'PYY' >> $R 2>&1
import glob
import numpy as np
from PIL import Image
fs=sorted(glob.glob("data/raw/unicamp_uav/train/images/*.jpg"))[:800]
px=[]
for p in fs:
    t=p.replace("/images/","/labels/").rsplit(".",1)[0]+".txt"
    try: rows=[l.split() for l in open(t).read().splitlines() if l.strip()]
    except OSError: continue
    with Image.open(p) as im: W,H=im.size
    px+= [max(float(r[3])*W,float(r[4])*H) for r in rows]
if px:
    a=np.array(px)
    print(f"표본 {len(fs)}장 · 사람 {len(a):,}")
    print(f"긴 변 px 5/25/50/75/95: {np.percentile(a,[5,25,50,75,95]).round(1)}")
    print(f"운용 구간(18~50 px) 비율: {((a>=18)&(a<=50)).mean():.1%}")
else:
    print("라벨을 못 찾음 — 폴더 구조 확인 필요")
    import os
    print(sorted(os.listdir("data/raw/unicamp_uav"))[:8])
PYY
say "\`\`\`"

log "완료 → $R"
