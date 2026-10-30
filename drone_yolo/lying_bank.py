"""
lying_bank.py — 누운 사람 인스턴스 은행 (10-30 · 복사-붙여넣기 증강 재료 · 판정 = _학습 큐 「10-30 CP」)

  출처 (학습 출처만 · 평가에 쓰는 Okutama · NOMAD 검증 배우 96~100 · UE 는 안 씀)
    Archangel-Real  주석 자세 lying down · crawling (load_ann) · 영상마다 10 프레임 간격 · 영상당 ≤ 30 · 박스 긴 변 ≥ 16 px
    NOMAD           배우 1~95 · 행동 Laying · Hiding (Laying) · 가림 표시 visibility ≥ 70 · a10 · a30 · 배우 × 거리마다 ≤ 4
  마스크 = SAM (facebook/sam-vit-base · 박스 프롬프트 · 3개 중 점수 최고) · 박스 안 마스크 비율 0.15~0.95 만
  출력 data/lying_bank/{arch,nomad}/<id>.png (RGBA · 여백 6 px) · data/lying_bank/meta.jsonl (git 밖 · 원본 파생물)
실행  /home/se/venvs/state/bin/python lying_bank.py
"""
import json, random, re, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
OUT = BASE / "data" / "lying_bank"


def sam():
    from transformers import SamModel, SamProcessor
    p = SamProcessor.from_pretrained("facebook/sam-vit-base")
    m = SamModel.from_pretrained("facebook/sam-vit-base").cuda().eval()
    return p, m


def mask_for(proc, model, rgb, box):
    inp = proc(images=rgb, input_boxes=[[list(map(float, box))]], return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = model(**inp)
    ms = proc.image_processor.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0][0]
    sc = o.iou_scores.cpu()[0, 0]
    k = int(torch.argmax(sc))
    return ms[k].numpy().astype(np.uint8), float(sc[k])


def save(img_bgr, mask, box, path, pad=6):
    H, W = mask.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    X1, Y1, X2, Y2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
    m = mask.copy(); keep = np.zeros_like(m); keep[Y1:Y2, X1:X2] = 1; m &= keep          # 박스 밖 마스크는 버림
    bm = m[y1:y2, x1:x2]
    ratio = bm.sum() / max(bm.size, 1)
    if not (0.15 <= ratio <= 0.95):
        return None
    rgba = np.dstack([img_bgr[Y1:Y2, X1:X2], m[Y1:Y2, X1:X2] * 255])
    cv2.imwrite(str(path), rgba)
    return round(float(ratio), 3)


def main():
    from make_v11_arch import load_ann, VID, NAME
    from make_posture_crops import NOMAD, _nomad_act
    proc, model = sam()
    (OUT / "arch").mkdir(parents=True, exist_ok=True); (OUT / "nomad").mkdir(parents=True, exist_ok=True)
    meta = open(OUT / "meta.jsonl", "w"); n = defaultdict(int)
    boxes, covered = load_ann()
    for vid in sorted(boxes):
        mp4 = VID / f"{vid}.mp4"
        if not mp4.exists():
            continue
        m = NAME.search(vid); alt = int(m.group(3)) if m else None
        frs = sorted(f for f, bb in boxes[vid].items() if f % 10 == 0 and any(b[4] in ("lying down", "crawling") for b in bb))
        cap = cv2.VideoCapture(str(mp4)); got = 0
        for f in frs:
            if got >= 30:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for k, (x1, y1, x2, y2, pose) in enumerate(boxes[vid][f]):
                if pose not in ("lying down", "crawling") or max(x2 - x1, y2 - y1) < 16 or got >= 30:
                    continue
                mk, sc = mask_for(proc, model, rgb, (x1, y1, x2, y2))
                p = OUT / "arch" / f"{vid}_{f:05d}_{k}.png"
                r = save(img, mk, (x1, y1, x2, y2), p)
                if r is None:
                    continue
                meta.write(json.dumps({"file": str(p.relative_to(OUT)), "src": "archangel", "pose": pose, "alt": alt, "w": x2 - x1, "h": y2 - y1, "mask_ratio": r, "sam": round(sc, 3)}) + "\n")
                got += 1; n["archangel"] += 1
        print(vid, got, flush=True)
    acts = {x["id"]: x["labels"] for x in json.load(open(NOMAD / "activityLabels.json"))}
    per = defaultdict(int); rows = json.load(open(NOMAD / "annotations.json")); random.Random(0).shuffle(rows)
    for r in rows:
        actor_s, dist_s, f_s = r["file_name"][:-4].split("_")
        a = int(actor_s[5:])
        if a >= 96 or dist_s not in ("a10", "a30") or per[(a, dist_s)] >= 4:
            continue
        act = _nomad_act(acts, a, dist_s[1:], int(f_s[1:]))
        bb = [b for b in r["annotations"] if int(b.get("visibility", 100)) >= 70]
        if act not in ("Laying", "Hiding (Laying)") or len(bb) != 1:
            continue
        img = cv2.imread(str(NOMAD / "images" / actor_s / f"{actor_s}_{dist_s}" / r["file_name"]))
        if img is None:
            continue
        x, y, w, h = bb[0]["bbox"]
        mk, sc = mask_for(proc, model, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (x, y, x + w, y + h))
        p = OUT / "nomad" / f"{r['file_name'][:-4]}.png"
        rr = save(img, mk, (x, y, x + w, y + h), p, pad=12)
        if rr is None:
            continue
        meta.write(json.dumps({"file": str(p.relative_to(OUT)), "src": "nomad", "pose": act, "dist": dist_s, "w": w, "h": h, "mask_ratio": rr, "sam": round(sc, 3)}) + "\n")
        per[(a, dist_s)] += 1; n["nomad"] += 1
    meta.close()
    print("은행", dict(n))


if __name__ == "__main__":
    main()
