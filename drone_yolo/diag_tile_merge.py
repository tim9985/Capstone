"""
diag_tile_merge.py — 타일 합치기 규칙 비교 (10-11 · v7 넘기 검토 · 추론 쪽)

  문제  1920×1080 → 1280×720 타일 4장 (겹침 50 %) → 사람이 타일 안쪽 경계에 걸리면 한 타일은 잘린 박스, 다른 타일은 온전한 박스
        → 둘의 IoU 가 0.6 미만 (잘린 박스는 넓이 0.4 배) 이라 NMS 0.6 이 못 지운다 → 확신 높은 오탐
        (soup_v7r2 확신 ≥ 0.5 의 「위치 어긋남 · 정답은 이미 짝」 오탐 test_kr 51건 중 50 · test_v2 57 중 46 이 안쪽 경계에 닿음)
  규칙  base   = 지금 (타일 안 NMS 0.7 → 전체 NMS 0.6)
        own    = 중심 소유 — 박스 중심이 그 타일의 「소유 구역」 (겹침 한가운데 x 960 · y 540 로 나눈 사분면) 밖이면 버림 → NMS 0.6
                 매개변수 없음 · 소유 타일은 그 사람을 온전히 본다 (사람 < 640×360 px)
        edge   = 안쪽 경계에 닿은 (≤ 3 px) 박스가 다른 타일의 더 큰 박스 안에 60 % 넘게 들어가면 버림 · 점수는 큰 박스로 넘김 (max)
  고르기 = soup_v7r2 가 학습하지 않은 1920×1080 셋 (test_ladd · test_nii · test_cy) → 정한 규칙 하나를 세 평가셋에 한 번
실행: /home/se/miniconda3/envs/drone/bin/python diag_tile_merge.py --sets test_ladd,test_nii,test_cy [--model soup_v7r2]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_dets import ap_at, load, nms
from eval_test_v2 import TILES, iou_mat
from tile_merge import edge_merge

BASE = Path(__file__).resolve().parent


def merge(b, s, t, W, H, rule):
    if rule == "own" and (W, H) == (1920, 1080):
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        tx = np.array([TILES[k][0] for k in t]); ty = np.array([TILES[k][1] for k in t])
        keep = ((cx < W / 2) == (tx == 0)) & ((cy < H / 2) == (ty == 0))
        b, s, t = b[keep], s[keep], t[keep]
    k = nms(b, s, 0.6); b, s, t = b[k], s[k].copy(), t[k]
    if rule == "edge":
        b, s = edge_merge(b, s, t, W, H)
    return b, s


def run(model, tag, rule):
    d = load(model, tag); gts = list(d["gt"]); wh = d["wh"]
    img, box, sc, tl = d["img"], d["box"], d["score"], d["tile"]
    order = np.argsort(img, kind="stable"); img, box, sc, tl = img[order], box[order], sc[order], tl[order]
    dets = {}
    for seg in np.split(np.arange(len(img)), np.flatnonzero(np.diff(img)) + 1):
        if len(seg):
            i = int(img[seg[0]]); dets[i] = merge(box[seg], sc[seg], tl[seg], *wh[i], rule)
    for i in range(len(gts)):
        dets.setdefault(i, (np.zeros((0, 4), np.float32), np.zeros(0, np.float32)))
    return dets, gts, [str(n) for n in d["names"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="soup_v7r2")
    ap.add_argument("--sets", default="test_ladd,test_nii,test_cy")
    ap.add_argument("--rules", default="base,own,edge")
    ap.add_argument("--save", action="store_true", help="eval_<tag>.npz 꼴로 저장 → runs_person/<model>_<rule>/ (compare_ci 짝 비교)")
    a = ap.parse_args()
    res = {}
    for tag in a.sets.split(","):
        R = {}
        for rule in a.rules.split(","):
            dets, gts, names = run(a.model, tag, rule)
            R[rule] = [ap_at(dets, gts, t) for t in (0.5, 0.3)]
            if a.save:
                S, T, I = [], [], []
                for i, gt in enumerate(gts):
                    b, s = dets[i]; o = np.argsort(-s); b, s = b[o], s[o]
                    M = iou_mat(gt, b) if len(gt) else np.zeros((0, len(b))); used = np.zeros(len(gt), bool)
                    for j in range(len(b)):
                        k = -1
                        if len(gt):
                            c = np.where((M[:, j] >= 0.5) & ~used)[0]
                            if len(c):
                                k = c[np.argmax(M[c, j])]; used[k] = True
                        S.append(float(s[j])); T.append(k >= 0); I.append(i)
                out = BASE / "runs_person" / f"{a.model}_{rule}"; out.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out / f"eval_{tag}.npz", s=np.array(S), t=np.array(T, bool), img=np.array(I, int),
                                    ngt=np.array([len(g) for g in gts], float), names=np.array(names))
        res[tag] = R; print(tag, "AP50 · AP30", json.dumps(R), flush=True)
    (BASE / "metrics" / f"diag_tile_merge_{a.model}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
