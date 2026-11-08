"""
robust_eval.py — 조건별 강건성 (조도 · 가림 · 자세 · 시점) 우리 vs 기성 · 같은 채점 (11-08 판넬) → metrics/robust_conditions.json

  원자료  runs_fair/<셋>/<모델@방식>.npz (fair_eval · 모든 모델 같은 그림 · 같은 정답) — 다시 추론 안 함
  모델    우리 v12 (운용 4조각) · 기성 = 셋마다 방식 (4조각 · 640 · 통째) 중 그 셋 전체 AP50 최고 (판넬 표와 같은 규칙 · 기성에 유리)
  조건    조도  VisDrone 밤 · 어두운 장면 (그림 평균 밝기 < 80) vs 밝은 장면 · Okutama 아침 (x.1.x) vs 한낮 (x.2.x)
          가림  NOMAD 박스 가시도 (원본 annotations visibility 10~100 %) — 다 보임 100 · 일부 가림 50~90 · 많이 가림 ≤ 40 · Okutama 원본 라벨 가림 표시
          자세  Okutama 원본 행동 라벨 — 서기 · 걷기/뛰기 · 앉기 · 눕기 · NOMAD 프레임 행동 — 걷기 · 숨기 · 눕기 (숨어 누움 포함)
          시점  UAV-SAR 파일 이름 — 45° (비스듬) vs 90° (수직) · 고도 15 m vs 30 m
  채점    **재현율** = 그 조건 사람 중 찾은 비율 (IoU ≥ 0.5) · 모델마다 그 데이터셋에서 F1 최고 문턱 (판넬 B 와 같은 규칙 · fair_metrics best_thr)
          ⚠ AP 로 안 함 — 소수 조건 (누움 98명) 은 AP 분모를 남의 헛탐지가 채워 조건 차이가 아니라 전체 헛탐지를 잼 (11-08 첫 판에서 확인)
          판정  우리 − 상대 차의 블록 부트스트랩 5,000회 (조건마다 시드 고정) (연속 20장 묶음) 95 % 구간 · 0 을 넘으면 「앞섬」 · 0 을 품으면 「비슷」 · 사람 < 40 은 「참고」
                상대 둘 — 실시간 기성 최고 (COCO · AeroNova · VisDrone v9e · YOLO-World · YOLO11x · RT-DETR-x 중) · Grounding DINO (대형 · 1080p 355 ms)
실행  /home/se/miniconda3/envs/drone/bin/python robust_eval.py
"""
import json, zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from eval_external_det import ap101, iou_mat

BASE = Path(__file__).resolve().parent
RUN = BASE / "runs_fair"
RAW = BASE / "data" / "raw"
M = json.load(open(BASE / "metrics" / "fair_metrics.json"))
MODELS = {"ours": "v12_f91_swa", "coco": "coco_yolo11m", "aeronova": "aeronova", "v9e": "visdrone_yolov9e", "gdino": "sota_gdino",
          "yworld": "sota_yoloworld", "y11x": "sota_yolo11x", "rtdetr": "sota_rtdetrx"}


def tag_of(model, s):
    if model == "v12_f91_swa":
        return "v12_f91_swa@tile1280"
    c = {t: v for t, v in M[s].items() if t.split("@")[0] == model}
    return max(c, key=lambda t: c[t]["AP50"])


def hits(s, tag, thr, keep_img, cat):
    """그림마다 (이 조건 정답 수, 찾은 수) — 확신 ≥ thr · 점수 순 탐욕 짝 IoU 0.5"""
    g = np.load(RUN / s / "gt.npz"); d = np.load(RUN / s / f"{tag}.npz"); out = {}
    for i in range(len(g["names"])):
        if not keep_img(i):
            continue
        gt = g["gt"][g["gt_img"] == i]; c = cat(i, gt); m = (d["img"] == i) & (d["conf"] >= thr); b, sc = d["box"][m], d["conf"][m]
        o = np.argsort(-sc); b = b[o]; used = np.zeros(len(gt), bool)
        M_ = iou_mat(gt, b) if len(gt) and len(b) else np.zeros((len(gt), len(b)))
        for j in range(len(b)):
            cand = np.where((M_[:, j] >= 0.5) & ~used)[0]
            if len(cand):
                used[cand[np.argmax(M_[cand, j])]] = True
        if c.sum():
            out[i] = (int(c.sum()), int((used & c).sum()))
    return out


def blocks(idx, names):
    grp = defaultdict(list)
    for i in idx:
        grp[str(names[i]).rsplit("_", 1)[0]].append(i)
    return [g[k:k + 20] for g in grp.values() for k in range(0, len(g), 20)]


def ap_subset(s, tag, keep_img, cat):
    """keep_img: 그림 번호 → 쓸지 · cat: 그림 번호 → 정답마다 True (이 조건) / False (무시)"""
    g = np.load(RUN / s / "gt.npz"); d = np.load(RUN / s / f"{tag}.npz"); S, T, n = [], [], 0
    for i in range(len(g["names"])):
        if not keep_img(i):
            continue
        gt = g["gt"][g["gt_img"] == i]; c = cat(i, gt); m = d["img"] == i; b, sc = d["box"][m], d["conf"][m]
        o = np.argsort(-sc); b, sc = b[o], sc[o]; n += int(c.sum())
        M_ = iou_mat(gt, b) if len(gt) else np.zeros((0, len(b))); used = np.zeros(len(gt), bool)
        for j in range(len(b)):
            k = -1
            if len(gt):
                cand = np.where((M_[:, j] >= 0.5) & ~used)[0]
                if len(cand):
                    k = cand[np.argmax(M_[cand, j])]; used[k] = True
            if k >= 0 and not c[k]:
                continue                                                        # 다른 조건 정답 → 무시
            S.append(sc[j]); T.append(k >= 0)
    return (round(float(ap101(np.array(S), np.array(T, bool), n)), 4) if n else None), n


# ── 조건 만들기 ──
def okutama_meta():
    """test_obl 정답 상자마다 (행동, 가림) — 원본 4K 라벨을 1280×720 으로 줄여 IoU 짝"""
    g = np.load(RUN / "test_obl" / "gt.npz"); raw = {}
    out = {}
    for i, nm in enumerate(g["names"]):
        _, seq, fr = nm[:-4].split("_"); fr = int(fr)
        if seq not in raw:
            f = sorted((RAW / "okutama").glob(f"**/Labels/SingleActionLabels/3840x2160/{seq}.txt"), key=lambda p: "TrainSetVideos" in str(p))[0]
            per = defaultdict(list)
            for ln in open(f, encoding="utf-8", errors="replace"):
                t = ln.split()
                if len(t) >= 10 and t[6] == "0":
                    per[int(t[5])].append(([int(t[k]) / 3 for k in (1, 2, 3, 4)], int(t[7]), t[-1].strip('"')))
            raw[seq] = per
        gt = g["gt"][g["gt_img"] == i]; cand = raw[seq].get(fr, [])
        lab = []
        for b in gt:
            if cand:
                ious = iou_mat(b[None], np.array([c[0] for c in cand], np.float32))[0]; k = int(np.argmax(ious))
                lab.append((cand[k][2], cand[k][1]) if ious[k] > 0.7 else (None, None))
            else:
                lab.append((None, None))
        out[i] = lab
    return out


def nomad_meta():
    """test_nomad 그림마다 (가시도, 행동) — make_testsets_ext.nomad() 와 같은 순서로 원본 짝"""
    ann = json.load(open(RAW / "NOMAD" / "annotations.json")); acts = {x["id"]: x["labels"] for x in json.load(open(RAW / "NOMAD" / "activityLabels.json"))}
    items = []
    for r in ann:
        actor, dist, f = r["file_name"][:-4].split("_")
        if int(actor[5:]) < 96 or not (RAW / "NOMAD/images" / actor / f"{actor}_{dist}" / r["file_name"]).exists():
            continue
        fr = int(f[1:]); act = None
        for a, spans in acts.get(int(actor[5:]), {}).get(dist[1:], {}).items():
            if any(int(lo) <= fr <= int(hi) for lo, hi in spans):
                act = a
        items.append((f"nomad_{actor}{dist}_{len(items):04d}.jpg", [int(b.get("visibility", 100)) for b in r["annotations"]], act))
    by = {n: (v, a) for n, v, a in items}; g = np.load(RUN / "test_nomad" / "gt.npz")
    return {i: by.get(str(nm)) for i, nm in enumerate(g["names"])}


def main():
    res = defaultdict(dict)

    rng = np.random.default_rng(0)

    def run(group, label, s, keep_img, cat, note=""):
        row, H = {}, {}
        names = np.load(RUN / s / "gt.npz")["names"]
        for k, mdl in MODELS.items():
            t = tag_of(mdl, s); H[k] = hits(s, t, M[s][t]["best_thr"], keep_img, cat)
            n = sum(a for a, _ in H[k].values()); row[k] = round(sum(b for _, b in H[k].values()) / n, 4) if n else None
        row["n"] = n; row["셋"] = s; row["설명"] = note
        RT = ("coco", "aeronova", "v9e", "yworld", "y11x", "rtdetr")             # 실시간 가능 (YOLO · RT-DETR 급) — GDINO 는 1080p 한 장 355 ms 라 따로
        best = max(RT, key=lambda k: row[k] or 0); row["실시간 기성 최고"] = best
        bl = blocks(sorted(H["ours"]), names)
        for key, rival in (("실시간", best), ("GDINO", "gdino")):
            rng = np.random.default_rng(zlib.crc32(f"{group}|{label}|{key}".encode())); diffs = []   # 조건마다 시드 고정
            for _ in range(5000):
                pick = [i for b in rng.integers(0, len(bl), len(bl)) for i in bl[b]]
                no = sum(H["ours"][i][0] for i in pick)
                if no:
                    diffs.append((sum(H["ours"][i][1] for i in pick) - sum(H[rival][i][1] for i in pick)) / no)
            lo, hi = np.percentile(diffs, [2.5, 97.5]); row[f"차 구간 ({key})"] = [round(float(lo), 4), round(float(hi), 4)]
            row[f"판정 ({key})"] = "참고" if n < 40 else ("앞섬" if lo > 0 else ("뒤짐" if hi < 0 else "비슷"))
        res[group][label] = row
        print(group, label, n, row["ours"], best, row[best], row["판정 (실시간)"], "| GDINO", row["gdino"], row["판정 (GDINO)"], flush=True)

    allc = lambda i, gt: np.ones(len(gt), bool)
    # 조도
    g = np.load(RUN / "test_visdrone" / "gt.npz")
    br = np.array([cv2.imread(str(BASE / "data/test_visdrone/images" / n), cv2.IMREAD_GRAYSCALE).mean() for n in g["names"]])
    run("조도", "밤 · 어두운 장면", "test_visdrone", lambda i: br[i] < 80, allc, "VisDrone 그림 평균 밝기 < 80 (0~255)")
    run("조도", "낮 · 밝은 장면", "test_visdrone", lambda i: br[i] >= 80, allc, "VisDrone 밝기 ≥ 80")
    g = np.load(RUN / "test_obl" / "gt.npz"); seq = [n.split("_")[1] for n in g["names"]]
    run("조도", "아침 (낮은 해 · 긴 그림자)", "test_obl", lambda i: seq[i].split(".")[1] == "1", allc, "Okutama x.1.x")
    run("조도", "한낮 (강한 햇빛)", "test_obl", lambda i: seq[i].split(".")[1] == "2", allc, "Okutama x.2.x")
    # 가림 · 자세 (NOMAD)
    nm = nomad_meta()
    vis = lambda lo, hi: (lambda i, gt: np.array([lo <= v <= hi for v in (nm[i][0] if nm[i] else [100] * len(gt))][:len(gt)] + [False] * max(0, len(gt) - len(nm[i][0] if nm[i] else [])), bool))
    run("가림", "다 보임 (100 %)", "test_nomad", lambda i: True, vis(100, 100), "NOMAD 박스 가시도")
    run("가림", "일부 가림 (50~90 %)", "test_nomad", lambda i: True, vis(50, 90), "NOMAD")
    run("가림", "많이 가림 (40 % 이하)", "test_nomad", lambda i: True, vis(0, 40), "NOMAD · 수풀 · 나무 뒤")
    ok = okutama_meta()
    run("가림", "가려진 사람 (Okutama)", "test_obl", lambda i: True, lambda i, gt: np.array([o == 1 for _, o in ok[i]], bool), "Okutama 원본 라벨 가림 표시")
    act = lambda names: (lambda i, gt: np.array([a in names for a, _ in ok[i]], bool))
    run("자세", "서 있음", "test_obl", lambda i: True, act({"Standing"}), "Okutama 행동 라벨")
    run("자세", "걷기 · 뛰기", "test_obl", lambda i: True, act({"Walking", "Running"}), "Okutama")
    run("자세", "앉음", "test_obl", lambda i: True, act({"Sitting"}), "Okutama")
    run("자세", "누움", "test_obl", lambda i: True, act({"Lying"}), "Okutama")
    na = lambda names: (lambda i, gt: np.array([bool(nm[i]) and nm[i][1] in names] * len(gt), bool))
    run("자세", "숨기 (웅크림 · 엎드림)", "test_nomad", lambda i: True, na({"Hiding"}), "NOMAD 행동 Hiding")
    run("자세", "누움 (NOMAD)", "test_nomad", lambda i: True, na({"Laying", "Hiding (Laying)"}), "NOMAD Laying · 숨어 누움")
    # 시점
    g = np.load(RUN / "test_sar" / "gt.npz"); cond = [n.split("_")[1] for n in g["names"]]
    run("시점", "비스듬 45°", "test_sar", lambda i: cond[i].endswith("45d"), allc, "UAV-SAR")
    run("시점", "수직 90°", "test_sar", lambda i: cond[i].endswith("90d"), allc, "UAV-SAR")
    run("시점", "고도 15 m", "test_sar", lambda i: cond[i].startswith("15m"), allc, "UAV-SAR")
    run("시점", "고도 30 m (사람 작음)", "test_sar", lambda i: cond[i].startswith("30m"), allc, "UAV-SAR")
    out = {"모델": MODELS, "방식": {k: {s: tag_of(m, s) for s in ("test_visdrone", "test_obl", "test_nomad", "test_sar")} for k, m in MODELS.items()}, "조건": res}
    json.dump(out, open(BASE / "metrics" / "robust_conditions.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
