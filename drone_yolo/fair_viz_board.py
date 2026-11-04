"""
fair_viz_board.py — metrics/fair_viz.json + profile_worker.json + 상의 색 결정 노트 수치 → 페어 보드 데이터 파일 (JSON 행)
실행  python3 fair_viz_board.py <출력 폴더>
"""
import json, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
D = json.load(open(BASE / "metrics" / "fair_viz.json"))
P = json.load(open(BASE / "metrics" / "profile_worker.json"))
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
EXTN = {"aeronova": "AeroNova", "visdrone_yolo11x": "VisDrone YOLO11x", "visdrone_yolo26x": "VisDrone YOLO26x", "ap_mix_26l": "AerialPerson 혼합",
        "ap_heridal_26l": "HERIDAL 전용", "ap_sard_26l": "SARD 전용", "ap_ladd_26l": "LADD 전용"}
det, pr, thr = [], [], []
H = {}
for s, r in D["det"].items():
    if s == "test_auair":
        continue
    for mod in ("ours", "ext"):
        x = r[mod]; a = x["at020"]
        det.append({"set": s, "label": r["label"], "model": mod, "name": "우리 v12" if mod == "ours" else EXTN.get(x["name"], x["name"]),
                    "mode": x.get("mode", "tile"), "imgsz": x.get("imgsz", 1280), "AP50": x["AP50"], "R": a["R"], "P": a["P"],
                    "TP": a["TP"], "FP": a["FP"], "FN": a["FN"], "n_gt": x["n_gt"], "imgs": x["imgs"]})
        for i, p in enumerate(x["pr"]):
            rc = i / 100
            if rc <= x["max_recall"] + 1e-9:
                pr.append({"set": s, "model": mod, "recall": rc, "precision": p})
        pr.append({"set": s, "model": mod, "recall": x["max_recall"], "precision": x["pr"][min(100, int(x["max_recall"] * 100))]})
        for t, v in x["thr"].items():
            thr.append({"set": s, "model": mod, "thr": float(t), "R": v["R"], "P": v["P"]})
        for b, v in x["by_height"].items():
            if v["R"] is not None:
                h = H.setdefault((b, mod), [0, 0.0]); h[0] += v["n"]; h[1] += v["R"] * v["n"]
LB = {"0-24": "24 px 미만", "24-48": "24~48 px", "48-96": "48~96 px", "96-9999": "96 px 이상"}
height = [{"bin": b, "label": LB[b], "model": m, "n": n, "R": round(s / n, 4)} for (b, m), (n, s) in H.items()]
height.sort(key=lambda r: (list(LB).index(r["bin"]), r["model"]))
road = [{"model": "ours", "label": "우리 v12", "fp": D["det"]["test_auair"]["ours"]["FP_neg"]}] + \
       [{"model": k, "label": EXTN.get(k, k), "fp": v} for k, v in D["road_ext"].items()]
post = [{"set": s, "name": v["name"], "n": v["n"], "n_lying": v["n_lying"], "AUROC": v["AUROC"]} for s, v in D["posture"].items()]
roc = [{"set": s, "fpr": a, "tpr": b} for s, v in D["posture"].items() for a, b in v["roc"]]
pcm = [{"set": s, "actual": v["classes"][i], "pred": v["classes"][j], "n": v["confusion"][i][j]} for s, v in D["posture"].items() for i in range(3) for j in range(3)]
# 상의 색 — 결정 노트 「상의 색 SigLIP 2 제로샷」 표 (UE 는 잔디 문장 최종판)
color = [{"label": l, "role": ro, "rule": a, "siglip": b} for l, ro, a, b in [
    ("드론 UAV-Human", "고를 때 씀", 0.286, 0.512), ("드론 NOMAD", "고를 때 씀", 0.336, 0.422), ("지상 CCTV PETA", "처음 봄", 0.270, 0.560),
    ("디지털관 사진", "처음 봄 · 잠정 라벨", 0.314, 0.744), ("디지털관 영상", "처음 봄 · 잠정 라벨", 0.422, 0.885), ("시뮬레이션 20 m", "확인", 0.393, 0.536)]]
S = P["stages"]
lat = [{"stage": "영상 풀기", "label": "영상 풀기", "median": S["읽기 (JPEG 풀기)"]["median_ms"], "p95": S["읽기 (JPEG 풀기)"]["p95_ms"], "note": "시험은 JPEG · 실영상은 H.264"},
       {"stage": "탐지", "label": "사람 찾기 (4조각 · TensorRT)", "median": S["탐지"]["median_ms"], "p95": S["탐지"]["p95_ms"], "note": ""},
       {"stage": "추적 + 상태", "label": "추적 · 자세 · 멈춤 · 등급", "median": S["추적 + 상태"]["median_ms"], "p95": S["추적 + 상태"]["p95_ms"], "note": "1초에 한 번 드론 움직임 빼기"},
       {"stage": "좌표", "label": "위치 (위경도)", "median": S["좌표 (사람 전부)"]["median_ms"], "p95": S["좌표 (사람 전부)"]["p95_ms"], "note": ""},
       {"stage": "상의 색", "label": "상의 색 SigLIP 2 · 1명", "median": P["siglip_per_person"]["median_ms"], "p95": P["siglip_per_person"]["p95_ms"], "note": "후보마다 0.5초에 한 번"}]
M = P["memory"]
mem = [{"stage": a, "gpu_gb": round(M[k]["gpu_mb"] / 1024, 2), "ram_gb": round(M[k]["rss_mb"] / 1024, 2)} for a, k in
       [("탐지 모델 올림", "+탐지 (TensorRT)"), ("실제로 돌린 뒤", "배포 판 다 돈 뒤"), ("+ 상의 색 SigLIP 2", "끝")]]
for n, rows in [("det", det), ("pr", pr), ("thr", thr), ("height", height), ("road", road), ("posture", post), ("roc", roc), ("pcm", pcm),
                ("color", color), ("latency", lat), ("memory", mem)]:
    json.dump(rows, open(out / f"{n}.json", "w"), ensure_ascii=False)
    print(n, len(rows), (out / f"{n}.json").stat().st_size)
