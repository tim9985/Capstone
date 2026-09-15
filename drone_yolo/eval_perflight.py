"""
eval_perflight.py — 검증셋을 WiSARD 비행별 · NOMAD 거리별로 쪼개 따로 잰다 (학습 없음)

왜 필요한가 (2026-09-14 M1 분석)
  data_all val 의 mAP50 0.65 는 비행마다 0.24 ~ 0.94 로 크게 갈린 값의 평균이다.
  어느 비행이 점수를 끌어내리는지 봐야 설정으로 풀지 데이터로 풀지 정할 수 있다.
  · WiSARD 음성 크롭(_n.jpg)은 오탐 계산에 들어가야 하므로 같은 비행에 합친다
    (합치지 않으면 음성 한 장이 그룹 하나가 된다)
  · split=time — 같은 비행의 앞부분이 train 에 있는 시간 분할 비행. 점수가 부풀려진다
  · 박스가 없는 그룹(음성만 있는 비행)은 mAP 를 정의할 수 없어 빈칸으로 둔다

실행:
  python eval_perflight.py --weights runs_person/m1_11m_1280/weights/best.pt \
      --out metrics/eval_perflight_m1_11m_1280.csv
"""
import argparse
import collections
import csv
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

BASE_DIR = Path(__file__).resolve().parent
DET = BASE_DIR / "data" / "det"
NOMAD_BATCHES = ("nomad_actor01_10", "nomad_actor11_20", "nomad_actor21_30", "nomad_actor_sel31_100")


def flight_of(name):
    """'DJI_0407.mp4_00406_c0.jpg' · '..._00078_n.jpg' → 'DJI_0407.mp4'"""
    return re.sub(r"_\d+_(c\d+|n)\.jpg$", "", name)


def build_groups():
    groups = collections.defaultdict(list)
    for p in sorted((DET / "wisard" / "images" / "val").glob("*.jpg")):
        groups["wisard:" + flight_of(p.name)].append(p)
    for b in NOMAD_BATCHES:
        for p in sorted((DET / b / "images" / "val").glob("*.jpg")):
            groups["nomad:" + p.name.split("_")[1]].append(p)   # Actor004_a10_f0001_c0.jpg → a10
    return groups


def n_boxes(img):
    t = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
    return sum(1 for ln in t.read_text().splitlines() if ln.strip()) if t.exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True, help="결과 CSV (예: metrics/eval_perflight_<실행>.csv)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from ultralytics import YOLO

    w = Path(args.weights)
    model_name = w.parent.parent.name if w.parent.name == "weights" else w.stem
    train_flights = {flight_of(p.name) for p in (DET / "wisard" / "images" / "train").glob("*.jpg")}
    model = YOLO(str(w))

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for g, files in sorted(build_groups().items()):
            kind, key = g.split(":", 1)
            split = "actor" if kind == "nomad" else ("time" if key in train_flights else "flight")
            boxes = sum(n_boxes(f) for f in files)
            if boxes == 0:
                rows.append([model_name, g, split, len(files), 0, "", "", "", ""])
                continue
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", g)
            lst = Path(tmp) / f"{safe}.txt"
            lst.write_text("\n".join(map(str, files)) + "\n")
            yml = Path(tmp) / f"{safe}.yaml"
            yml.write_text(f"train: {lst}\nval: {lst}\nnc: 1\nnames: ['person']\n")
            r = model.val(data=str(yml), imgsz=args.imgsz, batch=args.batch, plots=False,
                          verbose=False, project=tmp, name=safe, exist_ok=True)
            rows.append([model_name, g, split, len(files), boxes,
                         f"{r.box.map50:.4f}", f"{r.box.map:.4f}", f"{r.box.mp:.4f}", f"{r.box.mr:.4f}"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "group", "split", "images", "boxes", "mAP50", "mAP50_95", "precision", "recall"])
        wr.writerows(rows)

    print(f"\n{'그룹':<52} {'분할':<6} {'장수':>5} {'박스':>5} {'mAP50':>6} {'50-95':>6} {'R':>6}")
    for m, g, s, n, b, m50, m95, p, rr in rows:
        print(f"{g:<52} {s:<6} {n:>5} {b:>5} {m50 or '-':>6} {m95 or '-':>6} {rr or '-':>6}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
