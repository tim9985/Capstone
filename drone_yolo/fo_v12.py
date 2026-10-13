"""
fo_v12.py — v12 학습 · val 창 전부를 FiftyOne 에 (`train_v12` · 10-13 · 사용자 「무엇을 썼는지 51 에서 확인」)

  필드: ground_truth (YOLO → 상대 좌표) · source (출처) · domain (영역) · split · n_person · in_f91 · in_f92 (반복 목록 포함 여부)
  태그: 출처 · 영역 · split · `음성` · `반복빠짐` (95 % 뽑기에서 빠진 장)
  저장 뷰: src_<출처> · dom_<영역> · negative · val · small_person (긴 변 < 16 px)
  ⚠ 연구용 · 약관 제한 데이터 — 서버 안 팀 열람용 · 51 화면을 팀 밖에 공유하지 않는다
실행: /home/se/miniconda3/envs/drone/bin/python fo_v12.py
"""
from pathlib import Path

import fiftyone as fo
from fiftyone import ViewField as F

BASE = Path(__file__).resolve().parent
L = BASE / "configs" / "lists"
DOM = {"visdrone_obl": "obl_urban", "unicamp": "obl_urban", "niicu": "obl_urban", "nomad": "obl_nat", "sard": "obl_nat",
       "archangel": "obl_nat", "visdrone_nadir": "nad_urban", "wisard": "nad_nat", "heridal": "nad_nat", "ladd": "nad_nat", "aihub": "nad_kr"}


def src_of(p):
    n = Path(p).name
    return "visdrone_obl" if n.startswith("visdrone_obl") else "visdrone_nadir" if n.startswith("visdrone_nadir") else n.split("_")[0]


def main():
    tr = [l.strip() for l in open(L / "train_v12.txt") if l.strip()]
    va = [l.strip() for l in open(L / "val_v12.txt") if l.strip()]
    f91 = {l.strip() for l in open(L / "train_v12_f91.txt")}; f92 = {l.strip() for l in open(L / "train_v12_f92.txt")}
    name = "train_v12"
    if name in fo.list_datasets():
        fo.delete_dataset(name)
    ds = fo.Dataset(name, persistent=True); S = []
    for split, files in (("train", tr), ("val", va)):
        for f in files:
            dets = []
            for row in open(f.replace("/images/", "/labels/")[:-4] + ".txt"):
                p = row.split()
                if len(p) == 5:
                    cx, cy, w, h = map(float, p[1:])
                    dets.append(fo.Detection(label="person", bounding_box=[cx - w / 2, cy - h / 2, w, h], px=round(max(w * 1280, h * 720), 1)))
            s = src_of(f); d = DOM.get(s, "?")
            tags = [s, d, split] + (["음성"] if not dets else []) + (["반복빠짐"] if split == "train" and not (f in f91 and f in f92) else [])
            S.append(fo.Sample(filepath=f, ground_truth=fo.Detections(detections=dets), source=s, domain=d, split=split,
                               n_person=len(dets), in_f91=f in f91, in_f92=f in f92, tags=tags,
                               metadata=fo.ImageMetadata(width=1280, height=720)))
    ds.add_samples(S)
    ds.info = {"설명": "v12 — 원본부터 다시 짠 학습 창 (make_v12_data · make_v12_neg) · 볼트 「17 v12 데이터 재구성」", "학습": len(tr), "val": len(va)}
    ds.save()
    for s in ds.distinct("source"):
        ds.save_view(f"src_{s}", ds.match(F("source") == s), overwrite=True)
    for d in ds.distinct("domain"):
        ds.save_view(f"dom_{d}", ds.match(F("domain") == d), overwrite=True)
    ds.save_view("negative", ds.match(F("n_person") == 0), overwrite=True)
    ds.save_view("val", ds.match(F("split") == "val"), overwrite=True)
    ds.save_view("small_person", ds.filter_labels("ground_truth", F("px") < 16), overwrite=True)
    print(f"{name}: {len(ds):,} 장 (학습 {len(tr):,} · val {len(va):,}) · 사람 {sum(ds.values('n_person')):,}")
    for s, n in sorted(ds.count_values("source").items(), key=lambda t: -t[1]):
        print(f"  {s:15s} {n:6,}")


if __name__ == "__main__":
    main()
