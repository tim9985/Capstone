"""
fo_train_set.py — 지금 학습 목록 그대로를 FiftyOne 데이터셋 하나로 띄운다 (시각 점검 · 2026-10-06)

  fo_all_sets.py 는 원천별 + 모델 예측 · 여기는 **학습 목록(configs/lists/train_*.txt) 에 든 장만** · 정답 박스만
  필드: ground_truth (YOLO 라벨 → 상대 좌표) · source (원천 폴더) · n_person · 태그 = 원천 · `음성` (사람 0)
  저장 뷰 (이름은 영문만 됨): 원천마다 `src_<원천>` · `negative` · `small_person` (가장 작은 박스 짧은 변 < 1 %)

실행: python fo_train_set.py [목록=configs/lists/train_v10_s31.txt] [데이터셋 이름=train_v10]
앱: 이미 떠 있는 FiftyOne (포트 5151) 에서 데이터셋 고르기
"""
import re
import sys
from pathlib import Path

import fiftyone as fo
from fiftyone import ViewField as F

BASE = Path(__file__).resolve().parent


def source_of(path):
    m = re.search(r"/data/(.+?)/images/", path)
    return m.group(1).replace("det_v6/", "").replace("det_fov/", "fov_").replace("/", "_") if m else "?"


def main():
    lst = Path(sys.argv[1] if len(sys.argv) > 1 else BASE / "configs/lists/train_v10_s31.txt")
    name = sys.argv[2] if len(sys.argv) > 2 else "train_v10"
    files = [l.strip() for l in open(lst) if l.strip() and not l.startswith("#")]
    if name in fo.list_datasets():
        fo.delete_dataset(name)
    ds = fo.Dataset(name, persistent=True)
    samples = []
    for f in files:
        lab = Path(re.sub(r"/images/", "/labels/", f)).with_suffix(".txt")
        dets = []
        if lab.exists():
            for row in open(lab):
                p = row.split()
                if len(p) >= 5:
                    cx, cy, w, h = map(float, p[1:5])
                    dets.append(fo.Detection(label="person", bounding_box=[cx - w / 2, cy - h / 2, w, h]))
        src = source_of(f)
        tags = [src] + (["음성"] if not dets else [])
        samples.append(fo.Sample(filepath=f, ground_truth=fo.Detections(detections=dets), source=src,
                                 n_person=len(dets), tags=tags))
    ds.add_samples(samples)
    ds.info = {"목록": str(lst), "장수": len(files)}
    ds.save()
    for src in ds.distinct("source"):
        ds.save_view(f"src_{src}", ds.match(F("source") == src), overwrite=True)
    ds.save_view("negative", ds.match(F("n_person") == 0), overwrite=True)
    small = F("bounding_box")[2].min(F("bounding_box")[3]) < 0.01
    ds.save_view("small_person", ds.filter_labels("ground_truth", small), overwrite=True)
    print(f"{name}: {len(ds):,} 장 · 사람 {sum(ds.values('n_person')):,} · 음성 {len(ds.match(F('n_person') == 0)):,}")
    for src, n in sorted(ds.count_values("source").items(), key=lambda t: -t[1]):
        v = ds.match(F("source") == src)
        print(f"  {src:12s} {n:6,} 장 · 사람 {sum(v.values('n_person')):6,}")


if __name__ == "__main__":
    main()
