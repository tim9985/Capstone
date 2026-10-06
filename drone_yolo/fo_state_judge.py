"""
fo_state_judge.py — 상태 판단 모델 비교 (state_judge_eval.py) 를 FiftyOne 으로 본다 (2026-10-06)

  데이터셋 `state_judge` — 시험 1 크롭 210장 (Okutama 비스듬 120 · AI-Hub 수직 90 · 정답 박스 크롭)
  필드: gt (정답 자세) · 방법마다 <방법>_pose (예측 자세 · 확신도) · <방법>_help (도움 필요 확률 0~1)
        · <방법>_wrong (자세 틀림 여부) — 방법 = metrics/state_judge_<방법>_items.json 이 있는 것 전부
  저장 뷰: <방법>_wrong (자세 틀린 장) · <방법>_miss_lying (누운 사람인데 도움 필요 < 0.5) · <방법>_false_help (안 누웠는데 ≥ 0.5)
  태그: 평가셋 (okutama_obl · aihub_test)

실행: /home/se/miniconda3/envs/drone/bin/python fo_state_judge.py   (방법 결과가 늘면 다시 실행 — 데이터셋을 새로 만든다)
"""
import json
from pathlib import Path

import fiftyone as fo
from fiftyone import ViewField as F

BASE = Path(__file__).resolve().parent
CLASSES = ("lying", "sitting", "standing")
NAMES = {"baseline": "ours", "qwen3vl": "qwen3vl", "clef_flash": "clef_flash", "clef": "clef27b"}


def main():
    res = {m: json.load(open(BASE / "metrics" / f"state_judge_{m}_items.json")) for m in NAMES
           if (BASE / "metrics" / f"state_judge_{m}_items.json").exists()}
    first = next(iter(res.values()))
    if "state_judge" in fo.list_datasets():
        fo.delete_dataset("state_judge")
    ds = fo.Dataset("state_judge", persistent=True)
    by = {m: {r["file"]: r for r in rows} for m, rows in res.items()}
    samples = []
    for r in first:
        s = fo.Sample(filepath=r["file"], tags=[r["set"]], gt=fo.Classification(label=r["pose"]), eval_set=r["set"])
        for m, d in by.items():
            x = d.get(r["file"])
            if x is None:
                continue
            k = max(range(3), key=lambda i: x["p"][i])
            n = NAMES[m]
            s[f"{n}_pose"] = fo.Classification(label=CLASSES[k], confidence=x["p"][k])
            s[f"{n}_help"] = x["help"]
            s[f"{n}_wrong"] = CLASSES[k] != r["pose"]
        samples.append(s)
    ds.add_samples(samples)
    for m in by:
        n = NAMES[m]
        ds.save_view(f"{n}_wrong", ds.match(F(f"{n}_wrong") == True), overwrite=True)  # noqa: E712
        ds.save_view(f"{n}_miss_lying", ds.match((F("gt.label") == "lying") & (F(f"{n}_help") < 0.5)), overwrite=True)
        ds.save_view(f"{n}_false_help", ds.match((F("gt.label") != "lying") & (F(f"{n}_help") >= 0.5)), overwrite=True)
    ds.info = {"설명": "상태 판단 모델 비교 · 시험 1 크롭 · ours = 박스 모양 (비스듬) / DINOv2 (수직) · help = 도움 필요 확률 (ours 는 P(누움))"}
    ds.save()
    print(f"state_judge: {len(ds)} 장 · 방법 {list(by)}")
    for m in by:
        n = NAMES[m]
        print(f"  {n:10s} 자세 틀림 {len(ds.match(F(f'{n}_wrong') == True))} · 누움 놓침 {len(ds.load_saved_view(f'{n}_miss_lying'))}"  # noqa: E712
              f" · 도움 오판 {len(ds.load_saved_view(f'{n}_false_help'))}")


if __name__ == "__main__":
    main()
