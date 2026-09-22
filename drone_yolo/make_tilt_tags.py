"""
make_tilt_tags.py — 학습 크롭마다 마운트각 태그를 붙인다 (2026-09-23)

지금까지
  survey_tilt*.py 가 **그룹 단위 판정표**를 만들었다 (NOMAD Actor001_a10 → 비스듬).
  그런데 학습 크롭 자체에는 아무 표시가 없어서 **층화 학습·평가에 못 쓴다.**

무엇을
  크롭 파일명 → 원본 그룹 → 상관 r → 태그 를 이어 붙인다.
    NOMAD  : Actor001_a10_f0001_c0.jpg      → "NOMAD Actor001_a10"
    WiSARD : 200402_Carnation_..._c0.jpg    → "WiSARD 200402_Carnation_Inspire_VIS"
  AI-Hub 는 파일명에 각도가 **확정**돼 있다 (…-맑음-90-35m-…) → r 없이 90° 로 박는다.

태그 (r = 박스 크기와 y좌표의 상관)
  r ≥ 0.30  비스듬_아래큼    r ≤ −0.30 비스듬_위큼    그 외 수직하향
  ⚠ r 은 **추정**이다 — 지형 경사·배우 이동이 섞인다. 선별보다 **층화 평가**에 먼저 쓴다.

실행: python make_tilt_tags.py
출력: metrics/tilt_tags.csv  (크롭 경로 · 그룹 · r · 태그)
"""
import csv, re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent


def load_survey():
    m = {}
    for f in ("survey_tilt.csv", "survey_tilt2.csv"):
        p = BASE / "metrics" / f
        if not p.exists():
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            m[r["group"]] = (float(r["corr_cy_px"]), r["verdict"])
    return m


# 실제 비행 폴더명과 대조한다 — 파일명의 `_VIS_2227` 은 프레임 번호라
# 정규식으로 자르면 비행명으로 오인한다 (09-23 발견)
# 크롭 파일명이 원본 프레임명(DJI_0001_00000000)을 쓰는데 그 프레임이 어느 비행 폴더에
# 있는지는 이름만으로 모른다 → **원본을 한 번 훑어 프레임→비행 대응표**를 만든다 (09-23)
def _frame_map():
    m = {}
    for d in (BASE / "data/raw/WiSARD").iterdir():
        if not d.is_dir():
            continue
        for f in list(d.rglob("*.jpg")) + list(d.rglob("*.jpeg")):
            m[f.stem] = d.name
    return m


FRAME2FLIGHT = _frame_map()


def group_of(stem):
    m = re.match(r"(Actor\d+_a\d+)_f\d+", stem)
    if m:
        return "NOMAD " + m.group(1)
    src = re.sub(r"_c\d+$", "", stem)       # 크롭 접미사 제거
    fl = FRAME2FLIGHT.get(src)
    return ("WiSARD " + fl) if fl else None


def tag_of(r):
    return "비스듬_아래큼" if r >= 0.30 else ("비스듬_위큼" if r <= -0.30 else "수직하향")


def main():
    sur = load_survey()
    rows, cnt, miss = [], Counter(), Counter()

    for d in sorted((BASE / "data/det_fov").iterdir()):
        for split in ("train", "val"):
            for p in sorted((d / "images" / split).glob("*.jpg")):
                g = group_of(p.stem)
                if g is None or g not in sur:
                    miss[g or "패턴불일치"] += 1
                    rows.append({"crop": str(p.relative_to(BASE)), "split": split,
                                 "group": g or "", "corr": "", "tilt": "미상"})
                    cnt["미상"] += 1
                    continue
                r, _ = sur[g]
                t = tag_of(r)
                rows.append({"crop": str(p.relative_to(BASE)), "split": split,
                             "group": g, "corr": round(r, 4), "tilt": t})
                cnt[t] += 1

    # 하드 네거티브 (사람이 없으니 각도 무관) — 기록만
    for p in sorted((BASE / "data/det_neg/images/train").glob("*.jpg")):
        rows.append({"crop": str(p.relative_to(BASE)), "split": "train",
                     "group": "hard_negative", "corr": "", "tilt": "음성"})
        cnt["음성"] += 1

    out = BASE / "metrics" / "tilt_tags.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["crop", "split", "group", "corr", "tilt"])
        w.writeheader(); w.writerows(rows)

    tot = sum(cnt.values())
    print(f"크롭 {tot:,}장에 태그 부착 → {out}\n")
    print(f"{'태그':<14} {'장수':>8} {'비율':>7}")
    for k, v in cnt.most_common():
        print(f"{k:<14} {v:>8,} {v/tot:>6.1%}")
    if miss:
        print("\n⚠ 태그 못 붙인 이유:", dict(miss.most_common(5)))

    # 학습 목록(.txt) 도 태그별로 뽑아 둔다 — 층화 학습이 필요해지면 바로 쓴다
    ld = BASE / "configs" / "lists"; ld.mkdir(parents=True, exist_ok=True)
    for t in ("비스듬_아래큼", "비스듬_위큼", "수직하향"):
        sel = [r["crop"] for r in rows if r["tilt"] == t and r["split"] == "train"]
        if sel:
            f = ld / f"train_tilt_{t}.txt"
            f.write_text("\n".join(str(BASE / c) for c in sel) + "\n")
            print(f"  → {f.name} ({len(sel):,}장)")


if __name__ == "__main__":
    main()
