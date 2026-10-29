"""
lying_index.py — v12 학습 사진 중 누운 사람이 든 것 표시 (10-30 L · 누움 보강 재학습)

  v12 이름 = <출처>_<crc32(원본 경로)>_<창 번호> (make_v12_data.py) → 원본으로 거슬러 자세 정보
    NOMAD: activityLabels 의 행동이 Laying · Hiding (Laying) 인 프레임
    Archangel: 주석 자세 lying down · crawling 이 있는 프레임 (det_v11 arch 크롭 이름 = arch_<영상>_<프레임>)
  ⚠ 원본 프레임 단위 — 창 (k) 이 그 사람을 안 담았을 수도 있음 → 창 라벨에 사람이 있는 것만
출력 configs/lists/lying_v12.txt (학습 목록 경로)
"""
import json, re, sys, zlib
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import make_v12_data as M
from make_posture_crops import NOMAD, _nomad_act
from make_v11_arch import load_ann

LYING_NOMAD = {"Laying", "Hiding (Laying)"}


def main():
    acts = {x["id"]: x["labels"] for x in json.load(open(NOMAD / "activityLabels.json"))}
    want = {}
    for r in M.src_nomad():
        p = Path(r["img"]); m = re.match(r"(Actor\d+)_d?(\w+)_f(\d+)", p.stem)
        actor, dist = p.parent.name.split("_")
        f = int(re.search(r"_f(\d+)", p.stem).group(1)) if re.search(r"_f(\d+)", p.stem) else None
        if f is None:
            continue
        a = _nomad_act(acts, int(actor[5:]), dist[1:], f)
        if a in LYING_NOMAD:
            want[f"nomad_{zlib.crc32(r['img'].encode()):08x}"] = "nomad"
    boxes, _ = load_ann()
    for r in M.src_archangel():
        stem = Path(r["img"]).stem                        # arch_<영상>_<프레임>
        vid, fr = stem[5:].rsplit("_", 1)
        if any(b[4] in ("lying down", "crawling") for b in boxes.get(vid, {}).get(int(fr), [])):
            want[f"archangel_{zlib.crc32(r['img'].encode()):08x}"] = "archangel"
    out, by = [], {}
    for l in open(BASE / "configs/lists/train_v12_f91.txt"):
        p = l.strip(); key = "_".join(Path(p).stem.split("_")[:2])
        if key in want:
            lab = Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
            if lab.exists() and lab.read_text().strip():
                out.append(p); by[want[key]] = by.get(want[key], 0) + 1
    (BASE / "configs/lists/lying_v12.txt").write_text("\n".join(out) + "\n")
    print("누움 원본 프레임", len(want), "· v12_f91 학습 사진 중", len(out), by)


if __name__ == "__main__":
    main()
