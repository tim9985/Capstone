"""
tests_tile_merge.py — detector.edge_merge 확인 (10-11 · GPU 없이)

  ① 손으로 만든 경우 — 안쪽 경계에서 잘린 박스는 지우고 점수는 큰 박스로 · 이미지 테두리에 닿은 박스 · 떨어진 두 사람은 그대로
  ② 연구 규칙과 같은가 — drone_yolo/tile_merge.py 와 soup_v7r2 test_kr 덤프 (있으면) 에서 결과가 똑같아야 한다
실행: python tests_tile_merge.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from app.vision_core.detector import edge_merge  # noqa: E402

ok = 0


def check(name, cond):
    global ok
    print(("✅ " if cond else "❌ ") + name)
    ok += int(bool(cond))
    return cond


# ① 타일 0 = (0,0) · 1 = (640,0) · 2 = (0,360) · 3 = (640,360)
full = np.array([1250, 300, 1310, 420], np.float32)               # 타일 1 이 온전히 본 사람
cut = np.array([1250, 300, 1280, 420], np.float32)                # 타일 0 오른쪽 안쪽 경계 (x 1280) 에서 잘림
b, s = edge_merge(np.stack([cut, full]), np.array([0.9, 0.6], np.float32), np.array([0, 1]))
check("잘린 박스 지움", len(b) == 1 and np.allclose(b[0], full))
check("점수는 큰 박스로 (max)", len(s) == 1 and abs(s[0] - 0.9) < 1e-6)
edge = np.array([0, 300, 40, 420], np.float32)                     # 이미지 왼쪽 테두리 (안쪽 경계 아님)
other = np.array([0, 290, 80, 430], np.float32)
b, s = edge_merge(np.stack([edge, other]), np.array([0.8, 0.7], np.float32), np.array([0, 2]))
check("이미지 테두리 박스는 그대로", len(b) == 2)
p1 = np.array([1250, 300, 1280, 420], np.float32); p2 = np.array([1400, 600, 1440, 700], np.float32)
b, s = edge_merge(np.stack([p1, p2]), np.array([0.8, 0.7], np.float32), np.array([0, 1]))
check("떨어진 사람은 그대로 (잘렸어도 감쌀 박스가 없으면 남김)", len(b) == 2)
same = np.array([1250, 300, 1280, 420], np.float32); same2 = np.array([1200, 290, 1280, 430], np.float32)
b, s = edge_merge(np.stack([same, same2]), np.array([0.8, 0.7], np.float32), np.array([0, 0]))
check("같은 타일 박스끼리는 지우지 않음", len(b) == 2)

# ② 연구 규칙과 같은가
dump = HERE.parents[1] / "runs_person" / "dumps" / "soup_v7r2_test_kr.npz"
if dump.exists():
    sys.path.insert(0, str(HERE.parents[1]))
    from analyze_dets import nms                                     # noqa: E402
    from tile_merge import edge_merge as ref                         # noqa: E402
    d = np.load(dump, allow_pickle=True); diff = n = 0
    for i in np.unique(d["img"]):
        m = d["img"] == i; bb, ss, tt = d["box"][m], d["score"][m], d["tile"][m]
        k = nms(bb, ss, 0.6); bb, ss, tt = bb[k], ss[k], tt[k]
        b1, s1 = edge_merge(bb, ss, tt); b2, s2 = ref(bb, ss, tt)
        n += 1; diff += int(len(b1) != len(b2) or not (np.allclose(b1, b2) and np.allclose(s1, s2)))
    check(f"연구 규칙 (tile_merge.py) 과 같음 — test_kr {n}장 중 다른 장 {diff}", diff == 0)
print(f"{ok} 통과")
