"""
obs_time.py — 관측 시간: 한 사람을 몇 번(몇 초) 보면 한 번이라도 찾는가 (2026-09-28 · 9/28 회의 안건)

  test_obl (Okutama 비스듬 16 시퀀스 · 시퀀스당 약 1 초 간격 60장) 에 원본 라벨의 **추적 ID** 를 다시 붙여
  같은 사람을 연속 k 장(≈ k 초) 보는 동안 conf ≥ 0.15 · IoU ≥ 0.5 로 **한 번 이상** 찾을 확률 P(k) 를 잰다.
  연속 프레임은 서로 닮아서(같은 가림·자세) 한 장 재현율 p 로 1 − (1 − p)^k 를 계산하면 과장된다 → 실측으로 낸다.
  P_2회이상 = k 번 중 두 번 이상 찾음 (후보 등록에 연속 확인을 요구할 때)
  입력: runs_person/<모델>/preds_test_obl.npz (analyze_models.py 캐시 · conf ≥ 0.01)
출력: metrics/obs_time_<모델>.json
실행: python obs_time.py --model soup_v7r2
"""
import argparse, glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import diag_misses as D

BASE = Path(__file__).resolve().parent
W, SW, FPS = 1280, 3840, 30


def raw_tracks(seq):
    """make_testset_obl.labels 와 같은 거르기 · (추적 ID, 가림, 행동) 을 함께"""
    per = defaultdict(list)
    files = sorted(glob.glob(str(BASE / f"data/raw/okutama/**/Labels/SingleActionLabels/3840x2160/{seq}.txt"), recursive=True),
                   key=lambda f: "TrainSetVideos" in f)
    for ln in open(files[0], encoding="utf-8", errors="replace"):
        t = ln.split()
        if len(t) < 7:
            continue
        try:
            tid, x1, y1, x2, y2, fr, lost = (int(t[i]) for i in range(7))
        except ValueError:
            continue
        if lost:
            continue
        s = W / SW
        if (x2 - x1) * s > 3 and (y2 - y1) * s > 3:
            occ = int(t[7]) if len(t) > 7 and t[7].isdigit() else 0
            act = t[10].strip('"') if len(t) > 10 else ""
            per[fr].append((tid, occ, act))
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="soup_v7r2")
    ap.add_argument("--kmax", type=int, default=6)
    args = ap.parse_args()
    recs = list(np.load(BASE / "runs_person" / args.model / "preds_test_obl.npz", allow_pickle=True)["recs"])
    raw = {}
    seqs = defaultdict(list)      # seq -> [(frame, {tid: (hit50, hit30, occ, act)})]
    bad = 0
    for stem, G, B, C in recs:
        _, seq, fr = stem.split("_"); fr = int(fr)
        if seq not in raw:
            raw[seq] = raw_tracks(seq)
        ids = raw[seq].get(fr, [])
        if len(ids) != len(G):
            bad += 1; continue
        k = D.nms(B, C, 0.7); P, Cc = B[k], C[k]
        hi = Cc >= 0.15; P = P[hi]
        M = D.iou(G, P) if len(G) and len(P) else np.zeros((len(G), 0))
        hit50 = np.zeros(len(G), bool); used = np.zeros(M.shape[1], bool)
        for i in np.argsort(-(M.max(1) if M.shape[1] else np.zeros(len(G)))):
            if M.shape[1]:
                j = int(np.argmax(np.where(used, -1, M[i])))
                if M[i, j] >= 0.5 and not used[j]:
                    hit50[i] = True; used[j] = True
        hit30 = (M.max(1) >= 0.3) if M.shape[1] else np.zeros(len(G), bool)
        seqs[seq].append((fr, {tid: (bool(h5), bool(h3), occ, act) for (tid, occ, act), h5, h3 in zip(ids, hit50, hit30)}))
    gaps = []
    out = {"모델": args.model, "라벨_불일치_장수": bad, "k": {}}
    for iou_key, idx in (("IoU0.5", 0), ("IoU0.3", 1)):
        res = {}
        for kk in range(1, args.kmax + 1):
            hits, n, occ_hits, occ_n, two = 0, 0, 0, 0, 0
            for seq, fl in seqs.items():
                fl.sort()
                if kk == 1:
                    gaps += [(b[0] - a[0]) / FPS for a, b in zip(fl, fl[1:])]
                for s in range(len(fl) - kk + 1):
                    win = fl[s:s + kk]
                    common = set(win[0][1]).intersection(*[set(w[1]) for w in win[1:]])
                    for tid in common:
                        cnt = sum(w[1][tid][idx] for w in win); got = cnt >= 1
                        n += 1; hits += got; two += cnt >= 2
                        if any(w[1][tid][2] for w in win):
                            occ_n += 1; occ_hits += got
            res[kk] = {"P": round(hits / max(n, 1), 3), "P_2회이상": round(two / max(n, 1), 3), "창": n,
                       "가림_P": round(occ_hits / max(occ_n, 1), 3)}
        out["k"][iou_key] = res
    out["프레임_간격_s_중앙"] = round(float(np.median(gaps)), 2) if gaps else None
    (BASE / "metrics" / f"obs_time_{args.model}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
