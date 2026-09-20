"""
make_px_table.py — 운용 조건별 사람 px 범위표 (2026-09-20)

왜
  전처리의 목표 크기 범위(지금 로그 균등 16~160 px)는 2026-09-15 에 정한 값인데
  그 전제 네 개가 모두 바뀌었다 — 서 있는 사람 0.5 m(→0.81 실측) · 화각 54°(→75/90°) ·
  모델 입력 1280 환산(→크롭이 1:1 배율) · 고도 16~30 m(→10~40 m · 마운트각 미정).
  조건이 정해지는 대로 바로 읽어 쓸 수 있게 표로 만들어 둔다.

식 (화면 중앙 기준)
  R = h / sin θ
  세로 px = f·(H·cosθ + w·sinθ) / R      가로 px = f·w / R      f = (1920/2)/tan(FOV/2)
  누운 사람은 지면 위이므로 방향에 따라 종·횡 GSD 가 달라진다 → 범위로 낸다

실행: python make_px_table.py [--heights 20 25 30] [--out metrics/px_table.csv]
"""
import argparse, csv, math
from pathlib import Path

BASE = Path(__file__).resolve().parent
D = math.radians
STAND_H, STAND_W = 1.70, 0.81      # 키, 지면 발자국 (09-17 AI-Hub 라벨 역산)
LIE_L, LIE_H = 1.70, 0.30          # 누운 길이(=키), 두께


def f_px(fov, width=1920):
    return (width / 2) / math.tan(D(fov / 2))


def standing(fov, h, th, width=1920):
    f, R = f_px(fov, width), h / math.sin(D(th))
    return max((STAND_H * math.cos(D(th)) + STAND_W * math.sin(D(th))) / R, STAND_W / R) * f


def lying(fov, h, th, width=1920):
    """누운 방향에 따라 달라진다 → (나란히, 가로질러)"""
    f, R = f_px(fov, width), h / math.sin(D(th))
    along = (LIE_H * math.cos(D(th)) + LIE_L * math.sin(D(th))) / R * f
    cross = LIE_L / R * f
    return min(along, cross), max(along, cross)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fovs", type=float, nargs="+", default=[75, 90, 93])
    ap.add_argument("--heights", type=float, nargs="+", default=[10, 15, 20, 25, 30, 35, 40])
    ap.add_argument("--mounts", type=float, nargs="+", default=[90, 75, 60, 45, 30])
    ap.add_argument("--width", type=int, default=1920, help="가로 해상도 (2K 검토용 2560 도 가능)")
    ap.add_argument("--out", default=str(BASE / "metrics" / "px_table.csv"))
    args = ap.parse_args()

    rows = []
    for fov in args.fovs:
        for th in args.mounts:
            for h in args.heights:
                s = standing(fov, h, th, args.width)
                lo, hi = lying(fov, h, th, args.width)
                rows.append({"width": args.width, "fov_deg": fov, "mount_deg": th, "altitude_m": h,
                             "f_px": round(f_px(fov, args.width)), "range_m": round(h / math.sin(D(th)), 1),
                             "standing_px": round(s, 1), "lying_min_px": round(lo, 1), "lying_max_px": round(hi, 1),
                             "min_px": round(min(s, lo), 1), "max_px": round(max(s, hi), 1)})

    Path(args.out).parent.mkdir(exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # 화면 표 — 마운트각 90° (하향) 기준
    print(f"가로 {args.width} px · 서 있는 사람 {STAND_H} m(발자국 {STAND_W}) · 누운 사람 {LIE_L} m\n")
    for th in args.mounts:
        print(f"── 마운트각 {th:g}° " + ("(수직 하향)" if th == 90 else ""))
        print("  화각 " + "".join(f"{h:g} m".rjust(16) for h in args.heights))
        for fov in args.fovs:
            cells = []
            for h in args.heights:
                s = standing(fov, h, th, args.width); lo, hi = lying(fov, h, th, args.width)
                cells.append(f"{s:.0f} / {lo:.0f}~{hi:.0f}".rjust(16))
            print(f"  {fov:>3.0f}°" + "".join(cells))
        print("        (서 있음 / 누움 최소~최대)")
        print()

    print("조건 조합별 전체 범위 — 전처리 목표 크기를 정할 때 읽는다")
    for label, sub in (("전 조건 (마운트 30~90°)", rows),
                       ("하향 90° 고정", [r for r in rows if r["mount_deg"] == 90]),
                       ("하향 90° · 고도 20~30 m", [r for r in rows if r["mount_deg"] == 90 and 20 <= r["altitude_m"] <= 30]),
                       ("화각 75° · 하향 90° · 20~30 m", [r for r in rows if r["fov_deg"] == 75 and r["mount_deg"] == 90 and 20 <= r["altitude_m"] <= 30])):
        if not sub:
            continue
        print(f"  {label:>28}: {min(r['min_px'] for r in sub):>5.0f} ~ {max(r['max_px'] for r in sub):>5.0f} px")
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
