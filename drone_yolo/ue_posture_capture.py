"""
ue_posture_capture.py — UE(Cosys-AirSim) 에서 자세 데이터 만들기 (노트북에서 실행 · 2026-10-03)

왜
  비스듬 시점의 진짜 앉음 데이터가 SARD 621장뿐이다 (10-03 P0~P5) — 합성 C2A(15 px · 붙여 넣기)는 실제 앉음을 무너뜨렸다.
  UE 는 **우리 카메라 · 마운트각 · 고도 그대로** 렌더하고 자세 라벨이 자동이다 → 앉음 · 누움 학습 데이터 + 방위별 모양 연구
  계획 → obsidian 「11 서버 학습 계획/09 자세 데이터 확보」

레벨 준비 (UE)
  · 사람 배우 이름 = Person_<자세>_<번호>  예: Person_Sitting_03 · Person_Lying_07
      자세: Lying · Sitting · Kneeling · Crouching · Standing · Walking  (→ lying / sitting / standing 로 묶는다 · 원래 이름도 남긴다)
  · 자세는 애니메이션 한 프레임으로 고정 (Animation Mode = Use Animation Asset · Play 끔) · 배우마다 몸 방향(yaw)을 다르게
  · 한 레벨에 자세별 5~10명 · 서로 3 m 이상 떨어뜨림 · 나무 아래 · 풀숲 · 트인 곳을 섞는다
  · 레벨(맵)이 여럿이면 --place 로 이름을 달고, 하나는 평가 전용으로 남긴다 (장소 분리)
  · settings: configs/settings_geo.json (1920×1080 · FOV 80.2 · Scene · Segmentation) → ~/Documents/AirSim/settings.json

찍는 법
  배우마다 마운트 45 · 60 · 75° × 고도 16 · 20 · 25 · 30 m × 방위 8방향 (기본) — 대상이 화면 세로 안 어딘가에 오게 내려다보는 각을 무작위
  짐벌 안정화 가정 (기체 수평 · 카메라 피치 = −마운트) · 화면에 같이 잡힌 다른 배우도 모두 라벨
  박스 = 분할 마스크의 외곽 (완벽한 박스) · 8 px 미만 · 화면 가장자리에 잘린 배우는 표시만 해 둔다

출력 (--out 아래 · git 에 올리지 않는다 — 크기 때문 · 서버로는 rclone/scp)
  images/<n>.jpg · labels/<n>.txt (YOLO · 사람 1클래스 — 탐지 학습/평가에도 쓸 수 있다)
  crops/<자세>/*.jpg · crops.csv  — 서버 자세 크롭과 같은 형식 (file · source · place · view · alt · w1080 · h1080 · pose)
                                    + mount · dep · rel_az (카메라 방위 − 배우 몸 방위) · actor · pose_raw · mask_px · cut
  meta.csv — 장면마다 카메라 자세
  --weights 를 주면 탐지 모델로도 돌려 배우별 det_hit (IoU ≥ 0.5) 를 남긴다 → UE 누운 사람 탐지율

실행 (노트북 · UE 실행 중)
  python ue_posture_capture.py --place level01 --out ue_posture/level01 [--az 8] [--weights runs_person/soup_v9x2/weights/best.pt] [--tod]
"""
import argparse
import csv
import math
import random
import re
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import cosysairsim as airsim
from cosysairsim.utils import euler_to_quaternion, load_colormap

from make_posture_crops import crop

W, H, HFOV = 1920, 1080, 80.2
F = (W / 2) / math.tan(math.radians(HFOV / 2))
VFOV = math.degrees(2 * math.atan((H / 2) / F))
ACTOR = re.compile(r"Person_(Lying|Sitting|Kneeling|Crouching|Standing|Walking)_\d+", re.I)
POSE3 = {"lying": "lying", "sitting": "sitting", "kneeling": "sitting", "crouching": "sitting", "standing": "standing", "walking": "standing"}


def yaw_of(q):
    """쿼터니언 → yaw (rad, NED)."""
    return math.atan2(2 * (q.w_val * q.z_val + q.x_val * q.y_val), 1 - 2 * (q.y_val ** 2 + q.z_val ** 2))


def capture(client):
    reqs = [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)]
    client.simGetImages(reqs)                                  # 텔레포트 직후 첫 프레임은 버린다
    r = client.simGetImages(reqs)
    rgb = np.frombuffer(r[0].image_data_uint8, np.uint8).reshape(r[0].height, r[0].width, 3)
    seg = np.frombuffer(r[1].image_data_uint8, np.uint8).reshape(r[1].height, r[1].width, 3)
    return rgb, seg


def actors(client):
    """배우 이름 → {pose_raw, 색 목록 (부품마다 다른 색일 수 있다), 위치, 몸 방위}."""
    objs = client.simListInstanceSegmentationObjects()
    poses = client.simListInstanceSegmentationPoses(ned=True)
    cmap = load_colormap()
    out = {}
    for i, name in enumerate(objs):
        m = ACTOR.search(name)
        if not m:
            continue
        p = poses[i].position
        if not (math.isfinite(p.x_val) and math.isfinite(p.y_val)):
            continue
        a = out.setdefault(m.group(0), {"pose_raw": m.group(1).lower(), "colors": [], "pos": (p.x_val, p.y_val, p.z_val),
                                        "yaw": yaw_of(poses[i].orientation)})
        a["colors"].append(tuple(int(c) for c in cmap[i]))
    return out


def boxes_in(seg, acts):
    """화면에 보이는 배우마다 (박스, 마스크 화소 수, 가장자리 잘림)."""
    code = seg[:, :, 0].astype(np.int32) << 16 | seg[:, :, 1].astype(np.int32) << 8 | seg[:, :, 2].astype(np.int32)
    present = set(np.unique(code).tolist())
    out = {}
    for name, a in acts.items():
        cs = [c[0] << 16 | c[1] << 8 | c[2] for c in a["colors"]]
        cs = [c for c in cs if c in present]
        if not cs:
            continue
        ys, xs = np.where(np.isin(code, cs))
        if len(xs) < 30:
            continue
        b = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        if max(b[2] - b[0], b[3] - b[1]) < 8:
            continue
        cut = b[0] <= 0 or b[1] <= 0 or b[2] >= W or b[3] >= H
        out[name] = (b, int(len(xs)), bool(cut))
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def write_csvs(out, crow, mrow):
    for fn, rows in (("crops.csv", crow), ("meta.csv", mrow)):
        if rows:
            with open(out / fn, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--place", required=True, help="레벨(장소) 이름 — 장소 분리 평가의 단위")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mounts", type=int, nargs="+", default=[45, 60, 75])
    ap.add_argument("--alts", type=float, nargs="+", default=[16, 20, 25, 30])
    ap.add_argument("--az", type=int, default=8, help="방위 몇 방향")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--weights", default=None, help="주면 탐지 모델로 배우별 det_hit 기록")
    ap.add_argument("--tod", action="store_true", help="장면마다 시각(09 · 12 · 16시)을 바꿔 그림자 · 밝기 변화 (API 가 되면)")
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    out = Path(args.out)
    for d in ("images", "labels", *(f"crops/{p}" for p in ("lying", "sitting", "standing"))):
        (out / d).mkdir(parents=True, exist_ok=True)
    det = None
    if args.weights:
        from ultralytics import YOLO
        det = YOLO(args.weights)

    client = airsim.VehicleClient(); client.confirmConnection()
    acts = actors(client)
    if not acts:
        raise SystemExit("배우를 찾지 못함 — 이름을 Person_<자세>_<번호> 로 (예: Person_Sitting_03)")
    cnt = {}
    for a in acts.values():
        cnt[a["pose_raw"]] = cnt.get(a["pose_raw"], 0) + 1
    print(f"배우 {len(acts)} 명 {cnt} · 수직 화각 {VFOV:.1f}° · 장면 {len(acts) * len(args.mounts) * len(args.alts) * args.az:,} 예정")

    crow, mrow, n = [], [], 0
    for name, a in acts.items():
        tx, ty, tz = a["pos"]
        for mount in args.mounts:
            for h in args.alts:
                for k in range(args.az):
                    n += 1
                    if args.tod:
                        try:
                            client.simSetTimeOfDay(True, f"2026-10-15 {rnd.choice((9, 12, 16)):02d}:00:00", False, 1, 60, True)
                        except Exception:
                            args.tod = False
                    dep = rnd.uniform(max(10, mount - VFOV / 2 + 4), min(89, mount + VFOV / 2 - 4))   # 대상이 화면 세로 안에
                    d = h / math.tan(math.radians(dep))
                    b = 2 * math.pi * k / args.az + math.radians(rnd.uniform(-10, 10))             # 기체 → 대상 방위
                    yaw = b + math.radians(rnd.uniform(-25, 25))                                   # 가로로도 흩어지게
                    pos = (tx - d * math.cos(b), ty - d * math.sin(b), tz - h)
                    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(*pos), euler_to_quaternion(0, 0, yaw)), True)
                    client.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), euler_to_quaternion(0, -math.radians(mount), 0)))
                    time.sleep(0.15)
                    rgb, seg = capture(client)
                    bx = boxes_in(seg, acts)
                    if name not in bx:
                        continue
                    stem = f"{args.place}_{n:05d}"
                    bgr = rgb[:, :, ::-1].copy() if rgb.shape[2] == 3 else rgb
                    cv2.imwrite(str(out / "images" / f"{stem}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    hits = {}
                    if det is not None:
                        r = det.predict(bgr, imgsz=1280, conf=0.15, verbose=False)[0]
                        cands = [tuple(map(float, x)) for x in r.boxes.xyxy.cpu().numpy()]
                        hits = {an: int(any(iou(c, bb[0]) >= 0.5 for c in cands)) for an, bb in bx.items()}
                    with open(out / "labels" / f"{stem}.txt", "w") as f:
                        for an, (bb, px, cut) in bx.items():
                            f.write(f"0 {(bb[0] + bb[2]) / 2 / W:.6f} {(bb[1] + bb[3]) / 2 / H:.6f} {(bb[2] - bb[0]) / W:.6f} {(bb[3] - bb[1]) / H:.6f}\n")
                    for an, (bb, px, cut) in bx.items():
                        aa = acts[an]; pose = POSE3[aa["pose_raw"]]
                        cam_az = math.atan2(aa["pos"][1] - pos[1], aa["pos"][0] - pos[0])
                        rel = (math.degrees(cam_az - aa["yaw"]) + 180) % 360 - 180
                        cf = f"{pose}/{stem}_{an}.jpg"
                        cv2.imwrite(str(out / "crops" / cf), crop(bgr, bb, 1.0), [cv2.IMWRITE_JPEG_QUALITY, 95])
                        crow.append({"file": cf, "source": "ue", "place": args.place, "view": "nadir" if mount >= 80 else "oblique",
                                     "alt": h, "w1080": round(bb[2] - bb[0], 1), "h1080": round(bb[3] - bb[1], 1), "pose": pose,
                                     "mount": mount, "dep": round(dep, 1), "rel_az": round(rel, 1), "actor": an,
                                     "pose_raw": aa["pose_raw"], "mask_px": px, "cut": int(cut), "det_hit": hits.get(an, "")})
                    mrow.append({"image": f"{stem}.jpg", "target": name, "mount": mount, "alt": h, "dep": round(dep, 1),
                                 "az_deg": round(math.degrees(b), 1), "yaw_deg": round(math.degrees(yaw), 1), "people": len(bx)})
                    if n % 50 == 0:
                        print(f"  장면 {n:,} · 크롭 {len(crow):,}", flush=True)
                    if n % 200 == 0:
                        write_csvs(out, crow, mrow)   # 중간 저장 — 끊겨도 표가 남는다
    write_csvs(out, crow, mrow)
    by = {}
    for r in crow:
        by[r["pose"]] = by.get(r["pose"], 0) + 1
    print(f"\n장면 {len(mrow):,} · 크롭 {len(crow):,} {by} → {out}")
    if det is not None and crow:
        for p in ("lying", "sitting", "standing"):
            v = [r["det_hit"] for r in crow if r["pose"] == p and r["det_hit"] != "" and not r["cut"]]
            if v:
                print(f"  탐지율 {p}: {np.mean(v):.3f} (n={len(v)})")


if __name__ == "__main__":
    main()
