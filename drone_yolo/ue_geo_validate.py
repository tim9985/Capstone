"""
ue_geo_validate.py — UE(Cosys-AirSim) 에서 GeoResolver 검증 (노트북에서 실행 · 2026-09-28)

  깊이 센서를 쓰지 않는다 — 실기체와 같게 **박스 아래 변(발끝) + 지면 교차** 로 좌표를 내고 UE 정답과 비교한다.
  표본마다: 사람(대상)을 고르고 → 고도 16~30 m · 마운트 45/60/75° 에서 대상이 화면 안에 오도록 기체를 놓고
           → 분할 마스크로 '완벽한 박스', (선택) 탐지 모델로 '실제 박스' 를 얻어 → GeoResolver → 수평 오차
  두 방식을 번갈아 잰다
    stab   : 짐벌 안정화 가정 — 기체 수평 · 카메라 절대 피치 = −마운트 (GM3 V2 가 roll·pitch 를 상쇄)
    body   : 기체 roll·pitch ±5° + 카메라는 기체에 고정 — 회전 사슬(기체 → 카메라) 검증
  지면 높이 두 가지로 계산해 평지 가정의 대가를 본다
    alt_true  : 대상 발밑 기준 고도 (지형을 완벽히 안다)
    alt_home  : 원점(이륙점) 기준 고도 = 평지 가정 (산지면 여기서 오차가 난다)

준비: UE 실행 + configs/settings_geo.json (1920×1080 · FOV 80.2 · Scene · Segmentation) → ~/Documents/AirSim/settings.json
실행: python ue_geo_validate.py [--samples 120] [--weights runs_person/soup_v7r2/weights/best.pt]
출력: metrics/ue_geo_validate.csv · 콘솔 요약 (마운트 × 고도 × 방식)
"""
import argparse, csv, math, random, sys, time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import cosysairsim as airsim
from cosysairsim.utils import euler_to_quaternion, load_colormap
from geo_resolver import CameraModel, GeoResolver, Telemetry

BASE = Path(__file__).resolve().parent
OUT = BASE / "metrics" / "ue_geo_validate.csv"
CAM = CameraModel(1920, 1080, 80.2)
VFOV = math.degrees(2 * math.atan((CAM.height / 2) / CAM.f))
PERSON = ("SkeletalMeshActor", "Person", "Human", "Mannequin", "Character")
FALLBACK = ("Cube", "Cylinder", "Cone")
ORIGIN = (35.1796, 129.0756)


def capture(client):
    reqs = [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)]
    client.simGetImages(reqs)                      # 텔레포트 직후 첫 프레임은 버린다
    r = client.simGetImages(reqs)
    rgb = np.frombuffer(r[0].image_data_uint8, np.uint8).reshape(r[0].height, r[0].width, 3)
    seg = np.frombuffer(r[1].image_data_uint8, np.uint8).reshape(r[1].height, r[1].width, 3)
    return rgb, seg


def mask_box(seg, color):
    m = np.all(seg == np.array(color, np.uint8), axis=2)
    if m.sum() < 30:
        return None
    ys, xs = np.where(m)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--weights", default=None, help="주면 탐지 모델 박스로도 계산")
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    det = None
    if args.weights:
        from ultralytics import YOLO
        det = YOLO(args.weights)

    client = airsim.VehicleClient(); client.confirmConnection()
    objs = client.simListInstanceSegmentationObjects()
    poses = client.simListInstanceSegmentationPoses(ned=True)
    cmap = load_colormap()
    def pick(pats):
        out = []
        for i, name in enumerate(objs):
            if any(k.lower() in name.lower() for k in pats):
                p = poses[i].position
                if math.isfinite(p.x_val) and math.isfinite(p.y_val) and abs(p.x_val) < 150 and abs(p.y_val) < 150:
                    out.append((name, [int(c) for c in cmap[i]], (p.x_val, p.y_val, p.z_val)))
        return out
    targets = pick(PERSON) or pick(FALLBACK)
    if not targets:
        raise SystemExit("대상(사람 · 큐브) 을 찾지 못함 — 레벨에 배치 확인")
    home_z = min(t[2][2] for t in targets)          # 원점 지면 ≈ 가장 낮은 대상의 발밑 (평지 가정 기준)
    print(f"대상 {len(targets)}개 · 수직 화각 {VFOV:.1f}° · 원점 지면 z {home_z:.2f}")

    gr = GeoResolver(CAM)
    rows, tries = [], 0
    while len(rows) < args.samples and tries < args.samples * 20:
        tries += 1
        name, color, (tx, ty, tz) = rnd.choice(targets)
        mount = rnd.choice((45, 60, 75)); h = rnd.choice((16, 20, 25, 30)); mode = rnd.choice(("stab", "body"))
        dep = rnd.uniform(mount - VFOV / 2 + 4, min(89, mount + VFOV / 2 - 4))    # 대상이 화면 세로 안에
        d = h / math.tan(math.radians(dep)); b = rnd.uniform(0, 2 * math.pi)
        yaw = b + math.radians(rnd.uniform(-25, 25))                             # 가로로도 흩어지게
        roll, pitch = (math.radians(rnd.uniform(-5, 5)), math.radians(rnd.uniform(-5, 5))) if mode == "body" else (0.0, 0.0)
        dn = (tx - d * math.cos(b), ty - d * math.sin(b), tz - h)
        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(*dn), euler_to_quaternion(roll, pitch, yaw)), True)
        client.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), euler_to_quaternion(0, -math.radians(mount), 0)))
        time.sleep(0.2)
        rgb, seg = capture(client)
        mb = mask_box(seg, color)
        if mb is None:
            continue
        boxes = {"mask": mb}
        if det is not None:
            r = det.predict(rgb[:, :, ::-1].copy(), imgsz=1280, conf=0.15, verbose=False)[0]
            cands = [tuple(map(float, x)) for x in r.boxes.xyxy.cpu().numpy()]
            best = max(cands, key=lambda c: iou(c, mb), default=None)
            if best is not None and iou(best, mb) >= 0.3:
                boxes["det"] = best
        row = dict(n=len(rows) + 1, target=name, mode=mode, mount=mount, alt=h, depression=round(dep, 1),
                   roll=round(math.degrees(roll), 2), pitch=round(math.degrees(pitch), 2), yaw=round(math.degrees(yaw), 1),
                   gt_n=round(tx, 3), gt_e=round(ty, 3), ground_z=round(tz, 3))
        for kind, bb in boxes.items():
            for gname, agl in (("true", tz - dn[2]), ("home", home_z - dn[2])):
                tel = Telemetry(ORIGIN[0], ORIGIN[1], agl, roll, pitch, yaw, gimbal_pitch_deg=-mount,
                                stabilized=(mode == "stab"))
                res = gr.to_world(bb, tel)
                if res.north is None:
                    row[f"err_{kind}_{gname}"] = ""; continue
                row[f"err_{kind}_{gname}"] = round(math.hypot(dn[0] + res.north - tx, dn[1] + res.east - ty), 3)
                row[f"status_{kind}"] = res.status
            row[f"box_{kind}"] = ",".join(f"{v:.0f}" for v in bb)
        rows.append(row)
        print(f"  #{row['n']:3d} {mode} {mount}° {h} m  완벽박스 {row.get('err_mask_true')} m (평지 {row.get('err_mask_home')})"
              + (f" · 탐지박스 {row.get('err_det_true')} m" if 'err_det_true' in row else ""))

    if not rows:
        raise SystemExit("표본 없음")
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\n=== 수평 오차 m (평균 / 중앙 / 최대) · {len(rows)} 표본 ===")
    for col in ("err_mask_true", "err_mask_home", "err_det_true", "err_det_home"):
        for mount in (45, 60, 75):
            v = np.array([r[col] for r in rows if r.get(col) not in (None, "") and r["mount"] == mount], float)
            if len(v):
                print(f"  {col:15} {mount}°  n={len(v):3d}  {v.mean():5.2f} / {np.median(v):5.2f} / {v.max():5.2f}")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
