"""
make_present_dd.py — 발표 자료용 디지털관 (디관) 사진 결과 그림 (10-17)

  운용 입력과 같게: 원본 4:3 → 아래쪽 16:9 띠 → 1920×1080 (fo_dd_fhd.py 와 같은 변환)
  → 배포 탐지 (v12_f91_swa · TensorRT · 임계값 0.20 · 타일 4장 · 경계 합치기) → 박스마다 상의 색 (worker ColorProfiler 그대로)
  출력 (git 밖 · 발표자료 폴더): results.json · 전체 미리보기 · 고른 장면 합성 그림 (위 = 전체 화면 번호 박스 · 아래 = 확대 + 확신도 · 상의 색)
  ⚠ 디관 사진은 정답 라벨이 없다 → 정성 예시 (수치 판정은 장소 분리 평가셋 · UE)
실행  /home/se/miniconda3/envs/drone/bin/python make_present_dd.py <작업 폴더>            (추론 · results.json · 미리보기)
      /home/se/miniconda3/envs/drone/bin/python make_present_dd.py <그림 폴더> pick <작업 폴더>   (고른 6장 합성)
"""
import json, sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "deploy" / "vision_worker"))
from app.vision_core.detector import PersonDetector  # noqa: E402
from app.vision_core.color import ColorProfiler, frame_stats, KO  # noqa: E402

SRC = BASE.parent / "data" / "digital_department_test"
PACK = BASE / "runs_person" / "_pack_v12_sim" / "v12_f91_swa"
FONT = str(BASE.parent / "발표자료" / "_도구" / "NanumGothicBold.ttf")      # 나눔고딕 (OFL) · 발표자료 폴더 (git 밖)
SWATCH = {"red": (220, 40, 40), "orange": (245, 140, 30), "yellow": (240, 210, 40), "green": (40, 160, 70), "blue": (40, 100, 220),
          "navy": (30, 40, 110), "purple": (130, 60, 170), "pink": (240, 140, 180), "white": (245, 245, 245), "gray": (140, 140, 140),
          "black": (20, 20, 20), "brown": (120, 75, 40)}


def fhd(img):
    h, w = img.shape[:2]
    ch = round(w * 9 / 16); y0 = h - ch
    return cv2.resize(img[y0:, :], (1920, 1080), interpolation=cv2.INTER_AREA)


def font(n):
    return ImageFont.truetype(FONT, n)


def run_all(out):
    det = PersonDetector(PACK / "best.pt", PACK / "best_trt11.3.0.99_RTX3090.engine", conf=0.20)
    prof = ColorProfiler()
    res = []
    for i, f in enumerate(sorted(SRC.glob("*.jpg"))):
        frame = fhd(cv2.imread(str(f)))
        boxes, confs, ms = det.detect(frame)
        st = frame_stats(frame)
        dets = []
        for (x1, y1, x2, y2), c in sorted(zip(boxes.tolist(), confs.tolist()), key=lambda t: -t[1]):
            p = prof.profile(frame, (x1, y1, x2, y2), stats=st)
            dets.append({"box": [round(v, 1) for v in (x1, y1, x2, y2)], "conf": round(c, 3), "h_px": round(y2 - y1, 1),
                         "color_status": p["status"], "color_top": p["top"], "n_pixels": p["n_pixels"]})
        res.append({"idx": i, "file": f.name, "ms": round(ms, 1), "dets": dets})
        print(i, f.name, len(dets), "박스", round(ms, 1), "ms")
    (out / "results.json").write_text(json.dumps({"model": "v12_f91_swa · TensorRT FP16 · conf 0.20 · 타일 4장 · 경계 합치기",
                                                  "input": "원본 4:3 아래쪽 16:9 → 1920×1080", "images": res}, ensure_ascii=False, indent=1))
    return res


def overview(res, out):
    tiles = []
    for r in res:
        frame = fhd(cv2.imread(str(SRC / r["file"])))
        for d in r["dets"]:
            x1, y1, x2, y2 = map(int, d["box"])
            cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 0, 255), 4)
        t = cv2.resize(frame, (640, 360))
        cv2.putText(t, f"{r['idx']} ({len(r['dets'])})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        tiles.append(t)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out / "_overview.jpg"), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 80])


def composite(r, out, name, title, sub=None):
    """위: 1920×1080 전체 (번호 박스) · 아래: 확대 칸 (최대 8) — 확신도 · 상의 1·2순위"""
    frame = fhd(cv2.imread(str(SRC / r["file"])))
    dets = r["dets"][:8]
    W, cell, cap = 1920, 240, 96
    ncol = max(len(dets), 1)
    cw = min(cell, W // ncol)
    top = 70 + (44 if sub else 0)
    canvas = Image.new("RGB", (W, 1080 + top + cw + cap), (250, 250, 250))
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(img)
    for k, d in enumerate(r["dets"], 1):            # 박스는 전부 · 아래 확대 칸은 확신도 상위 8개
        x1, y1, x2, y2 = d["box"]
        pad, col = 6, (255, 40, 40) if k <= 8 else (255, 150, 0)
        dr.rectangle([x1 - pad, y1 - pad, x2 + pad, y2 + pad], outline=col, width=4 if k <= 8 else 3)
        tw = 34 if k < 10 else 46
        dr.rectangle([x1 - pad, y1 - pad - 34, x1 - pad + tw, y1 - pad], fill=col)
        dr.text((x1 - pad + 6, y1 - pad - 33), str(k), font=font(28), fill=(255, 255, 255))
    canvas.paste(img, (0, top))
    dc = ImageDraw.Draw(canvas)
    dc.text((20, 16), title, font=font(36), fill=(30, 30, 30))
    if sub:
        dc.text((22, 66), sub, font=font(24), fill=(90, 90, 90))
    y0 = 1080 + top
    for k, d in enumerate(dets, 1):
        x1, y1, x2, y2 = d["box"]
        cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 1.6
        X1, Y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
        X2, Y2 = int(min(1920, cx + s / 2)), int(min(1080, cy + s / 2))
        crop = cv2.cvtColor(frame[Y1:Y2, X1:X2], cv2.COLOR_BGR2RGB)
        crop = Image.fromarray(crop).resize((cw - 8, cw - 8), Image.NEAREST if s < 60 else Image.BICUBIC)
        x = (k - 1) * cw
        canvas.paste(crop, (x + 4, y0 + 4))
        dc.rectangle([x + 4, y0 + 4, x + 22 + 4, y0 + 4 + 26], fill=(255, 40, 40))
        dc.text((x + 9, y0 + 5), str(k), font=font(22), fill=(255, 255, 255))
        dc.text((x + 8, y0 + cw + 4), f"확신도 {d['conf']:.2f} · 키 {d['h_px']:.0f} px", font=font(20), fill=(30, 30, 30))
        if d["color_status"] == "ok":
            for j, (c, p) in enumerate(d["color_top"][:2]):
                yy = y0 + cw + 34 + j * 28
                dc.rectangle([x + 8, yy + 2, x + 30, yy + 22], fill=SWATCH[c], outline=(0, 0, 0))
                dc.text((x + 38, yy), f"{j + 1}순위 {KO[c]} {p:.2f}", font=font(20), fill=(30, 30, 30))
        else:
            dc.text((x + 8, y0 + cw + 34), "상의 색 판정 불가", font=font(20), fill=(150, 30, 30))
            dc.text((x + 8, y0 + cw + 62), f"(상의 화소 {d['n_pixels']})", font=font(18), fill=(120, 120, 120))
    canvas.save(out / name, quality=90)


PICK = [  # (사진 번호, 파일 이름, 제목, 부제) — 10-17 고름 · 잘 되는 예 + 틀리는 예
    (15, "01_디관_광장_보행자.jpg", "사례 1 · 광장 보행자 — 11명 탐지 (그늘 속 · 화면 끝에 걸린 사람 포함)",
     "상의 색: 검정 계열 다수 · 4 = 밝은 회색 후드 → 회색 0.93 (맞음) · 9~11 은 확신도 상위 8개 밖 (주황 박스)"),
    (16, "02_디관_무리_킥보드.jpg", "사례 2 · 여러 명 무리 · 킥보드 — 가까운 9명 + 먼 사람 2명 탐지",
     "6 = 밝은 회색 후드 → 회색 0.80 (맞음) · ✗ 5 흰 셔츠 + 검정 가방 → 검정 0.28 (가방) · ✗ 7 자주색 상의 → 검정 0.41"),
    (17, "03_디관_먼사람_색오판.jpg", "사례 3 · 키 36~67 px 먼 사람까지 탐지 · 상의 색은 틀리는 예",
     "✗ 4 자주색 상의 → 검정 · ✗ 6 흰 셔츠+검정 가방 → 남색 · 10 = 화면 모서리에 걸린 물체 (가장자리 박스 → 후보로 안 만듦 · 10-17)"),
    (9, "04_디관_먼거리_해질녘.jpg", "사례 4 · 해 질 녘 · 멀리 작은 사람 (키 15~38 px · 운용 16~20 m 는 50~100 px)",
     "✗ 1 흰 옷 → 파랑 (그늘의 푸른빛) · 15~20 px 박스는 눈으로도 사람인지 확인 어려움 → 여러 장 · 가까이 다시 보기로 확인"),
    (14, "05_디관_나무가림_가로등오탐.jpg", "사례 5 · 나무 사이 · 그늘 속 사람 탐지 (5~9) · 가로등 오탐 (11)",
     "✗ 11 = 가로등 머리 (확신도 0.38 · 사람 크기 물체 오탐 — 수레 · 상자 · 표지물과 같은 종류) · 13 · 14 = 먼 작은 물체 중복 박스"),
    (1, "06_디관_우산.jpg", "사례 6 · 우산 쓴 사람도 탐지 (2 · 확신도 0.82)",
     "상의 색은 우산 색 (파랑 0.67) 으로 나옴 — 위에서 보면 상의가 가려짐 → 상의 색은 순위 보조로만 씀"),
]


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 2 and sys.argv[2] == "all":           # 21장 전부 (사용자가 직접 고름)
        res = json.load(open(Path(sys.argv[3]) / "results.json"))["images"]
        for r in res:
            n = len(r["dets"]); hs = [d["h_px"] for d in r["dets"]]
            composite(r, out, f"디관_{r['idx'] + 1:02d}.jpg", f"디지털관 {r['idx'] + 1:02d} · 탐지 박스 {n}개",
                      f"키 {min(hs):.0f}~{max(hs):.0f} px · 확신도 ≥ 0.20 · 빨간 박스 = 아래 확대 칸 (확신도 상위 8) · 원본 {r['file']}" if n else None)
    elif len(sys.argv) > 2 and sys.argv[2] == "pick":          # 이미 돌린 results.json 으로 고른 장면만 그림
        res = json.load(open(Path(sys.argv[3]) / "results.json"))["images"]
        for i, name, title, sub in PICK:
            composite(res[i], out, name, title, sub)
    else:
        res = run_all(out)
        overview(res, out)
