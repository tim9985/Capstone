"""
make_present_ue.py — 발표 자료용 UE 장면 그림 (10-17) — 추적 · 후보 (S2) · 요구조자 상태 (S3) · 상의 색 (S4)

  배포 판 worker 코드 (integration_v1_2 · drone-real app 위에 덮어씀 = tests_v1_2.py 와 같은 방식) 를 10 fps 그대로 돌려
  프레임마다 관측 (후보 · 추적 키 · 상태 · 상의 색) 을 모으고 정답 박스 (배우 이름 = 자세 · 상의 색) 와 맞춰 그린다
  설정: v12_f91_swa TensorRT · 0.20 · 후보 묶기에 pose 위치 (VISION_GEO_GROUPING=1 · UE 평지 가정 VISION_GEO_AGL_FLAT=1 — tests_v1_2 와 같음)
실행  /home/se/miniconda3/envs/drone/bin/python make_present_ue.py run <출력 폴더> S2_a16 S3_a16 S4_a20
      /home/se/miniconda3/envs/drone/bin/python make_present_ue.py draw <run 결과 폴더> <그림 폴더>
"""
import json, os, shutil, sys, tempfile, uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
HERE = BASE / "deploy" / "vision_worker" / "integration_v1_2"
BACKEND = "/home/se/sandboxes/drone-real"
UE = BASE / "ue_flight"
FONT = str(BASE.parent / "발표자료" / "_도구" / "NanumGothicBold.ttf")
POSE_KO = {"Lying": "누움", "Sitting": "앉음", "Standing": "서기", "Walking": "걷기", "Crawling": "기어감", "Crouching": "웅크림"}
COLOR_KO = {"red": "빨강", "orange": "주황", "yellow": "노랑", "green": "초록", "blue": "파랑", "navy": "남색", "purple": "보라",
            "pink": "분홍", "white": "흰색", "gray": "회색", "black": "검정", "brown": "갈색"}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / u if u > 0 else 0.0


def run(out, runs):
    stage = Path(tempfile.mkdtemp(prefix="present_ue_"))
    shutil.copytree(Path(BACKEND) / "app", stage / "app", ignore=shutil.ignore_patterns("*.bak-*", "__pycache__"))
    for f in (HERE / "app").rglob("*.py"):
        dst = stage / "app" / f.relative_to(HERE / "app"); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst)
    sys.path.insert(0, str(stage)); sys.path.insert(0, str(HERE))
    os.environ.update(VISION_GEO_GROUPING="1", VISION_GEO_AGL_FLAT="1", VISION_GEO_CLAIMS="0", VISION_MIN_AGL_M="10")
    import cv2
    from app.vision_core.detector import PersonDetector
    from app.vision_core.ingest import IngestWorker
    from tests_v1_2 import frame_pose
    W = BASE / "runs_person" / "_pack_v12_sim" / "v12_f91_swa"
    det = PersonDetector(W / "best.pt", engine=W / "best_trt11.3.0.99_RTX3090.engine", conf=0.20)
    posture = str(BASE / "runs_person" / "_pack_v12" / "posture.json")
    t0 = datetime(2026, 10, 17, 6, 0, tzinfo=timezone.utc)
    import csv
    for rn in runs:
        d = UE / rn
        meta = json.load(open(d / "run.json")); actors = json.load(open(d / "actors.json"))
        gt = {json.loads(l)["frame"]: json.loads(l)["boxes"] for l in open(d / "boxes.jsonl")}
        w = IngestWorker(None, lambda: "x", det, str(uuid.uuid4()), scratch=tempfile.mkdtemp(prefix="present_w_"),
                         upload_crops=False, posture=posture, environment="SIM")
        mid = str(uuid.uuid4()); frames = []
        for r in csv.DictReader(open(d / "flight.csv")):
            t = float(r["t_s"]); cap = (t0 + timedelta(seconds=t)).isoformat().replace("+00:00", "Z")
            pose = frame_pose(r, meta, str(uuid.uuid4()), cap, "SIM")
            img = cv2.imread(str(d / "frames" / f"{r['frame']}.jpg"))
            obs = w.observe(mid, img, t, pose, upload_crops=False)
            g = {n: b for n, b in (gt.get(r["frame"]) or {}).items() if n in actors and not b.get("cut")}
            rows = []
            for o in obs:
                bb = (o["bbox"]["x1"], o["bbox"]["y1"], o["bbox"]["x2"], o["bbox"]["y2"])
                best, actor = 0.3, None
                for n, b in g.items():
                    v = iou(bb, b["bb"])
                    if v >= best:
                        best, actor = v, n
                st = o.get("state") or {}
                rows.append({"bbox": [round(x, 1) for x in bb], "conf": o["confidence"], "cand": o["candidate_id"][:8], "track": o["track_id"],
                             "actor": actor, "grade": st.get("grade"), "lying": (st.get("posture") or {}).get("p", {}).get("lying"),
                             "still_s": (st.get("motion") or {}).get("still_s"), "tracked": st.get("tracked"),
                             "s_lying": (st.get("score_terms") or {}).get("lying"), "s_sitting": (st.get("score_terms") or {}).get("sitting"),
                             "score": st.get("score"),
                             "upper": (o["appearance"]["upper"]["top"] or [{}])[0].get("color"), "upper_status": o["appearance"]["upper"]["status"]})
            frames.append({"frame": r["frame"], "t": t, "obs": rows,
                           "gt": {n: {"bb": b["bb"], "px": b["px"]} for n, b in g.items()}})
        (out / f"{rn}.json").write_text(json.dumps({"run": rn, "actors": actors, "frames": frames}, ensure_ascii=False))
        print(rn, len(frames), "장", "관측", sum(len(f["obs"]) for f in frames))


GRADE_COL = {"URGENT": (220, 30, 30), "HIGH": (245, 130, 20), "CHECK": (230, 200, 40), "NORMAL": (150, 150, 150)}
SW = {"red": (220, 40, 40), "orange": (245, 140, 30), "yellow": (240, 210, 40), "green": (40, 160, 70), "blue": (40, 100, 220),
      "navy": (30, 40, 110), "purple": (130, 60, 170), "pink": (240, 140, 180), "white": (245, 245, 245), "gray": (140, 140, 140),
      "black": (20, 20, 20), "brown": (120, 75, 40)}


def _font(n):
    from PIL import ImageFont
    return ImageFont.truetype(FONT, n)


def _region(boxes, W=1920, H=1080, margin=90, aspect=16 / 9, min_w=640):
    x1 = min(b[0] for b in boxes) - margin; y1 = min(b[1] for b in boxes) - margin
    x2 = max(b[2] for b in boxes) + margin; y2 = max(b[3] for b in boxes) + margin
    w, h = max(x2 - x1, min_w), y2 - y1
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = min(w, W), min(h, H)
    X1 = int(min(max(0, cx - w / 2), W - w)); Y1 = int(min(max(0, cy - h / 2), H - h))
    return X1, Y1, int(X1 + w), int(Y1 + h)


def _crop(run, frame, region, out_w=1600):
    from PIL import Image
    im = Image.open(UE / run / "frames" / f"{frame}.jpg").convert("RGB").crop(region)
    k = out_w / im.width
    return im.resize((out_w, int(im.height * k)), Image.BICUBIC), k


def _label(dr, xy, text, fill, size=26, fg=(255, 255, 255)):
    f = _font(size); x, y = xy
    l, t, r, b = dr.textbbox((x, y), text, font=f)
    dr.rectangle([l - 6, t - 4, r + 6, b + 4], fill=fill)
    dr.text((x, y), text, font=f, fill=fg)


def _tiles(run, frame, items, cell=230, cap_lines=3, cap_h=30):
    """사람마다 확대 칸 — items = [(box, [(글, 배경색, 글자색)…])] → PIL 그림 (가로 한 줄)"""
    from PIL import Image, ImageDraw
    src = Image.open(UE / run / "frames" / f"{frame}.jpg").convert("RGB")
    W = cell * len(items)
    cv = Image.new("RGB", (W, cell + cap_lines * cap_h + 12), (250, 250, 250)); dr = ImageDraw.Draw(cv)
    for k, (bb, lines) in enumerate(items):
        x1, y1, x2, y2 = bb
        cx, cy, sz = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 1.7
        crop = src.crop((int(cx - sz / 2), int(cy - sz / 2), int(cx + sz / 2), int(cy + sz / 2))).resize((cell - 10, cell - 10), Image.BICUBIC)
        cv.paste(crop, (k * cell + 5, 5))
        for j, (txt, bg, fg) in enumerate(lines):
            _label(dr, (k * cell + 10, cell + 4 + j * cap_h), txt, bg, 20, fg)
    return cv


def draw_state(src, out):
    """S3 요구조자 — 위: 전체 (번호 · 등급 색) · 아래: 사람마다 확대 + 정답 자세 · 등급 · 누움/앉음 확률 (등급에 쓰는 값) · 멈춘 시간"""
    from PIL import Image, ImageDraw
    d = json.load(open(src / "S3_a16.json")); f = next(x for x in d["frames"] if x["frame"] == "00381")
    best = {}
    for o in f["obs"]:                                    # 같은 배우에 박스가 둘이면 확신도 높은 것
        if o["actor"] and (o["actor"] not in best or o["conf"] > best[o["actor"]]["conf"]):
            best[o["actor"]] = o
    order = sorted(best.values(), key=lambda o: o["bbox"][0])
    reg = _region([o["bbox"] for o in order], margin=60)
    im, k = _crop("S3_a16", f["frame"], reg, out_w=1610)
    dr = ImageDraw.Draw(im)
    for n, o in enumerate(order, 1):
        x1, y1, x2, y2 = [(v - reg[i % 2]) * k for i, v in enumerate(o["bbox"])]
        col = GRADE_COL.get(o["grade"], (150, 150, 150))
        dr.rectangle([x1 - 4, y1 - 4, x2 + 4, y2 + 4], outline=col, width=5)
        _label(dr, (x1, y1 - 40), str(n), col, 26, (0, 0, 0) if o["grade"] == "CHECK" else (255, 255, 255))
    items = []
    for n, o in enumerate(order, 1):
        col = GRADE_COL.get(o["grade"], (150, 150, 150)); fg = (0, 0, 0) if o["grade"] == "CHECK" else (255, 255, 255)
        pose = POSE_KO.get(o["actor"].split("_")[1], o["actor"])
        items.append((o["bbox"], [(f"{n} 정답 {pose}", (40, 40, 40), (255, 255, 255)), (f"→ {o['grade']}", col, fg),
                                  (f"누움 {o['s_lying']:.2f} · 앉음 {o['s_sitting']:.2f}", (90, 90, 90), (255, 255, 255)),
                                  (f"멈춤 {o['still_s']} s", (90, 90, 90), (255, 255, 255))]))
    tl = _tiles("S3_a16", f["frame"], items, cap_lines=4)
    W = max(im.width, tl.width)
    canvas = Image.new("RGB", (W, 110 + im.height + tl.height + 44), (250, 250, 250))
    canvas.paste(im, (0, 110)); canvas.paste(tl, (0, 110 + im.height))
    dc = ImageDraw.Draw(canvas)
    dc.text((20, 14), "요구조자 상태 판단 (UE S3 · 16 m · 45° · 비행 38 s · 배포 worker 그대로 · 10 fps)", font=_font(32), fill=(20, 20, 20))
    dc.text((20, 60), "등급 = 박스 모양 자세 확률 + 멈춘 시간 · URGENT 누움≥0.7 & 멈춤≥10 s · HIGH 누움+앉음≥0.7 & 멈춤≥10 s · CHECK 애매 · NORMAL 그 밖",
            font=_font(22), fill=(70, 70, 70))
    dc.text((20, canvas.height - 34), "※ 정답 자세 = 배우 이름 · 6 「앉음」 애니메이션은 땅에 기대 누운 모습에 가깝다 (10-14 확인) · 5 누움은 추적이 끊겨 멈춤 시간이 2 s 로 다시 시작",
            font=_font(20), fill=(110, 110, 110))
    canvas.save(out / "07_UE_요구조자_상태_S3.jpg", quality=90)


def draw_track(src, out):
    """S2 추적 · 후보 — 네 시점 (정상 · 엇갈림 직전 · 엇갈린 뒤 후보 넘어감 · 다시 나타난 뒤 새 후보)"""
    from PIL import Image, ImageDraw
    d = json.load(open(src / "S2_a16.json"))
    letters, pal = {}, [(230, 60, 60), (40, 120, 230), (40, 170, 80), (180, 70, 200), (240, 150, 20), (20, 170, 170), (120, 120, 120)]
    shirt = {a: COLOR_KO[v["upper_color"]] for a, v in d["actors"].items()}
    panels = []
    for t_want, title in ((10.0, "(1) 10 s — 흰 옷 = 후보 A · 파랑 옷 = 후보 B (잘 이어짐)"),
                          (17.2, "(2) 17 s — 주황 옷 = 후보 C 등장 · 파랑 옷은 B ↔ D 로 번갈아 붙음 (쪼개짐)"),
                          (20.5, "(3) 20.5 s — 엇갈린 뒤 파랑 옷이 주황 옷의 후보 C 로 넘어감 (섞임)"),
                          (48.5, "(4) 48.5 s — 흰 옷이 건물 뒤에서 다시 나옴 → 새 후보 E (A 로 다시 잇지 못함)")):
        f = min(d["frames"], key=lambda x: abs(x["t"] - t_want))
        best = {}
        for o in f["obs"]:
            if o["actor"] and (o["actor"] not in best or o["conf"] > best[o["actor"]]["conf"]):
                best[o["actor"]] = o
        obs = list(best.values())
        reg = _region([o["bbox"] for o in obs], margin=130, min_w=760)
        im, k = _crop("S2_a16", f["frame"], reg, out_w=1280)
        dr = ImageDraw.Draw(im)
        for o in obs:
            c = letters.setdefault(o["cand"], chr(ord("A") + len(letters)))
            col = pal[(ord(c) - ord("A")) % len(pal)]
            x1, y1, x2, y2 = [(v - reg[i % 2]) * k for i, v in enumerate(o["bbox"])]
            dr.rectangle([x1 - 3, y1 - 3, x2 + 3, y2 + 3], outline=col, width=5)
            _label(dr, (x2 + 12, y1), f"후보 {c}", col, 26)
            _label(dr, (x2 + 12, y1 + 42), f"정답 {shirt[o['actor']]} 옷", (40, 40, 40), 20)
        cv = Image.new("RGB", (im.width, im.height + 56), (250, 250, 250)); cv.paste(im, (0, 56))
        ImageDraw.Draw(cv).text((14, 12), title, font=_font(26), fill=(20, 20, 20))
        panels.append(cv)
    cols = 2
    pw, ph = panels[0].width, max(p.height for p in panels)
    canvas = Image.new("RGB", (pw * cols + 20, 90 + ph * 2 + 20), (250, 250, 250))
    ImageDraw.Draw(canvas).text((14, 22), "추적 · 후보 묶기 (UE S2 · 16 m · 걷는 3명 · 건물 뒤 가림) — 정답은 「한 사람 = 한 글자」",
                                font=_font(32), fill=(20, 20, 20))
    for i, p in enumerate(panels):
        canvas.paste(p, ((i % cols) * (pw + 20), 90 + (i // cols) * ph))
    canvas.save(out / "08_UE_추적_후보_S2.jpg", quality=90)


def draw_color(src, out):
    """S4 상의 색 — 8색 8명 · 정답 vs 후보 누적 1순위"""
    from PIL import Image, ImageDraw
    d = json.load(open(src / "S4_a20.json")); f = next(x for x in d["frames"] if x["frame"] == "00611")
    obs = [o for o in f["obs"] if o["actor"]]
    reg = _region([o["bbox"] for o in obs], margin=110)
    im, k = _crop("S4_a20", f["frame"], reg)
    dr = ImageDraw.Draw(im)
    ok = 0
    for o in obs:
        truth = d["actors"][o["actor"]]["upper_color"]; pred = o["upper"]
        good = truth == pred; ok += good
        x1, y1, x2, y2 = [(v - reg[i % 2]) * k for i, v in enumerate(o["bbox"])]
        col = (40, 160, 70) if good else (220, 40, 40)
        dr.rectangle([x1, y1, x2, y2], outline=col, width=5)
        _label(dr, (x1, y2 + 8), f"정답 {COLOR_KO[truth]}", SW[truth], 22, (0, 0, 0) if truth in ("white", "yellow", "orange", "pink") else (255, 255, 255))
        _label(dr, (x1, y2 + 44), f"추정 {COLOR_KO.get(pred, '판정 불가')} {'O' if good else 'X'}", col, 22)
    canvas = Image.new("RGB", (im.width, im.height + 110), (250, 250, 250)); canvas.paste(im, (0, 110))
    dc = ImageDraw.Draw(canvas)
    dc.text((20, 14), f"상의 색 추정 (UE S4 · 20 m · 8색 8명 · 후보 누적 1순위) — 맞음 {ok}/{len(obs)}", font=_font(32), fill=(20, 20, 20))
    dc.text((20, 60), "틀림: 빨강→검정 (밝은 풀밭에서 노출 보정이 상의를 어둡힘 · 10-14 원인 확인) · 흰→회 · 주황→노랑 · 검정→회 (이웃 색)", font=_font(22), fill=(70, 70, 70))
    canvas.save(out / "09_UE_상의색_S4.jpg", quality=90)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
        run(out, sys.argv[3:])
    else:
        src, out = Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
        draw_state(src, out); draw_track(src, out); draw_color(src, out)
