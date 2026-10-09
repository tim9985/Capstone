"""
fo_raw.py — 서버의 raw 원본 데이터셋 전부를 FiftyOne 에 (`raw/<이름>` · 데이터셋 태그 raw · 10-08)

  원칙  사진은 복사하지 않고 원래 위치를 가리킨다 · 라벨의 속성은 전부 남긴다
        박스 속성 (자세 · 행동 · 가림 · 보임) = Detection 속성 · 사진 속성 (고도 · 각도 · 날씨 · 장소 · 분할 …) = 샘플 필드 + 일부 태그
        사람은 label="person" 으로 통일하고 원래 클래스 이름은 `orig_class` · 자세는 `pose` (데이터셋마다 이름이 달라 원래 값 그대로)
  영상  Archangel (real · mannequin) — ffmpeg 가 없어 OpenCV 로 영상 정보를 직접 넣는다 (프레임 번호 1부터)
  썸네일 `--thumbs` → data/fo_thumbs/<이름>/ (긴 변 640) · App 격자가 빨라짐 (원본은 그대로 열람)
  ⚠ 연구용 · 약관 제한 데이터 — 서버 안 팀 열람용 · 51 화면을 팀 밖에 공유하지 않는다

실행: /home/se/miniconda3/envs/drone/bin/python fo_raw.py [이름 …] [--force] [--thumbs]
      이름: aihub182 nomad wisard nii_cu okutama auair visdrone unicamp_uav heridal ladd cyprus_sop c2a sard2 archangel_real archangel_mannequin
"""
import csv
import glob
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

RAW = Path("/home/se/JupyterLAB/Capstone/data/raw")
THUMBS = Path("/home/se/JupyterLAB/Capstone/data/fo_thumbs")
NADIR = set("2.2.2 1.1.8 1.1.4 2.2.10 2.2.5 2.2.1 1.1.11 1.1.7 2.2.7 1.1.5 2.2.4 2.2.3 1.1.9 2.2.6 2.2.8".split())   # okutama_motion 과 같음


def fo():
    import fiftyone
    return fiftyone


def det(label, x, y, w, h, **attrs):
    """정규화 좌상단 x · y · 폭 · 높이"""
    F = fo()
    return F.Detection(label=label, bounding_box=[max(0.0, x), max(0.0, y), w, h], **{k: v for k, v in attrs.items() if v is not None})


def yolo_dets(txt, names=None, extra=None):
    out = []
    if not os.path.exists(txt):
        return out
    for ln in open(txt):
        p = ln.split()
        if len(p) < 5:
            continue
        c = int(float(p[0])); cx, cy, w, h = map(float, p[1:5])
        name = names[c] if names and c < len(names) else str(c)
        a = extra(p) if extra else {}
        out.append(det(a.pop("label", "person"), cx - w / 2, cy - h / 2, w, h, orig_class=name, **a))
    return out


def sample(path, dets=None, fields=None, tags=None, field="ground_truth"):
    F = fo()
    s = F.Sample(filepath=str(path), tags=list(tags or []))
    if dets is not None:
        s[field] = F.Detections(detections=dets)
        s["n_objects"] = len(dets)
    for k, v in (fields or {}).items():
        if v is not None:
            s[k] = v
    return s


# ── 데이터셋별 ──────────────────────────────────────────────────────────────

def aihub182():
    """AI-Hub 182 조난자 수색 (산악 · 저수지 · 평지) — 4점 폴리곤 → 박스 · person_pose · 고도 · 각도 · 날씨 · 시간"""
    lab = {Path(p).stem: p for p in glob.glob(str(RAW / "AIHub182/**/라벨링데이터/**/*.json"), recursive=True)}
    for img in sorted(glob.glob(str(RAW / "AIHub182/**/원천데이터/**/*.jpg"), recursive=True)):
        j = lab.get(Path(img).stem)
        if not j:
            yield sample(img, [], {"labeled": False}, ["no_label"]); continue
        d = json.load(open(j, encoding="utf-8", errors="replace"))
        W, H = d.get("metadata", {}).get("width", 3840), d.get("metadata", {}).get("height", 2160)
        M = d.get("Metadata", {})
        dets = []
        for a in d.get("annotations", []):
            pts = a.get("points") or []
            if len(pts) < 3:
                continue
            xs, ys = [q[0] for q in pts], [q[1] for q in pts]
            pose = (a.get("attributes") or {}).get("person_pose")
            dets.append(det("person", min(xs) / W, min(ys) / H, (max(xs) - min(xs)) / W, (max(ys) - min(ys)) / H,
                            orig_class=a.get("label"), pose=pose))
        pp = d.get("parent_path", "")
        parts = [x for x in pp.split("/") if x]
        terrain = parts[2] if len(parts) > 2 else None
        split = "train" if "1.Training" in img else "val" if "2.Validation" in img else None
        yield sample(img, dets, {"labeled": True, "split": split, "altitude_m": M.get("altitude"), "angle_deg": M.get("angle"),
                                 "weather": M.get("weather"), "time_of_day": M.get("time"), "date": M.get("date"),
                                 "place": terrain, "site": parts[1] if len(parts) > 1 else None, "mission": M.get("mission-id")},
                     [t for t in (split, f"alt_{M.get('altitude')}m", f"angle_{M.get('angle')}", terrain) if t])


def nomad():
    """NOMAD — 배우 100명 × 고도 (a10~a90) · 박스마다 보임 % · 프레임 행동 (Walking · Hiding · Laying …) · 배우 정보"""
    ann = {a["file_name"]: a for a in json.load(open(RAW / "NOMAD/annotations.json"))}
    meta = {m["id"]: m for m in json.load(open(RAW / "NOMAD/metadata.json"))}
    acts = {}
    for f in ("activityLabels.json", "activityLabels_waterRoutine.json"):
        for a in json.load(open(RAW / "NOMAD" / f)):
            for alt, d in a["labels"].items():
                for act, rng in d.items():
                    for r in rng:                     # 원본에 오타가 섞여 있다 ("0174.0334" · "1320}") → 숫자만 뽑는다
                        nums = [int(x) for x in re.findall(r"\d+", json.dumps(r))]
                        if nums:
                            acts.setdefault((a["id"], alt), []).append((nums[0], nums[-1], act))
    for img in sorted(glob.glob(str(RAW / "NOMAD/images/*/*/*.jpg"))):
        name = os.path.basename(img)
        m = re.match(r"Actor(\d+)_a(\d+)_f(\d+)", name)
        actor, alt, fr = int(m.group(1)), m.group(2), int(m.group(3))
        a = ann.get(name)
        dets, vis_all = [], []
        if a:
            W, H = a["width"], a["height"]
            for b in a["annotations"]:
                x, y, w, h = b["bbox"]
                vis = b.get("visibility")
                vis = int(vis) if str(vis).isdigit() else vis
                if isinstance(vis, int):
                    vis_all.append(vis)
                dets.append(det("person", x / W, y / H, w / W, h / H, visibility=vis))
        else:
            dets = yolo_dets(img.replace("/images/", "/labels/").replace(".jpg", ".txt"))
        act = next((t for s0, s1, t in acts.get((actor, alt), []) if s0 <= fr <= s1), None)
        md = meta.get(actor, {})
        env, dem = md.get("environmental", {}), md.get("demographic", {})
        outfit = dem.get("outfit") or {}
        yield sample(img, dets, {"actor": actor, "altitude_tag": f"a{alt}", "frame": fr, "activity": act,
                                 "location": env.get("location"), "weather": (env.get("weather") or {}).get("descriptor"),
                                 "time_hh": env.get("time[HH]"), "gender": dem.get("gender"), "age": dem.get("age"),
                                 "outfit": json.dumps(outfit, ensure_ascii=False)[:300] if outfit else None,
                                 "min_visibility": min(vis_all, default=None)},
                     [f"a{alt}"] + ([act] if act else []))


def wisard():
    """WiSARD — 비행 폴더 (날짜 · 장소 · 기체 · VIS/IR)"""
    for img in sorted(glob.glob(str(RAW / "WiSARD/*/*.jpg"))):
        flight = Path(img).parent.name
        m = re.match(r"(\d{6})_(.+?)_(VIS|IR|FLIR)", flight)
        mod = "IR" if ("_IR" in flight or "FLIR_IR" in flight) else "VIS"
        yield sample(img, yolo_dets(img[:-4] + ".txt"), {"flight": flight, "date": m.group(1) if m else None,
                                                          "site_platform": m.group(2) if m else None, "modality": mod},
                     [mod])


def nii_cu():
    """NII-CU (일본 · 높은 곳 · 4K) — rgb-t (px 박스 + 표시 3개 · 원본 설명 없음 → flag_1~3) 우선 · yolo/ 에만 있는 프레임은 YOLO 라벨"""
    seen = set()
    for img in sorted(glob.glob(str(RAW / "NII-CU/rgb-t/images/rgb/*/*.jpg"))):
        split = Path(img).parent.name; stem = Path(img).stem; seen.add(stem)
        lab = RAW / "NII-CU/rgb-t/labels" / split / f"{stem}.txt"
        dets = []
        if lab.exists():
            for ln in open(lab):
                q = ln.split()
                if len(q) < 4:
                    continue
                x1, y1, x2, y2 = map(float, q[:4]); fl = [int(v) for v in q[4:7]]
                dets.append(det("person", x1 / 3840, y1 / 2160, (x2 - x1) / 3840, (y2 - y1) / 2160,
                                **{f"flag_{i + 1}": v for i, v in enumerate(fl)}))
        m = re.search(r"(flight\d+)", stem)
        yield sample(img, dets, {"part": "rgb-t", "split": split, "flight": m.group(1) if m else None}, ["rgb-t", split])
    for img in sorted(glob.glob(str(RAW / "NII-CU/yolo/images/*.jpg"))):
        stem = Path(img).stem
        if stem in seen:
            continue
        m = re.search(r"(flight\d+)", stem)
        yield sample(img, yolo_dets(str(RAW / "NII-CU/yolo/labels" / f"{stem}.txt")), {"part": "yolo", "flight": m.group(1) if m else None}, ["yolo"])


def okutama():
    """Okutama-Action — 추출 프레임 1280×720 (학습 · 시험) · 박스마다 행동 · 가림 · 잃음 · 자동 보간 · 영상 · 시간대 · 수직/비스듬"""
    for base, labdir, split in ((RAW / "okutama", RAW / "okutama/Labels/MultiActionLabels/3840x2160", "train"),
                                (RAW / "okutama/TestSetFrames", RAW / "okutama/TestSetFrames/Labels/MultiActionLabels/3840x2160", "test")):
        for vdir in sorted(glob.glob(str(base / "Drone*/*/Extracted-Frames-1280x720/*"))):
            vid = os.path.basename(vdir); tod = Path(vdir).parent.parent.name; drone = Path(vdir).parent.parent.parent.name
            labs = defaultdict(list)
            lf = labdir / f"{vid}.txt"
            if lf.exists():
                for ln in open(lf):
                    p = ln.split()
                    if len(p) < 10:
                        continue
                    tid, x1, y1, x2, y2, fr, lost, occ, gen = map(int, p[:9])
                    q = re.findall(r'"([^"]*)"', ln)
                    if lost:
                        continue
                    labs[fr].append(det("person", x1 / 3840, y1 / 2160, (x2 - x1) / 3840, (y2 - y1) / 2160, track=tid,
                                        occluded=bool(occ), generated=bool(gen), orig_class=q[0] if q else None,
                                        actions=",".join(q[1:]) if len(q) > 1 else None, pose=q[1] if len(q) > 1 else None))
            for img in sorted(glob.glob(f"{vdir}/*.jpg"), key=lambda s: int(Path(s).stem)):
                fr = int(Path(img).stem)
                yield sample(img, labs.get(fr, []), {"video": vid, "frame": fr, "split": split, "time_of_day": tod, "drone": drone,
                                                    "view": "nadir" if vid in NADIR else "oblique", "labeled": lf.exists()},
                             [split, "nadir" if vid in NADIR else "oblique", tod])


def auair():
    """AU-AIR — 도로 · 8 클래스 · 기체 위치 · 고도 · 자세 (phi · theta · psi)"""
    d = json.load(open(RAW / "auair/annotations.json"))
    cats = d["categories"]
    for a in d["annotations"]:
        img = RAW / "auair/images" / a["image_name"]
        if not img.exists():
            continue
        W, H = a.get("image_width:", 1920.0), a.get("image_height", 1080.0)
        dets = [det("person" if cats[b["class"]] == "Human" else cats[b["class"]].lower(), b["left"] / W, b["top"] / H, b["width"] / W, b["height"] / H,
                    orig_class=cats[b["class"]]) for b in a.get("bbox", [])]
        yield sample(img, dets, {"platform": a.get("platform"), "altitude_m": round(a["altitude"] / 1000, 2) if a.get("altitude") is not None else None,
                                 "lat": a.get("latitude"), "lon": a.get("longtitude"), "phi": a.get("angle_phi"), "theta": a.get("angle_theta"),
                                 "psi": a.get("angle_psi"), "n_person": sum(x.label == "person" for x in dets)},
                     ["has_person"] if any(x.label == "person" for x in dets) else ["no_person"])


VISDRONE = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]


def visdrone():
    """VisDrone-DET (YOLO 변환본 · 10 클래스) — pedestrian · people 는 person"""
    for img in sorted(glob.glob(str(RAW / "VisDrone/images/*/*.jpg"))):
        split = Path(img).parent.name
        lab = img.replace("/images/", "/labels/")[:-4] + ".txt"

        def ex(p):
            c = VISDRONE[int(float(p[0]))] if int(float(p[0])) < 10 else p[0]
            return {"label": "person" if c in ("pedestrian", "people") else c}
        yield sample(img, yolo_dets(lab, VISDRONE, ex), {"split": split}, [split])


def unicamp_uav():
    for img in sorted(glob.glob(str(RAW / "unicamp_uav/*/images/*.jpg"))):
        split = Path(img).parent.parent.name
        yield sample(img, yolo_dets(img.replace("/images/", "/labels/")[:-4] + ".txt"), {"split": split, "clip": Path(img).stem.rsplit("_", 1)[0]}, [split])


def heridal():
    """HERIDAL — VOC (사람) · 분할은 이미지 집합 파일"""
    root = RAW / "HERIDAL/heridal_keras_retinanet_voc"
    sets = {}
    for f in glob.glob(str(root / "ImageSets/Main/*.txt")):
        for ln in open(f):
            sets[ln.strip()] = Path(f).stem
    for img in sorted(glob.glob(str(root / "JPEGImages/*.jpg"))):
        stem = Path(img).stem
        xmlp = root / "Annotations" / f"{stem}.xml"
        dets = []
        if xmlp.exists():
            r = ET.parse(xmlp).getroot()
            sz = r.find("size")
            if sz is not None and sz.find("width") is not None:
                W, H = float(sz.find("width").text), float(sz.find("height").text)
            else:                                            # 일부 XML 에 크기가 없다 → 사진 머리에서
                from PIL import Image
                W, H = Image.open(img).size
            for o in r.findall("object"):
                b = o.find("bndbox"); x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
                n = o.find("name").text
                dets.append(det("person" if n.lower() in ("person", "human") else n, x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H, orig_class=n))
        yield sample(img, dets, {"split": sets.get(stem), "place_code": re.sub(r"_\d+$", "", stem)}, [sets[stem]] if stem in sets else [])


def ladd():
    """LADD (러시아 숲 수색 · GPL-3.0) — 이미지 번호 = 라벨 번호"""
    for img in sorted(glob.glob(str(RAW / "ladd/ds0/img/*.jpg")), key=lambda s: int(Path(s).stem)):
        n = int(Path(img).stem)
        part = "front70" if n < 955 else "margin" if n < 1005 else "back30_eval"            # make_v10_data 와 같은 나눔
        yield sample(img, yolo_dets(str(RAW / "ladd/yolo_labels" / f"{n}.txt")), {"index": n, "part": part}, [part])


def cyprus_sop():
    for img in sorted(glob.glob(str(RAW / "cyprus_sop/Images/*/*.jpg"))):
        split = Path(img).parent.name
        yield sample(img, yolo_dets(str(RAW / "cyprus_sop/Annotations/Yolo" / split / (Path(img).stem + ".txt"))), {"split": split}, [split])


C2A_POSE = ["bent", "kneeling", "lying", "sitting", "upright"]


def c2a():
    """C2A (합성 · 재난 배경 + 사람 붙여넣기) — 박스마다 자세 (6번째 값)"""
    pl = RAW / "C2A/new_dataset3/All labels with Pose information/labels"
    for img in sorted(glob.glob(str(RAW / "C2A/new_dataset3/*/images/*"))):
        split = Path(img).parent.parent.name
        lab = pl / (Path(img).stem + ".txt")

        def ex(p):
            return {"pose": C2A_POSE[int(float(p[5]))] if len(p) > 5 and int(float(p[5])) < 5 else None}
        scene = re.sub(r"_image\d+.*$", "", Path(img).stem)
        yield sample(img, yolo_dets(str(lab), None, ex), {"split": split, "scene": scene}, [split, scene])


SARD = ["Running", "Walking", "laying_down", "not_defined", "seated", "stands"]


def sard2():
    for img in sorted(glob.glob(str(RAW / "sard2/search-and-rescue-2/*/images/*.jpg"))):
        split = Path(img).parent.parent.name

        def ex(p):
            return {"pose": SARD[int(float(p[0]))] if int(float(p[0])) < 6 else None}
        yield sample(img, yolo_dets(img.replace("/images/", "/labels/")[:-4] + ".txt", SARD, ex), {"split": split}, [split])


# ── 영상 (Archangel) ──

def vmeta(path):
    import cv2
    F = fo()
    c = cv2.VideoCapture(str(path))
    m = F.VideoMetadata(frame_rate=c.get(cv2.CAP_PROP_FPS) or 10.0, total_frame_count=int(c.get(cv2.CAP_PROP_FRAME_COUNT)),
                        frame_width=int(c.get(3)), frame_height=int(c.get(4)), size_bytes=os.path.getsize(path), mime_type="video/mp4")
    c.release()
    return m


def archangel_real():
    """Archangel-Real — 실제 사람 · 고도 15~50 m · 원 궤도 · 영상마다 프레임 박스 (자세 standing · walking · kneeling · lying down · crawling)"""
    F = fo()
    boxes = defaultdict(lambda: defaultdict(list))
    with zipfile.ZipFile(RAW / "archangel/Archangel_annotations.zip") as z:
        for n in z.namelist():
            if not n.endswith(".json"):
                continue
            for v in json.loads(z.read(n))["data"]:
                m = re.match(r"(.+_EO)-(\d+)-(\d+)\.mp4$", v["name"])
                if not m:
                    continue
                vid, a = m.group(1), int(m.group(2))
                for o in v["objectOccurrences"]:
                    lab = o["labels"][0]["name"] if o["labels"] else o["name"]
                    for t in o["tracks"]:
                        for an in t["annotations"]:
                            c = an["coords"]
                            boxes[vid][a - 1 + an["position"]].append((c["xMin"], c["yMin"], c["xMax"], c["yMax"], lab, o["id"][:8]))
    for mp4 in sorted(glob.glob(str(RAW / "archangel/real/*.mp4"))):
        vid = Path(mp4).stem
        meta = vmeta(mp4); W, H = meta.frame_width, meta.frame_height
        m = re.match(r"AA_BP_(\d+)_([\d-]+)_(\d+)_(\d+)_(moving_)?", vid)
        s = F.Sample(filepath=mp4, metadata=meta, tags=["moving"] if m and m.group(5) else ["orbit"])
        if m:
            s["session"], s["x_code"], s["altitude_m"], s["radius_m"] = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        for fr, bs in boxes.get(vid, {}).items():
            s.frames[fr + 1]["ground_truth"] = F.Detections(detections=[
                det("vehicle" if "vehicle" in lab else "person", x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H,
                    pose=lab.split("- ")[-1].strip() if "- " in lab else lab, orig_class=lab, track=tid) for x1, y1, x2, y2, lab, tid in bs])
        s["n_labeled_frames"] = len(boxes.get(vid, {}))
        yield s


def archangel_mannequin():
    """Archangel-Mannequin — 마네킹 (서기 · 무릎 · 누움) + 차량 · 10초 조각 · 프레임마다 박스 · 보임"""
    F = fo()
    for js in sorted(glob.glob(str(RAW / "archangel/mannequin/*/*.json"))):
        mp4 = js[:-5] + ".mp4"
        if not os.path.exists(mp4):
            continue
        meta = vmeta(mp4); W, H = meta.frame_width, meta.frame_height
        m = re.search(r"DTRA_Trial-(\d+)_CIR_VIS_([\d-]+)m_(\d+)deg_cam_10sec-(\d+)", js)
        s = F.Sample(filepath=mp4, metadata=meta, tags=[f"alt_{m.group(2)}m", f"cam_{m.group(3)}deg"] if m else [])
        if m:
            s["trial"], s["altitude_tag"], s["cam_deg"], s["start_frame"] = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        for fi, bs in enumerate(json.load(open(js))):
            if bs:
                s.frames[fi + 1]["ground_truth"] = F.Detections(detections=[
                    det("vehicle" if "vehicle" in b["category"] else "person", b["x"] / W, b["y"] / H, b["width"] / W, b["height"] / H,
                        pose=b["category"].split(" - ")[-1].strip(), orig_class=b["category"], visibility=b.get("visibility"),
                        track=b.get("id", "")[:8], annotated_by=b.get("annotated_by")) for b in bs])
        yield s


LOADERS = {"aihub182": aihub182, "nomad": nomad, "wisard": wisard, "nii_cu": nii_cu, "okutama": okutama, "auair": auair,
           "visdrone": visdrone, "unicamp_uav": unicamp_uav, "heridal": heridal, "ladd": ladd, "cyprus_sop": cyprus_sop,
           "c2a": c2a, "sard2": sard2, "archangel_real": archangel_real, "archangel_mannequin": archangel_mannequin}
INFO = {"aihub182": "AI-Hub 182 조난자 수색 (한국 · 수직 90° 위주 · 연구용 · 재배포 금지)",
        "nomad": "NOMAD (미국 · 배우 100 · 고도 10~90 ft · 연구용)", "wisard": "WiSARD (미국 · 수색 비행 · VIS/IR · 연구용)",
        "nii_cu": "NII-CU (일본 · 높은 곳 45° 근처)", "okutama": "Okutama-Action (일본 공원 · 행동 · 45°/90° · 연구용 · 평가 전용)",
        "auair": "AU-AIR (덴마크 도로 · CC BY-NC-SA)", "visdrone": "VisDrone-DET (중국 도시)", "unicamp_uav": "Unicamp-UAV (브라질)",
        "heridal": "HERIDAL (크로아티아 산악)", "ladd": "LADD (러시아 숲 · GPL-3.0)", "cyprus_sop": "키프로스 SOP (CC BY 4.0 · 수직)",
        "c2a": "C2A (합성 재난 장면 · 자세)", "sard2": "SARD (Roboflow · CC BY 4.0 · 자세 6종)",
        "archangel_real": "Archangel-Real (실제 사람 · 15~50 m 원 궤도 · 자세 · 연구용)", "archangel_mannequin": "Archangel-Mannequin (마네킹 자세 · 차량)"}


def build(name, force=False, thumbs=False):
    F = fo()
    dn = os.environ.get("FO_RAW_PREFIX", "raw/") + name
    if F.dataset_exists(dn):
        if not force:
            print(f"{dn}: 있음 — 건너뜀 (--force 로 다시)", flush=True); return F.load_dataset(dn)
        F.delete_dataset(dn)
    ds = F.Dataset(dn); ds.persistent = True; ds.tags = ["raw", name]
    ds.info = {"source": INFO.get(name, name), "root": str(RAW), "built": time.strftime("%Y-%m-%d %H:%M"), "note": "원본을 가리킴 (복사 안 함) · 팀 열람용"}
    buf, n, t0 = [], 0, time.time()
    lim = int(os.environ.get("FO_RAW_LIMIT", "0"))
    for k, s in enumerate(LOADERS[name]()):
        if lim and k >= lim:
            break
        buf.append(s)
        if len(buf) >= 2000:
            ds.add_samples(buf, progress=False); n += len(buf); buf = []
            print(f"  {dn}: {n:,} ({time.time() - t0:.0f}s)", flush=True)
    if buf:
        ds.add_samples(buf, progress=False); n += len(buf)
    ds.save()
    views(ds, name)
    if thumbs and ds.media_type == "image":
        make_thumbs(ds, name)
    print(f"{dn}: 끝 — {len(ds):,} 샘플 · {time.time() - t0:.0f}s", flush=True)
    return ds


def views(ds, name):
    """속성별 저장 보기 + 사이드바 묶음"""
    from fiftyone import ViewField as Fv
    F = fo()
    try:
        ds.add_dynamic_sample_fields()                       # 박스 속성 (pose · visibility · actions …) 을 사이드바 필터에 보이게
        if ds.media_type == "video":
            ds.add_dynamic_frame_fields()
        if ds.media_type == "image" and ds.has_field("ground_truth"):
            ds.save_view("with_objects", ds.match(Fv("n_objects") > 0), description="박스가 하나 이상", overwrite=True)
            ds.save_view("no_objects", ds.match(Fv("n_objects") == 0), description="박스 없음 (음성 · 라벨 없음)", overwrite=True)
            poses = [p for p in ds.distinct("ground_truth.detections.pose") if p] if ds.has_field("ground_truth.detections.pose") else []
            for p in poses[:12]:
                slug = re.sub(r"[^a-z0-9]+", "_", str(p).lower()).strip("_") or "x"
                ds.save_view(f"pose_{slug}", ds.filter_labels("ground_truth", Fv("pose") == p), description=f"자세 = {p}", overwrite=True)
            ds.app_config.color_scheme = F.ColorScheme(color_by="value", fields=[{"path": "ground_truth", "colorByAttribute": "pose"}]) if poses else None
        ds.app_config.sidebar_groups = None
        ds.save()
    except Exception as e:
        print(f"  보기 만들기 실패 ({type(e).__name__}: {e})", flush=True)


def _thumb(job):
    src, dst = job
    import cv2
    if os.path.exists(dst):
        return dst
    img = cv2.imread(src, cv2.IMREAD_REDUCED_COLOR_2)       # JPEG 을 1/2 로 바로 디코드 (빠름)
    if img is None:
        img = cv2.imread(src)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > 640:
        img = cv2.resize(img, (640, max(1, round(h * 640 / w))), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return dst


def make_thumbs(ds, name):
    """긴 변 640 썸네일 — 원본 폴더 구조 그대로 data/fo_thumbs/ 아래 · 프로세스 10개 · 원본 filepath 는 그대로"""
    from multiprocessing import Pool
    ids, paths = ds.values("id"), ds.values("filepath")
    jobs = [(p, str(THUMBS / os.path.relpath(p, RAW)).rsplit(".", 1)[0] + ".jpg") for p in paths]
    with Pool(10) as pool:
        out = pool.map(_thumb, jobs, chunksize=64)
    ds.set_values("thumbnail_path", dict(zip(ids, out)), key_field="id")
    ds.app_config.media_fields = ["filepath", "thumbnail_path"]; ds.app_config.grid_media_field = "thumbnail_path"; ds.save()
    print(f"  썸네일 {sum(o is not None for o in out):,}/{len(out):,} → {THUMBS}", flush=True)


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(LOADERS)
    for nm in names:
        try:
            build(nm, force="--force" in sys.argv, thumbs="--thumbs" in sys.argv)
        except Exception as e:
            import traceback
            print(f"raw/{nm}: ❌ {type(e).__name__}: {e}", flush=True); traceback.print_exc()
