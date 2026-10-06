"""
state_judge_eval.py — YOLO 와 별개로 "도움이 필요한가" 를 판단하는 모델 짧은 비교 (2026-10-06 · 회의 10-07 자료)

  후보 (사용자 지정 3가지 중 받을 수 있는 것)
    qwen3vl     로컬 VLM Qwen3-VL-4B-Instruct — 크롭을 보고 예/아니오 · 자세 한 단어 (다음 글자 확률)
    clef_flash  Cloudflare Clef-Flash 9B — 정해진 답 + 확률 (Jev 와 같은 API) · 이미지 + JSON
    clef        Cloudflare Clef 27B (4비트) — 같음
    (OpenAI Decisions API — 제한 미리보기 · 키 · 승인 필요 → 지금 측정 못 함)

  시험 (few-test · 고정 시드)
    T1 크롭 판단  Okutama 비스듬 40/40/40 (누움/앉음/서기) · AI-Hub 산악5 수직 30/30/30 — 정답 박스 크롭 128 px → 256 px 로 키워 넣음
                 "도움 필요" 의 대리 정답 = 누움 (의식 없음 의심 신호)
    T2 상태 JSON  (Jev 자리) 파이프라인 1초 표본 (`runs_state/samples_q1_v2.json` · 탐지 박스 · 추적) 누움 60 · 앉음 30 · 서기 30
                 입력 = 박스 모양 자세 확률 · 무동작 초 · 탐지 확신도 → 규칙 점수와 같은 재료로 판정기만 바꿈
  지표: 누움 AUROC (P(도움 필요) · P(누움)) · 3자세 정확도 · 앉음 재현율 · 한 건 지연 (배치 1)

실행 (state 환경 · GPU 비었을 때): /home/se/venvs/state/bin/python state_judge_eval.py <방법> [--limit=N]
출력: metrics/state_judge_<방법>.json · 콘솔 표 · 정리 metrics/state_judge.md (방법마다 덧붙임)
"""
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
EVAL = BASE / "data" / "pose_eval"
CLASSES = ("lying", "sitting", "standing")
SCENE = ("Aerial search-and-rescue drone footage. The image is a crop around ONE person detected by a drone camera "
         "flying 16-30 m above ground, camera tilted 45-90 degrees down. The person may be small and blurry.")
Q_HELP = ("Does this person appear to need rescue assistance, for example collapsed or lying on the ground, "
          "motionless, or injured?")


# ── 시험 묶음 ──────────────────────────────────────────────────────────────
def t1_items(limit=None):
    rnd = random.Random(0)
    out = []
    for s, n in (("okutama_obl", 40), ("aihub_test", 30)):
        rows = list(csv.DictReader(open(EVAL / s / "crops.csv")))
        for c in CLASSES:
            pool = [r for r in rows if r["pose"] == c]
            for r in rnd.sample(pool, n):
                out.append({"set": s, "file": str(EVAL / s / r["file"]), "pose": c, "view": r["view"]})
    if limit:
        out = rnd.sample(out, limit)
    return out


def t2_items(limit=None):
    rnd = random.Random(1)
    S = json.load(open(BASE / "runs_state" / "samples_q1_v2.json"))
    S = [s for s in S if s.get("pose") in ("Lying", "Sitting", "Standing")]
    out = []
    for pose, n in (("Lying", 60), ("Sitting", 30), ("Standing", 30)):
        out += rnd.sample([s for s in S if s["pose"] == pose], n)
    if limit:
        out = rnd.sample(out, limit)
    return out


def t2_state(s):
    p = s["p"]
    return {"scene": "Drone search-and-rescue, oblique camera (45-60 deg). One tracked person, summarized by the vision system.",
            "posture_probability_from_box_shape": {"lying": round(p[0], 3), "sitting": round(p[1], 3), "standing": round(p[2], 3)},
            "motionless_seconds": s["still"], "detector_confidence": round(s["conf"], 3)}


def load_img(f):
    from PIL import Image
    return Image.open(f).convert("RGB").resize((256, 256), Image.BICUBIC)


# ── 판정기 ─────────────────────────────────────────────────────────────────
class QwenVL:
    name = "Qwen3-VL-4B-Instruct"

    def __init__(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        mid = "Qwen/Qwen3-VL-4B-Instruct"
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = AutoModelForImageTextToText.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda").eval()
        tok = self.proc.tokenizer
        self.ids = {w: tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No", "lying", "sitting", "standing")}
        assert len(set(self.ids.values())) == 5, self.ids

    def _next(self, img, q):
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
        inp = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                            return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            return self.model(**inp).logits[0, -1].float()

    def crop(self, it):
        img = load_img(it["file"])
        lg = self._next(img, f"{SCENE}\n{Q_HELP} Answer Yes or No.")
        help_p = self.torch.softmax(lg[[self.ids["Yes"], self.ids["No"]]], 0)[0].item()
        lg = self._next(img, f"{SCENE}\nIs the person lying, sitting, or standing? Answer with one word: lying, sitting, or standing.")
        pose = self.torch.softmax(lg[[self.ids[c] for c in CLASSES]], 0).tolist()
        return help_p, pose

    def state(self, s):
        return None                     # 텍스트 판정은 Clef 로 (Jev 자리)


class Clef:
    def __init__(self, repo, four_bit=False):
        import torch
        path = str(Path.home() / "models" / repo.split("/")[-1])      # aria2c 로 받음 (HF 기본 다운로더가 1.6 MB/s 로 멈춤)
        sys.path.insert(0, path)
        from joint_schema_model import collate_records, encode_record, load_release_model
        self.torch, self.enc, self.col = torch, encode_record, collate_records
        kw = {}
        if four_bit:                     # load_release_model 이 from_pretrained 인자를 그대로 넘긴다
            from transformers import BitsAndBytesConfig
            kw = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                            bnb_4bit_compute_dtype=torch.bfloat16)}
        self.model, self.proc = load_release_model(path, device="cuda", **kw)
        self.name = repo.split("/")[-1] + (" (4bit)" if four_bit else "")

    def _run(self, record):
        e = self.enc(self.proc.tokenizer, record, processor=self.proc)
        b = self.col([e], self.proc.tokenizer.pad_token_id, self.torch.device("cuda"))
        with self.torch.inference_mode():
            lg = self.model(b)[0]
        return {q.question_id: dict(zip(q.option_ids, l.float().softmax(-1).tolist())) for q, l in zip(e.questions, lg)}

    QS = {"needs_help": {"type": "noul", "instructions": Q_HELP},
          "posture": {"type": "choice", "instructions": "What is the person's posture?",
                      "criteria": {"lying": "Lying on the ground (on back, front or side).",
                                   "sitting": "Sitting, crouching or kneeling.",
                                   "standing": "Standing or walking upright."}}}

    def crop(self, it):
        r = self._run({"state": {"scene": SCENE, "view": it["view"]}, "images": [load_img(it["file"])], "questions": self.QS})
        return r["needs_help"].get("true", r["needs_help"].get(True)), [r["posture"][c] for c in CLASSES]

    def state(self, s):
        r = self._run({"state": t2_state(s), "questions": {"needs_help": self.QS["needs_help"]}})
        return r["needs_help"].get("true", r["needs_help"].get(True))


# ── 지표 ───────────────────────────────────────────────────────────────────
def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def main():
    meth = sys.argv[1]
    limit = next((int(a.split("=")[1]) for a in sys.argv[2:] if a.startswith("--limit=")), None)
    J = {"qwen3vl": lambda: QwenVL(), "clef_flash": lambda: Clef("Cloudflare/clef-flash"),
         "clef": lambda: Clef("Cloudflare/clef", four_bit=True)}[meth]()
    import torch
    res = {"방법": J.name, "T1": {}, "T2": None}
    rows, lat = [], []
    for it in t1_items(limit):
        t = time.time(); h, p = J.crop(it); torch.cuda.synchronize(); lat.append((time.time() - t) * 1000)
        rows.append({**it, "help": h, "p": p})
    for s in ("okutama_obl", "aihub_test", "전체"):
        R = [r for r in rows if s == "전체" or r["set"] == s]
        ly = [r for r in R if r["pose"] == "lying"]; nl = [r for r in R if r["pose"] != "lying"]
        pred = [CLASSES[int(np.argmax(r["p"]))] for r in R]
        res["T1"][s] = {"n": len(R), "도움필요_누움AUROC": round(auc([r["help"] for r in ly], [r["help"] for r in nl]), 3),
                        "P누움_AUROC": round(auc([r["p"][0] for r in ly], [r["p"][0] for r in nl]), 3),
                        "3자세_정확도": round(float(np.mean([a == r["pose"] for a, r in zip(pred, R)])), 3),
                        "재현율": {c: round(float(np.mean([a == c for a, r in zip(pred, R) if r["pose"] == c])), 3) for c in CLASSES}}
    res["T1"]["지연_ms_중앙"] = round(float(np.median(lat)), 1)
    T2 = t2_items(limit)
    if J.state(T2[0]) is not None:
        lat2, hp = [], []
        for s in T2:
            t = time.time(); hp.append(J.state(s)); torch.cuda.synchronize(); lat2.append((time.time() - t) * 1000)
        ly = [h for h, s in zip(hp, T2) if s["pose"] == "Lying"]; nl = [h for h, s in zip(hp, T2) if s["pose"] != "Lying"]
        rly = [s["score"] for s in T2 if s["pose"] == "Lying"]; rnl = [s["score"] for s in T2 if s["pose"] != "Lying"]
        res["T2"] = {"n": len(T2), "도움필요_누움AUROC": round(auc(ly, nl), 3), "규칙점수_누움AUROC (같은 표본)": round(auc(rly, rnl), 3),
                     "지연_ms_중앙": round(float(np.median(lat2)), 1)}
    res["GPU_최대_GB"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
    out = BASE / "metrics" / f"state_judge_{meth}{'_lim' if limit else ''}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    if not limit:                       # 문항별 예측 — FiftyOne 시각 점검용 (fo_state_judge.py)
        items = [{"file": r["file"], "set": r["set"], "pose": r["pose"], "help": round(float(r["help"]), 4),
                  "p": [round(float(v), 4) for v in r["p"]]} for r in rows]
        (BASE / "metrics" / f"state_judge_{meth}_items.json").write_text(json.dumps(items, ensure_ascii=False))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if not limit:
        with open(BASE / "metrics" / "state_judge.md", "a") as f:
            t = res["T1"]
            f.write(f"| {J.name} | {t['okutama_obl']['도움필요_누움AUROC']} / {t['okutama_obl']['P누움_AUROC']} | "
                    f"{t['aihub_test']['도움필요_누움AUROC']} / {t['aihub_test']['P누움_AUROC']} | {t['전체']['3자세_정확도']} | "
                    f"{t['전체']['재현율']['sitting']} | {t['지연_ms_중앙']} | "
                    f"{res['T2']['도움필요_누움AUROC'] if res['T2'] else '—'} | {res['GPU_최대_GB']} |\n")


if __name__ == "__main__":
    main()
