"""
fo_archangel.py — Archangel-Real 자세 판정 결과를 FiftyOne 으로 (10-07)
  1) 예측 (state 환경): python fo_archangel.py pred  → metrics/archangel_items.json (크롭마다 B0 · A3-0 · A4 겹 밖 예측 · 방위 · 고도 · 비)
  2) 올리기 (drone 환경): python fo_archangel.py load → 데이터셋 `archangel_posture`
  저장 뷰: b0_lying_as_standing (누운 사람을 서기로) · along_view_lying (시선 방향 0~15° 누움) · a4_fixed (B0 틀림 → A4 맞음) · kneeling
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "metrics" / "archangel_items.json"
CL = ("lying", "kneeling", "standing")


def pred():
    import posture_archangel_eval as E
    R = E.load(); g = [r["sess"] for r in R]
    P = {"b0": E.b0(R), "a30": E.oof(R, E.feats(R, "a2"), g), "a4": E.oof(R, E.feats(R, "a4"), g)}
    items = []
    for i, r in enumerate(R):
        items.append({"file": str(E.ARC / r["file"]), "pose": r["pose"], "label": r["label"], "alt": r["alt"], "radius": r["radius"],
                      "x_code": r["x_code"], "place": r["place"], "track": r["track"], "frame": int(r["frame"]),
                      "ratio": round(r["h"] / max(r["w"], 1), 3), "az": None if r["az"] is None else round(r["az"], 1),
                      **{k: [round(float(v), 4) for v in p[i]] for k, p in P.items()}})
    OUT.write_text(json.dumps(items))
    print(len(items))


def load():
    import fiftyone as fo
    from fiftyone import ViewField as F
    items = json.load(open(OUT))
    if "archangel_posture" in fo.list_datasets():
        fo.delete_dataset("archangel_posture")
    ds = fo.Dataset("archangel_posture", persistent=True)
    names = {"b0": ("lying", "sitting/kneeling", "standing"), "a30": CL, "a4": CL}
    S = []
    for it in items:
        s = fo.Sample(filepath=it["file"], tags=[it["pose"]], gt=fo.Classification(label=it["pose"]),
                      alt=it["alt"], radius=it["radius"], x_code=it["x_code"], video=it["place"], track=it["track"],
                      frame=it["frame"], ratio=it["ratio"], az=it["az"])
        for k in ("b0", "a30", "a4"):
            p = it[k]; j = max(range(3), key=lambda i: p[i])
            lab = names[k][j].split("/")[0] if k == "b0" else names[k][j]
            s[f"{k}_pose"] = fo.Classification(label=lab, confidence=p[j]); s[f"{k}_p_lying"] = p[0]
        S.append(s)
    ds.add_samples(S)
    ds.save_view("b0_lying_as_standing", ds.match((F("gt.label") == "lying") & (F("b0_pose.label") == "standing")), overwrite=True)
    ds.save_view("along_view_lying", ds.match((F("gt.label") == "lying") & (F("az") < 15)), overwrite=True)
    ds.save_view("a4_fixed", ds.match((F("gt.label") == "lying") & (F("b0_pose.label") == "standing") & (F("a4_pose.label") == "lying")), overwrite=True)
    ds.save_view("kneeling", ds.match(F("gt.label") == "kneeling"), overwrite=True)
    ds.info = {"설명": "Archangel-Real 자세 · b0 = 지금 판정기 (SARD+NOMAD 박스) · a30 / a4 = Archangel 회차 5겹 겹 밖 예측 · az = 시선 방향에서 위상 차 (누움만)"}
    ds.save()
    print(len(ds), "· B0 누움→서기", len(ds.load_saved_view("b0_lying_as_standing")), "· 시선 방향 누움", len(ds.load_saved_view("along_view_lying")))


if __name__ == "__main__":
    {"pred": pred, "load": load}[sys.argv[1]]()
