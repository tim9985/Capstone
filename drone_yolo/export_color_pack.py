"""
export_color_pack.py — 상의 색 판정 꾸러미 (worker 1.3 · SigLIP 2 제로샷 + 남색 문턱 · 10-29 K 채택판)

  google/siglip2-base-patch16-224 (Apache 2.0) → 이미지 인코더 가중치만 (`vision.pt` · fp16 · 키 = vision_model.* 앞머리 뺌)
  + 12색 문장 임베딩 (`text.npy` · 「a photo of a person wearing a {색} shirt.」 · 남색 = navy blue) + `color_model.json` (설정 · 해시 · 평가)
  worker 는 transformers 없이 `color.SiglipColorProfiler` (plain torch) 로 읽는다
  검사: 꾸러미 판정 = color_lab 「F1 B1+남색문턱」 판정 (같은 크롭 · 1순위 일치율 · 확률 차)
실행  /home/se/venvs/state/bin/python export_color_pack.py [출력 폴더]   (기본 runs_person/_pack_color_siglip2 · git 밖)
"""
import hashlib, json, pickle, sys, time
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "deploy" / "vision_worker"))
import color_lab as L  # noqa: E402
from app.vision_core.color import COLORS, SiglipColorProfiler  # noqa: E402


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    proc, model = L.siglip(L.SIGLIP)
    vm = model.vision_model
    sd = {k: v.detach().half().cpu().contiguous() for k, v in vm.state_dict().items()}
    torch.save(sd, out / "vision.pt")
    T, scale, bias = L.siglip_text(L.TMPL1, False, L.SIGLIP)
    np.save(out / "text.npy", np.stack([T[c][0] for c in COLORS]).astype(np.float32))
    delta = L.navy_delta("upper", L.SIGLIP)
    vc = model.config.vision_config
    meta = {"name": "siglip2-zs-upper+navy", "source_model": L.SIGLIP, "license": "Apache-2.0", "created": time.strftime("%Y-%m-%d"),
            "colors": list(COLORS), "template": L.TMPL1[0], "prompt_names": {c: L.PROMPT[c] for c in COLORS},
            "logit_scale": scale, "logit_bias": bias, "navy_delta": delta, "image_size": 224, "upper_region": [0.10, 0.62],
            "preprocess": "BGR→RGB · PIL BILINEAR 224×224 · /255 · (x−0.5)/0.5",
            "vision": {"hidden": vc.hidden_size, "layers": vc.num_hidden_layers, "heads": vc.num_attention_heads, "patch": vc.patch_size, "eps": vc.layer_norm_eps},
            "eval_10_29": {"지표": "색 균형 1순위 (규칙 V10 → 이 꾸러미)", "UAV-Human 72px": "0.286 → 0.508", "NOMAD": "0.336 → 0.410 (한 장)",
                           "PETA": "0.270 → 0.558", "디관 사진": "0.314 → 0.744", "디관 영상": "0.422 → 0.885", "근거": "obsidian _학습 큐 10-29 K"},
            "sha256": {}}
    meta["sha256"] = {fn: sha(out / fn) for fn in ("vision.pt", "text.npy")}
    (out / "color_model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print("꾸러미", out, {k: round((out / k).stat().st_size / 1e6, 1) for k in ("vision.pt", "text.npy", "color_model.json")}, "MB · δ", delta)
    # ── 검사: 꾸러미 (plain torch) ↔ color_lab F1 (transformers) 같은 판정인가 ──
    prof = SiglipColorProfiler(out)
    for s in ("N", "D", "V", "U"):
        items = pickle.load(open(L.LAB / f"{s}.pkl", "rb"))[::3 if s in ("N", "U") else 1]
        ref = L.zero_shot(items, part="upper", navy=True)
        got, t0 = [], time.time()
        for i in range(0, len(items), 32):
            chunk = items[i:i + 32]
            for it in chunk:
                got.append(prof.profile(it["frame"], it["box"], pose=it["pose"]))
        ms = (time.time() - t0) * 1000 / len(items)
        agree = np.mean([g["status"] == "ok" and max(g["probs"], key=g["probs"].get) == COLORS[int(np.argmax(r))] for g, r in zip(got, ref)])
        dmax = max(abs(g["probs"][c] - float(r[k])) for g, r in zip(got, ref) if g["status"] == "ok" for k, c in enumerate(COLORS))
        print(f"  {s} {len(items)}장 · 1순위 일치 {agree:.4f} · 확률 최대 차 {dmax:.4f} · 한 명씩 {ms:.1f} ms", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "runs_person" / "_pack_color_siglip2")
