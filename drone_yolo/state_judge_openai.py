"""
state_judge_openai.py — OpenAI 쪽 "도움 필요" 판정 (state_judge_eval.py 의 시험 2 와 같은 문항 · 2026-10-07)

  Decisions API 는 제한 미리보기라 403 ("Decision API is not enabled for this user", 10-07 확인)
  → 같은 엔진 GPT-6 Luna 를 일반 Chat API 로 · 추론 끔 (reasoning_effort none) · 한 단어 Yes/No · 첫 글자 확률로 P(Yes)
  시험 2 (상태 JSON · 숫자만 · 이미지 안 보냄) 만 — 크롭 이미지는 데이터 약관 (AI-Hub · Okutama) 확인 전 안 보냄
  키: ~/.config/openai/api_key (권한 600 · 채팅 · 커밋 금지)

실행: /home/se/venvs/state/bin/python state_judge_openai.py [모델=gpt-6-luna]
출력: metrics/state_judge_openai_<모델>.json
"""
import json
import math
import pathlib
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from state_judge_eval import BASE, Q_HELP, auc, t2_items, t2_state

KEY = (pathlib.Path.home() / ".config" / "openai" / "api_key").read_text().strip()


def ask(model, state, tries=4):
    body = {"model": model, "reasoning_effort": "none", "logprobs": True, "top_logprobs": 5, "max_completion_tokens": 16,
            "messages": [{"role": "system", "content": "You are the decision layer of a drone search-and-rescue system."},
                         {"role": "user", "content": f"State (JSON): {json.dumps(state)}\n{Q_HELP}\nAnswer with exactly one word: Yes or No."}]}
    for k in range(tries):
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", method="POST", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)
            dt = (time.time() - t) * 1000
            tok = out["choices"][0]["logprobs"]["content"][0]
            cands = {x["token"].strip().lower(): math.exp(x["logprob"]) for x in tok["top_logprobs"]}
            if "yes" in cands:
                p = cands["yes"] if "no" not in cands else cands["yes"] / (cands["yes"] + cands["no"])
            elif "no" in cands:
                p = 1 - cands["no"]
            else:
                p = float("nan")
            return p, dt
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and k + 1 < tries:
                time.sleep(2 ** k); continue
            raise RuntimeError(f"{e.code} {e.read().decode()[:200]}")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-6-luna"
    T2 = t2_items()
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda s: ask(model, t2_state(s)), T2))
    hp = [r[0] for r in res]
    ly = [h for h, s in zip(hp, T2) if s["pose"] == "Lying"]; nl = [h for h, s in zip(hp, T2) if s["pose"] != "Lying"]
    rly = [s["score"] for s in T2 if s["pose"] == "Lying"]; rnl = [s["score"] for s in T2 if s["pose"] != "Lying"]
    out = {"방법": f"OpenAI {model} (Decisions API 대신 · 추론 끔 · Yes/No 확률)", "T2": {
        "n": len(T2), "도움필요_누움AUROC": round(auc(ly, nl), 3), "규칙점수_누움AUROC (같은 표본)": round(auc(rly, rnl), 3),
        "P(Yes)≥0.5 비율 — 누움 / 안 누움": [round(float(np.mean([h >= 0.5 for h in ly])), 3), round(float(np.mean([h >= 0.5 for h in nl])), 3)],
        "지연_ms_중앙": round(float(np.median([r[1] for r in res])), 1), "확률 못 읽음": int(sum(np.isnan(hp)))}}
    (BASE / "metrics" / f"state_judge_openai_{model}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
