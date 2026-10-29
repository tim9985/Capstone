"""
tests_color_siglip.py — worker 1.3 상의 색 판정 교체 시험 (10-29 · SigLIP 2 제로샷 + 남색 문턱)

  **실제 배포 판** (샌드박스 · 읽기만) vision_core 를 임시 폴더에 복사 → `color.py` 덮어쓰기 + `ingest_<env>.patch` 적용 (= 통합 담당이 할 일)
    env = dev (SIM · drone-dev) · real (REAL · drone-real)
  ① make_colorer 고르기: 꾸러미 있음 → siglip2-zs · VISION_COLOR=rule → rule-v10 · 꾸러미 없음 → rule-v10 · 해시 틀림 → auto 는 규칙 · siglip 은 오류
  ② profile 반환 형식이 규칙판과 같고 ColorAccumulator · match 그대로 · 아주 작은 박스 = 판정 불가
  ③ 판정이 평가판 (color_lab F1) 과 같음 — 디관 영상 확인 셋 45명 1순위 일치 · 한 명 ms
  ④ 배포 경로 끝까지 — `integration_v1_2/tests_v1_2.py` (UE S1_a16 · REAL · SIM · 백엔드 계약) 를 패치된 배포 판으로 · IngestWorker.stats color_method
실행: /home/se/miniconda3/envs/drone/bin/python tests_color_siglip.py [--skip-e2e]   (transformers 없는 환경 = 배포 이미지와 같은 조건)
꾸러미: drone_yolo/runs_person/_pack_color_siglip2 (= _pack_v12_sim · _pack_v12_real 안 color_siglip2)
"""
import json, os, pickle, shutil, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PACK = REPO / "drone_yolo" / "runs_person" / "_pack_color_siglip2"
SANDBOX = {"dev": Path("/home/se/sandboxes/drone-dev"), "real": Path("/home/se/sandboxes/drone-real")}


def overlay(env):
    """배포 판 vision_core 복사 + color.py + 패치 → 임시 폴더 (app/vision_core/…)"""
    tmp = Path(tempfile.mkdtemp(prefix=f"color13_{env}_"))
    shutil.copytree(SANDBOX[env] / "app" / "vision_core", tmp / "app" / "vision_core", ignore=shutil.ignore_patterns("*.bak-*", "__pycache__"))
    shutil.copy2(HERE / "app" / "vision_core" / "color.py", tmp / "app" / "vision_core" / "color.py")
    r = subprocess.run(["patch", "-p1", "-d", str(tmp), "-i", str(HERE / f"ingest_{env}.patch")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return tmp


def unit(env):
    tmp = overlay(env)
    stage = Path(tempfile.mkdtemp(prefix="stage_"))
    shutil.copytree(SANDBOX[env] / "app", stage / "app", ignore=shutil.ignore_patterns("*.bak-*", "__pycache__"))
    for f in (tmp / "app").rglob("*.py"):
        shutil.copy2(f, stage / "app" / f.relative_to(tmp / "app"))
    sys.path.insert(0, str(stage))
    import numpy as np
    from app.vision_core.color import COLORS, ColorAccumulator, make_colorer, match
    import app.vision_core.ingest as ing
    assert "make_colorer" in ing.__dict__
    res = {"env": env}
    os.environ.pop("VISION_COLOR", None)
    c = make_colorer(str(PACK)); res["auto+꾸러미"] = c.name; assert c.name == "siglip2-zs"
    os.environ["VISION_COLOR"] = "rule"; res["rule"] = make_colorer(str(PACK)).name; assert res["rule"] == "rule-v10"
    os.environ.pop("VISION_COLOR")
    res["꾸러미 없음"] = make_colorer("/nonexistent").name; assert res["꾸러미 없음"] == "rule-v10"
    bad = Path(tempfile.mkdtemp()) / "pack"; shutil.copytree(PACK, bad)
    m = json.load(open(bad / "color_model.json")); m["sha256"]["text.npy"] = "0" * 64; (bad / "color_model.json").write_text(json.dumps(m))
    logs = []; res["해시 틀림 auto"] = make_colorer(str(bad), log=logs.append).name; assert res["해시 틀림 auto"] == "rule-v10" and logs
    os.environ["VISION_COLOR"] = "siglip"
    try:
        make_colorer(str(bad)); raise AssertionError("해시 틀림인데 siglip 이 오류를 안 냄")
    except RuntimeError:
        res["해시 틀림 siglip"] = "RuntimeError ✅"
    os.environ.pop("VISION_COLOR")
    V = pickle.load(open(REPO / "data" / "raw" / "color" / "_lab" / "V.pkl", "rb"))
    p = c.profile(V[0]["frame"], V[0]["box"], stats=None)
    assert set(p) >= {"status", "probs", "top", "n_pixels", "reason"} and len(p["probs"]) == 12 and abs(sum(p["probs"].values()) - 1) < 1e-3
    A = ColorAccumulator()
    for _ in range(3):
        A.add(p)
    assert match(A.result(), A.result()["top"][0][0])[0] == "일치"
    assert c.profile(V[0]["frame"], (0, 0, 3, 5))["status"] == "undetermined"
    ref = pickle.load(open(REPO / "data" / "raw" / "color" / "_lab" / "preds" / "F1_B1+남색문턱_V.pkl", "rb"))[0]
    t0 = time.time(); got = [c.profile(x["frame"], x["box"]) for x in V]; ms = (time.time() - t0) * 1000 / len(V)
    agree = float(np.mean([max(g["probs"], key=g["probs"].get) == COLORS[int(np.argmax(rr))] for g, rr in zip(got, ref)]))
    res["평가판 1순위 일치"] = round(agree, 4); res["한 명 ms"] = round(ms, 1)
    assert agree == 1.0
    print(json.dumps(res, ensure_ascii=False))


def e2e(env):
    """tests_v1_2 (UE S1_a16 · REAL · SIM · 백엔드 계약) 를 패치된 배포 판 vision_core 로 — 계약 모듈은 drone-real"""
    tmp = overlay(env)
    sys.path.insert(0, str(REPO / "drone_yolo" / "deploy" / "vision_worker" / "integration_v1_2"))
    import tests_v1_2 as T
    T.HERE = tmp                                                     # 이 폴더의 app/ 을 배포 판 위에 덮음
    mpath = REPO / "drone_yolo" / "metrics" / "integration_v1_2_S1_a16.json"
    keep = mpath.read_bytes() if mpath.exists() else None
    sys.argv = ["tests_v1_2.py", "--engine", "--backend", "/home/se/sandboxes/drone-real"]
    os.environ["VISION_COLOR"] = "siglip"; os.environ["VISION_COLOR_PACK"] = str(PACK)
    try:
        T.main()
        out = json.load(open(mpath))
    finally:
        if keep is not None:
            mpath.write_bytes(keep)                                  # 커밋된 1.2 기록은 그대로 둔다
    (REPO / "drone_yolo" / "metrics" / f"integration_v1_3_{env}_S1_a16.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    assert all(v["frames_validated"] > 0 for v in out.values())
    print(env, "e2e", {k: {kk: v[kk] for kk in ("frames_validated", "cand_per_actor", "query_green_match", "query_other_match")} for k, v in out.items()})


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] in ("--unit", "--e2e"):          # 하위 프로세스 (env 마다 모듈을 새로 읽기 위해)
        (unit if sys.argv[1] == "--unit" else e2e)(sys.argv[2])
    else:
        for env in ("dev", "real"):
            for step in ("--unit",) + (() if "--skip-e2e" in sys.argv else ("--e2e",)):
                r = subprocess.run([sys.executable, __file__, step, env], capture_output=True, text=True)
                lines = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith(("{", "dev", "real", "Traceback", "AssertionError")) or "Error" in l]
                print("\n".join(lines[-6:]))
                assert r.returncode == 0, f"{env} {step} 실패"
        print("PASS")
