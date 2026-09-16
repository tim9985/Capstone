"""
fov_status.py — 화각 · 고도 비교 실험(train_fov.sh 대기열) 진행 상황 한눈에

실행: python fov_status.py              # 한 번
      python fov_status.py --watch 60   # 60초마다 갱신 (Ctrl+C 로 종료)

보여주는 것
  1) 단계표: 학습 데이터 → 시험셋 → yolo11s 학습 → 평가(yolo11s · M1) → yolo11m 학습 → 평가 → 완료
  2) 지금 단계 상세: 학습이면 epoch 진행률 · 최근 epoch 지표 · best / 평가면 칸 진행률
  3) 남은 일정: 지금 단계 끝 · 첫 결과표(yolo11s) · 최종 결과표(yolo11m) 예상 시각(KST)
  4) 안전장치: GPU · 체크포인트 백업(fov_guard) · autoheal job.conf · 재부팅 · 사고 기록
  5) 결과가 나온 모델은 화각 × 고도 재현율 표 (metrics/eval_fov_<실행>.csv)
학습 진행 막대 · GPU · 프로세스 조회는 train_status.py 함수를 그대로 쓴다.
"""
import argparse
import csv
import json
import re
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path

from train_status import BASE_DIR, KST, fitness, gpu_line, python_procs, read_progress, sh

LOG = BASE_DIR / "logs" / "train_fov.log"
RUNS = BASE_DIR / "runs_person"
TEST = BASE_DIR / "data" / "det_fov_test"
RUN_11S, RUN_11M, RUN_M1 = "fov_11s_1280_all", "fov_11m_1280_all", "m1_11m_1280"
EPOCHS_11S = 30
EVAL_MIN = {RUN_11S: 25, RUN_M1: 25, RUN_11M: 25}   # 평가 1개 예상 시간(분) — 첫 평가가 끝나면 로그 시각으로 보정


def now_kst():
    return datetime.now(KST)


def fmt(dt):
    return dt.strftime("%m-%d %H:%M KST")


def log_stamps():
    """train_fov.log 의 '=== 단계 [시각 UTC] ===' 줄 → [(단계, datetime)]"""
    if not LOG.exists():
        return []
    out = []
    for m in re.finditer(r"^=== (.+?) \[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) UTC\] ===$",
                         LOG.read_text(errors="ignore").replace("\r", "\n"), re.M):
        out.append((m.group(1), datetime.strptime(m.group(2), "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST.__class__(timedelta(0))).astimezone(KST)))
    return out


def results(run):
    p = RUNS / run / "results.csv"
    if not p.exists():
        return []
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    return [{k.strip(): v for k, v in r.items()} for r in rows]


def epoch_minutes(rows):
    t = [float(r["time"]) for r in rows]
    d = [b - a for a, b in zip(t, t[1:]) if b > a]
    if d:
        return statistics.median(d[-5:]) / 60
    return t[0] / 60 if t else None


def run_args(run):
    p = RUNS / run / "args.yaml"
    if not p.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def eval_csv(run):
    return BASE_DIR / "metrics" / f"eval_fov_{run}.csv"


def stages():
    """(이름, 끝났나) — train_fov.sh 가 건너뛰는 기준과 같은 흔적 파일로 판정"""
    man = json.loads((TEST / "manifest.json").read_text()) if (TEST / "manifest.json").exists() else {}
    return [
        ("학습 데이터 (data/det_fov)", (BASE_DIR / "data" / "det_fov" / ".done").exists()),
        (f"시험셋 v{man.get('version', '?')} ({len({t['dir'] for t in man.get('table', [])})}칸)", man.get("version") == 4),
        ("yolo11s 학습 (30에폭)", (BASE_DIR / "weights" / f"{RUN_11S}.pt").exists()),
        ("평가 yolo11s", eval_csv(RUN_11S).exists()),
        ("평가 M1 (기준)", eval_csv(RUN_M1).exists()),
        ("yolo11m 학습 (시간 제한)", (BASE_DIR / "weights" / f"{RUN_11M}.pt").exists()),
        ("평가 yolo11m", eval_csv(RUN_11M).exists()),
        ("대기열 완료", (BASE_DIR / "logs" / "train_fov.done").exists()),
    ]


def training_detail(run, total_hint):
    rows = results(run)
    prog = read_progress(str(LOG))
    per = epoch_minutes(rows)
    total = total_hint
    lines = []
    remain_min = None
    if prog:
        _, phase, p = prog
        total = p["total"]           # 시간 제한 학습은 첫 epoch 뒤 다시 계산된 값
        if per is None:
            per = p["iters"] / p["its"] / 60 * 1.15
        if phase == "학습":
            frac = min(1.0, (p["it"] / p["its"] / 60) / per)
            where = f"epoch {p['epoch']}/{total} 학습 {p['it']}/{p['iters']} ({p['it']/p['iters']:.0%}) · {p['its']:.1f} it/s"
        else:
            frac = min(1.0, (p["iters"] / p["its"] / 60 + p["vit"] / p["vits"] / 60) / per)
            where = f"epoch {p['epoch']}/{total} 검증 {p['vit']}/{p['viters']} ({p['vit']/p['viters']:.0%})"
        remain_min = max(0.0, (total - len(rows) - frac) * per)
        lines.append(f"  진행   : {where}")
        if p["age"] > 300:
            lines.append(f"  ⚠ 로그가 {p['age']/60:.0f}분째 갱신되지 않았다 — 멈췄는지 확인")
    lines.append(f"  완료   : {len(rows)}/{total} epoch" + (f" · epoch당 {per:.1f}분 (검증 포함)" if per else ""))
    if rows:
        best = max(rows, key=fitness)
        lines.append("  최근   : " + " · ".join(
            f"{int(float(r['epoch']))}ep mAP50 {float(r['metrics/mAP50(B)']):.3f} R {float(r['metrics/recall(B)']):.3f}"
            for r in rows[-3:]))
        lines.append(f"  best   : {int(float(best['epoch']))}ep (fitness) mAP50 {float(best['metrics/mAP50(B)']):.3f} · "
                     f"mAP50-95 {float(best['metrics/mAP50-95(B)']):.3f}  ※ 검증셋이 16~160 px 라 M1(0.65)과 절대값 비교 불가")
    return lines, remain_min, per


def eval_detail(run):
    p = BASE_DIR / "logs" / f"eval_fov_{run}.log"
    total = len({t["dir"] for t in json.loads((TEST / "manifest.json").read_text())["table"]}) if (TEST / "manifest.json").exists() else 0
    done = len(re.findall(rf"^\s+{re.escape(run)} \S+: 채점 원본", p.read_text(errors="ignore"), re.M)) if p.exists() else 0
    return f"  진행   : 칸 {done}/{total} ({done/total:.0%})" if total else "  진행   : 시험셋 manifest 없음", done, total


def result_table(run):
    rows = list(csv.DictReader(open(eval_csv(run), encoding="utf-8")))
    alts = sorted({int(r["altitude_m"]) for r in rows})
    fovs = sorted({int(r["fov_deg"]) for r in rows})
    rk = next(k for k in rows[0] if k.startswith("recall@"))
    out = [f"\n[{run}] 재현율@0.15 (세트 공통 원본 · * 학습 크기 밖 외삽 · 10 m 열은 원본 세트가 달라 직접 비교 주의)"]
    for pose, label in (("standing", "서 있는 크기 0.5m"), ("lying", "누운 크기 1.7m")):
        out.append(f"  {label:<8}" + "".join(f"{a:>5} m     " for a in alts))
        for f in fovs:
            cells = []
            for a in alts:
                r = next(x for x in rows if int(x["fov_deg"]) == f and int(x["altitude_m"]) == a and x["pose"] == pose)
                star = "*" if r.get("extrapolated") == "1" else " "
                cells.append(f"{int(r['person_px']):>3}px{star}{float(r[rk]):.2f} ")
            out.append(f"  {f:>3}°     " + "".join(cells))
    return out


def report():
    now = now_kst()
    st = stages()
    stamps = log_stamps()
    first = next((s for s in st if not s[1]), None)
    train = python_procs("train_person.py")
    evals = python_procs("eval_fov.py")
    queue = sh(["pgrep", "-f", r"^/bin/bash \./train_fov\.sh"])
    guard = sh(["pgrep", "-f", r"fov_guard\.sh$"])
    print(f"=== 화각 · 고도 비교 실험 — {fmt(now)} ===")
    started = next((t for n, t in stamps if n == "시작"), None)
    if started:
        print(f"대기열 시작 {fmt(started)} · 경과 {(now - started).total_seconds()/3600:.1f}시간 (20시간 목표 → {fmt(started + timedelta(hours=20))})")
    print(f"프로세스: 대기열 {'실행 중' if queue else '없음'} · 학습 {'실행 중' if train else '-'} · 평가 {'실행 중' if evals else '-'}")

    print("\n[단계]")
    for name, done in st:
        mark = "✅" if done else ("🔄" if (first and name == first[0] and (queue or train or evals)) else "⏳")
        print(f"  {mark} {name}")

    print("\n[지금 단계]")
    remain_now = None
    cur = first[0] if first else None
    if cur is None:
        print("  모두 끝남")
    elif cur.startswith("yolo11s 학습") or cur.startswith("yolo11m 학습"):
        run = RUN_11S if cur.startswith("yolo11s") else RUN_11M
        lines, remain_now, _ = training_detail(run, EPOCHS_11S if run == RUN_11S else "?")
        print(f"  {run}")
        print("\n".join(lines))
        if run == RUN_11M:
            a = run_args(run)
            if a.get("time"):
                print(f"  시간 제한: {a['time']}시간")
    elif cur.startswith("평가"):
        run = {"평가 yolo11s": RUN_11S, "평가 M1 (기준)": RUN_M1, "평가 yolo11m": RUN_11M}[cur]
        line, done, total = eval_detail(run)
        print(f"  {run}\n{line}")
        began = next((t for n, t in reversed(stamps) if n == f"화각 평가 {run}"), None)
        if began and done:
            remain_now = (now - began).total_seconds() / 60 / done * (total - done)
        else:
            remain_now = EVAL_MIN[run]
    else:
        print(f"  {cur}")

    # 남은 일정 — 지금 단계 끝부터 뒤 단계를 차례로 더한다
    print("\n[남은 일정 (예상)]")
    t = now + timedelta(minutes=remain_now or 0)
    todo = [n for n, d in st if not d]
    time_11m = float(run_args(RUN_11M).get("time") or 8.0)
    plan = {"yolo11s 학습 (30에폭)": None, "평가 yolo11s": EVAL_MIN[RUN_11S], "평가 M1 (기준)": EVAL_MIN[RUN_M1],
            "yolo11m 학습 (시간 제한)": time_11m * 60 + 10, "평가 yolo11m": EVAL_MIN[RUN_11M]}
    if todo and todo[0] in plan:
        print(f"  {todo[0]} 끝: {fmt(t)}")
        for n in todo[1:]:
            if n not in plan:
                continue
            t += timedelta(minutes=plan[n])
            label = {"평가 M1 (기준)": "→ 첫 결과표 (yolo11s · M1)", "평가 yolo11m": "→ 최종 결과표 (yolo11m)"}.get(n, "")
            print(f"  {n} 끝: {fmt(t)} {label}")
    elif not todo:
        print("  없음")

    print("\n[안전장치]")
    print(f"  GPU      : {gpu_line()}")
    boot = sh(["uptime", "-s"])
    if boot and started:
        bdt = datetime.strptime(boot, "%Y-%m-%d %H:%M:%S").astimezone(KST)
        print(f"  부팅     : {fmt(bdt)}" + (" ⚠ 대기열 시작 뒤 재부팅됨 (GPU 장애 복구 가능성)" if bdt > started else " (대기열 시작 뒤 재부팅 없음)"))
    gl = (BASE_DIR / "logs" / "fov_guard.log")
    last_bk = re.findall(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) UTC\] (\S+) last\.pt → last_backup\.pt",
                         gl.read_text(errors="ignore"), re.M) if gl.exists() else []
    if last_bk:
        ts, bk_run = last_bk[-1]
        bdt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST.__class__(timedelta(0))).astimezone(KST)
        bk = f"{fmt(bdt)} ({bk_run} · {(now - bdt).total_seconds()/60:.0f}분 전)"
    else:
        bk = "없음"
    job = re.search(r"^JOB_NAME=(.*)$", (BASE_DIR / "autoheal" / "job.conf").read_text(), re.M)
    print(f"  백업     : fov_guard {'실행 중' if guard else '⚠ 꺼짐'} · 마지막 {bk} · job.conf JOB_NAME={job.group(1) if job else '?'}")
    inc = BASE_DIR / "overnight_incident.log"
    fov_inc = [l for l in inc.read_text(errors="ignore").splitlines() if "train_fov" in l] if inc.exists() else []
    print(f"  사고 기록: {fov_inc[-1] if fov_inc else '없음'}")

    for run in (RUN_11S, RUN_M1, RUN_11M):
        if eval_csv(run).exists():
            print("\n".join(result_table(run)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, nargs="?", const=60, default=0, metavar="SEC")
    args = ap.parse_args()
    if not args.watch:
        report()
        return
    try:
        while True:
            print("\033[2J\033[H", end="")
            try:
                report()
            except Exception as e:      # 결과 파일을 쓰는 순간과 겹치면 읽기가 실패할 수 있다 → 다음 주기에
                print(f"조회 실패 (다음 주기에 다시): {e}")
            print(f"\n{args.watch}초마다 갱신 · Ctrl+C 로 종료")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
