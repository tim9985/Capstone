"""
train_status.py — 학습 진행 상황 한눈에 확인

실행: python train_status.py                    # 가장 최근 실행 자동 선택
      python train_status.py --name m1_11m_1280
      python train_status.py --watch 60         # 60초마다 갱신 (Ctrl+C 로 종료)
      python train_status.py --rows 15          # 최근 15 epoch 표시
      python train_status.py --log resume_m1.log  # 진행 막대를 읽을 로그 직접 지정

보여주는 것
  · 학습 / 평가 / 파이프라인 프로세스
  · 지금 epoch 안 진행률(로그 진행 막대) · 남은 시간 · 예상 종료 시각(KST)
  · 최근 epoch 지표 · best (fitness 기준 — best.pt 가 실제로 고르는 기준)
  · GPU 온도 · 전력 · 스로틀 원인 · 감시 스크립트 · 사고 기록

2026-09-14 서버(리눅스)용으로 고침
  · 프로세스 확인이 powershell 전용이라 리눅스에서 늘 "실행 안 함" → pgrep 사용
  · best 를 mAP50 으로 골랐으나 ultralytics best.pt 는 fitness(0.1·mAP50 + 0.9·mAP50-95) 기준
  · 이어 학습(--resume)하면 results.csv 의 time 이 0 부터 다시 시작해 epoch 시간이 음수로 계산됐다
  · 가중치 복사명 규칙 변경 반영 (yolov8s_{name}.pt → {name}.pt)
"""
import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
KST = timezone(timedelta(hours=9))
PIPELINES = "overnight_m1280.sh|resume_m1.sh|finish_m1.sh|run_remaining_experiments.sh|gpu_stability_test.sh"
# nvidia-smi clocks_event_reasons 비트
THROTTLE = {0x1: "유휴", 0x4: "전력한도(SW)", 0x8: "HW 감속", 0x20: "과열(SW)",
            0x40: "과열(HW)", 0x80: "전원 브레이크"}
# ultralytics 진행 막대 — 학습: "  15/34  6.62G ... 1280: 84% ━━ 5300/6307 7.5it/s 11:45<2:14"
PROG_TRAIN = re.compile(r"^\s*(\d+)/(\d+)\s+[\d.]+G\s.*?(\d+)/(\d+)\s+([\d.]+)(it/s|s/it)")
PROG_VAL = re.compile(r"Class\s+Images.*?(\d+)/(\d+)\s+([\d.]+)(it/s|s/it)")
# 진행 막대 줄 앞에 터미널 제어 코드(ESC[K)가 붙어 ^ 매칭이 실패했다 → 먼저 걷어낸다
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def sh(cmd, timeout=15):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def python_procs(pattern):
    """명령줄에 pattern 이 있는 python 프로세스 [(pid, 명령줄)]. 셸 명령줄 자기 매칭은 뺀다."""
    if os.name == "nt":
        out = sh(["powershell", "-NoProfile", "-Command",
                  "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                  f"Where-Object {{ $_.CommandLine -like '*{pattern}*' }} | "
                  "ForEach-Object { \"$($_.ProcessId) $($_.CommandLine)\" }"], 30)
    else:
        out = sh(["pgrep", "-af", pattern])
    res = []
    for ln in out.splitlines():
        parts = ln.split(maxsplit=1)
        if len(parts) == 2 and "python" in parts[1].split()[0].lower():
            res.append((int(parts[0]), parts[1]))
    return sorted(res)  # 데이터로더 워커도 같은 명령줄 → PID 가 가장 작은 것이 본체


def gpu_line():
    q = "temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.total,clocks_event_reasons.active"
    out = sh(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    if not out or "," not in out:
        return "조회 불가 — GPU 장애 가능성 (nvidia-smi 무응답)"
    try:
        t, pd, pl, u, mu, mt, thr = [x.strip() for x in out.splitlines()[0].split(",")]
        v = int(thr, 16) if thr.startswith("0x") else 0
        why = ",".join(n for b, n in THROTTLE.items() if v & b) or "없음"
        return (f"{t}°C · {float(pd):.0f}/{float(pl):.0f} W · 사용률 {u}% · "
                f"메모리 {int(mu)/1024:.1f}/{int(mt)/1024:.0f} GB · 스로틀 {why}")
    except ValueError:
        return out.splitlines()[0]


def tail_text(path, nbytes=400_000):
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - nbytes))
        return f.read().decode("utf-8", "ignore")


def read_progress(log_arg):
    """가장 최근 로그에서 마지막 진행 막대를 읽는다 → (로그, phase, 정보) 또는 None"""
    # 로그는 최상위 · logs/ · autoheal/logs/ (자동 재개 학습) 에 흩어져 있다
    cands = [Path(log_arg)] if log_arg else sorted(
        [*BASE_DIR.glob("*.log"), *BASE_DIR.glob("logs/*.log"), *BASE_DIR.glob("autoheal/logs/*.log")],
        key=lambda p: p.stat().st_mtime, reverse=True)[:6]
    for log in cands:
        if not log.exists():
            continue
        lines = re.split(r"[\r\n]", ANSI.sub("", tail_text(log)))
        last_train = last_val = None
        for i, ln in enumerate(lines):
            if (m := PROG_TRAIN.search(ln)):
                last_train = (i, m)
            elif (m := PROG_VAL.search(ln)):
                last_val = (i, m)
        if last_train is None:
            continue
        rate = lambda v, unit: float(v) if unit == "it/s" else 1 / max(float(v), 1e-9)
        _, t = last_train
        info = {"epoch": int(t.group(1)), "total": int(t.group(2)), "it": int(t.group(3)),
                "iters": int(t.group(4)), "its": rate(t.group(5), t.group(6)),
                "age": time.time() - log.stat().st_mtime}
        phase = "학습"
        if last_val and last_val[0] > last_train[0]:
            _, v = last_val
            phase = "검증"
            info.update(vit=int(v.group(1)), viters=int(v.group(2)), vits=rate(v.group(3), v.group(4)))
        return log, phase, info
    return None


def fitness(r):
    return 0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"])


def last_incident():
    found = []
    for name in ("gpu_watchdog_incident.log", "overnight_incident.log"):
        p = BASE_DIR / name
        if p.exists():
            found.append((p.stat().st_mtime, name))
    if not found:
        return "없음"
    ts, name = max(found)
    return f"마지막 기록 {datetime.fromtimestamp(ts, KST):%m-%d %H:%M} KST ({name})"


def _report(args):
    runs = BASE_DIR / "runs_person"
    if args.name:
        name = args.name
    else:
        # 첫 epoch 중인 실행은 results.csv 가 아직 없다 → args.yaml(학습 시작 때 생성)도 후보로 본다.
        # results.csv 만 보면 방금 시작한 M2 대신 끝난 M1 이 골라졌다 (09-14)
        stamp = lambda d: max((d / f).stat().st_mtime for f in ("args.yaml", "results.csv") if (d / f).exists())
        cands = [d for d in runs.glob("*") if (d / "args.yaml").exists() or (d / "results.csv").exists()] \
            if runs.exists() else []
        if not cands:
            print("runs_person 에 결과가 없습니다.")
            return
        name = max(cands, key=stamp).name
    run_dir = runs / name
    csv_path = run_dir / "results.csv"

    train = python_procs("train_person.py")
    evals = python_procs("eval_domain.py")
    pipes = [ln.split(maxsplit=1)[1] for ln in sh(["pgrep", "-af", PIPELINES]).splitlines()
             if ln.split(maxsplit=1)[1].startswith(("bash", "/bin/bash"))] if os.name != "nt" else []
    # ^ 고정: 명령줄에 이 문자열을 담은 다른 셸까지 잡는 자기 매칭 방지
    watchdog = bool(sh(["pgrep", "-f", r"^bash gpu_watchdog\.sh"])) if os.name != "nt" else None

    print(f"실행        : {name}")
    print(f"학습 프로세스: {'실행 중 (PID ' + str(train[0][0]) + ')' if train else '없음'}"
          f"{'  · 평가 실행 중 (eval_domain.py)' if evals else ''}")
    if pipes:
        print(f"파이프라인  : {', '.join(p.split('/')[-1] for p in pipes)}")
    print(f"GPU         : {gpu_line()}")
    if watchdog is not None:
        print(f"감시 스크립트: {'켜짐' if watchdog else '꺼짐'} · 사고 {last_incident()}")

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8"))) if csv_path.exists() else []
    total, run_args = None, {}
    try:
        import yaml
        run_args = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8")) or {}
        total = int(run_args["epochs"])
    except Exception:
        pass

    # ── 진행 막대 (학습 중일 때만 의미 있음) ──
    prog = read_progress(args.log) if train else None
    per = None
    pos_diffs = [b - a for a, b in zip([float(r["time"]) for r in rows], [float(r["time"]) for r in rows][1:])
                 if b - a > 0]  # resume 경계(time 리셋)의 음수 차이는 버린다
    if pos_diffs:
        per = statistics.median(pos_diffs[-5:])
    if prog:
        log, phase, p = prog
        total = p["total"]  # time 제한 학습은 args.yaml 의 epochs 가 나중에 바뀌므로 로그 값을 믿는다
        train_sec = p["iters"] / p["its"]
        if per is None:
            per = train_sec * 1.15  # 첫 epoch: 검증 몫을 대략 더한다
        if phase == "학습":
            done_frac = min(1.0, (p["it"] / p["its"]) / per)
            where = f"epoch {p['epoch']}/{total} 학습 {p['it']}/{p['iters']} ({p['it']/p['iters']:.0%}) · {p['its']:.1f} it/s"
        else:
            done_frac = min(1.0, (train_sec + p["vit"] / p["vits"]) / per)
            where = f"epoch {p['epoch']}/{total} 검증 {p['vit']}/{p['viters']} ({p['vit']/p['viters']:.0%})"
        remain = max(0.0, (total - len(rows) - done_frac) * per)
        eta = datetime.now(KST) + timedelta(seconds=remain)
        print(f"\n진행        : {where}")
        print(f"전체        : {len(rows)}/{total} epoch 완료 · epoch당 {per/60:.1f}분 · "
              f"남은 시간 {remain/3600:.1f}시간 → 학습 종료 예상 {eta:%m-%d %H:%M} KST")
        if p["age"] > 300:
            print(f"※ 로그({log.name})가 {p['age']/60:.0f}분째 갱신되지 않았다 — 멈췄는지 확인할 것")

    if not rows:
        print("\n첫 epoch 진행 중 (results.csv 기록 없음)")
        return

    if "metrics/accuracy_top1" in rows[0]:  # 분류 학습
        print(f"\n=== {name}  {len(rows)}/{total or '?'} epoch  (분류) ===")
        print(f"{'ep':>3} {'top1':>8} {'top5':>8} {'train_loss':>11} {'val_loss':>10}")
        for x in rows[-args.rows:]:
            print(f"{int(float(x['epoch'])):>3} {float(x['metrics/accuracy_top1']):>8.4f} "
                  f"{float(x.get('metrics/accuracy_top5', 0)):>8.4f} "
                  f"{float(x.get('train/loss', 0)):>11.4f} {float(x.get('val/loss', 0)):>10.4f}")
        return

    best = max(rows, key=fitness)
    print(f"\n=== {name}  최근 {min(args.rows, len(rows))} epoch ===")
    print(f"{'ep':>3} {'mAP50':>7} {'mAP50-95':>9} {'fitness':>8} {'P':>6} {'R':>6} {'box_loss':>9}")
    for r in rows[-args.rows:]:
        mark = " ★best" if r is best else ""
        print(f"{int(float(r['epoch'])):>3} {float(r['metrics/mAP50(B)']):>7.3f} "
              f"{float(r['metrics/mAP50-95(B)']):>9.3f} {fitness(r):>8.4f} "
              f"{float(r['metrics/precision(B)']):>6.3f} {float(r['metrics/recall(B)']):>6.3f} "
              f"{float(r['train/box_loss']):>9.3f}{mark}")
    stale = len(rows) - int(float(best["epoch"]))
    print(f"\nbest (fitness) epoch {int(float(best['epoch']))}: mAP50 {float(best['metrics/mAP50(B)']):.3f} · "
          f"mAP50-95 {float(best['metrics/mAP50-95(B)']):.3f} → best.pt · {stale} epoch 째 미갱신")

    if not train:
        finals = [BASE_DIR / "weights" / f"{name}.pt", BASE_DIR / "weights" / f"yolov8s_{name}.pt"]
        done = [w for w in finals if w.exists() and w.stat().st_mtime >= csv_path.stat().st_mtime - 300]
        if done:
            # time 제한 학습은 args.yaml 의 epochs(예: 60)가 실제 에폭 수(예: 27)와 달라
            # "epochs 미달 = 조기 종료" 로 판단하면 틀린다 (M2 가 그렇게 표시됐다, 09-15)
            patience, limit = int(run_args.get("patience") or 0), float(run_args.get("time") or 0)
            if patience and stale >= patience:
                reason = f"조기 종료(patience {patience})"
            elif limit and total and len(rows) < total:
                reason = f"시간 제한 {limit:g}시간 도달 ({len(rows)} epoch)"
            elif total and len(rows) < total:
                reason = f"epochs {total} 미달로 끝남 ({len(rows)} epoch) — 로그 확인"
            else:
                reason = "전체 epoch 소화"
            print(f"\n✔ 학습 완료 — {reason}. 최종 가중치: weights/{done[0].name}")
        elif evals:
            print("\n학습은 끝났고 평가가 진행 중이다.")
        else:
            print("\n※ 학습 프로세스가 없는데 완료 흔적(weights/ 복사)이 없다. 중단됐다면 이어서:")
            print(f"   python train_person.py --stage 1 --resume --name {name}")


def main():
    ap = argparse.ArgumentParser(description="학습 진행 상황 확인")
    ap.add_argument("--name", default=None, help="결과 폴더명. 생략하면 runs_person 에서 가장 최근 것")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--log", default=None, help="진행 막대를 읽을 로그 (생략하면 최근 *.log 자동)")
    ap.add_argument("--watch", type=int, nargs="?", const=30, default=0, metavar="SEC",
                    help="지정 초마다 화면을 새로 그린다 (기본 30초). Ctrl+C 로 종료")
    args = ap.parse_args()

    if not args.watch:
        _report(args)
        return
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print(f"[{datetime.now(KST):%H:%M:%S} KST] {args.watch}초마다 갱신 · Ctrl+C 로 종료\n")
            try:
                _report(args)
            except Exception as e:
                # 학습이 results.csv 를 쓰는 순간과 겹치면 읽기가 실패할 수 있다 — 다음 주기에 다시 읽는다
                print(f"조회 실패(무시): {e}")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\n감시 종료")


if __name__ == "__main__":
    main()
