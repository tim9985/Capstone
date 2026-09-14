#!/usr/bin/env python3
"""download_status.py — NOMAD 50 · 70 m 다운로드(download_nomad_far.sh) 진행 확인

  python3 download_status.py              # 한 번
  python3 download_status.py --watch 60   # 60초마다 갱신 (Ctrl+C 로 끝)

rclone 로그의 순간 속도 · ETA 는 크게 튄다 → 최근 10분 평균으로 속도와 남은 시간을 다시 계산한다.
"""
import argparse
import os
import re
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
LOG = BASE / "logs" / "nomad_far_dl.log"
RAW = BASE.parent / "data" / "raw"
NOMAD = RAW / "NOMAD"
DONE = RAW / "nomad_far_dl.done"
ALTS = ("a50", "a70")
RESTART = ("cd ~/JupyterLAB/Capstone/drone_yolo && setsid nohup ./download_nomad_far.sh "
           "> logs/nomad_far_dl.log 2>&1 < /dev/null &")

UNITS = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40}
NOTICE = re.compile(
    r"(\d{4}/\d\d/\d\d \d\d:\d\d:\d\d) NOTICE:\s+([\d.]+) (\w+) / ([\d.]+) (\w+), (\d+)%, "
    r".*?\(xfr#(\d+)/(\d+)\)")
PHASE = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) UTC\] (labels|images) 받기 \(시도 (\d+)\)")
START = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) UTC\] 시작")


def human(n):
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or u == "TiB":
            return f"{n:.1f} {u}" if u != "B" else f"{n:.0f} B"
        n /= 1024


def dur(sec):
    sec = int(sec)
    h, m = divmod(sec // 60, 60)
    return f"{h}시간 {m}분" if h else f"{m}분"


def processes():
    """(스크립트 PID, rclone 명령) — /proc 를 직접 읽어 자기 자신 · 셸 명령줄 오탐을 피한다."""
    script = rclone = None
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            argv = (Path("/proc") / d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        argv = [a.decode(errors="replace") for a in argv if a]
        if len(argv) >= 2 and argv[0].endswith("bash") and argv[1].endswith("download_nomad_far.sh"):
            script = int(d)
        elif len(argv) >= 3 and argv[0].endswith("rclone") and argv[1] == "copy" and argv[2].startswith("gdrive:"):
            rclone = argv[2]
    return script, rclone


def count_disk():
    out = {}
    for alt in ALTS:
        imgs = labels = size = 0
        for kind in ("images", "labels"):
            root = NOMAD / kind
            if not root.is_dir():
                continue
            for actor in os.scandir(root):
                d = Path(actor.path) / f"{actor.name}_{alt}"
                if not d.is_dir():
                    continue
                for f in os.scandir(d):
                    if not f.is_file():
                        continue
                    if kind == "images":
                        if f.name.lower().endswith((".jpg", ".jpeg", ".png")):
                            imgs += 1
                            size += f.stat().st_size
                    else:
                        labels += 1
        out[alt] = (imgs, labels, size)
    return out


def report():
    now = datetime.now(timezone.utc).replace(tzinfo=None)  # 서버 시간대 = UTC (rclone 로그 시각도 UTC)
    print(f"== NOMAD 50·70 m 다운로드 — {(now + timedelta(hours=9)):%m-%d %H:%M} KST")
    if not LOG.exists():
        print(f"로그 없음: {LOG}\n시작: {RESTART}")
        return
    text = LOG.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    lines = text.splitlines()

    script, rclone = processes()
    started = START.findall(text)
    t0 = datetime.strptime(started[-1], "%Y-%m-%d %H:%M:%S") if started else None
    if DONE.exists():
        state = "✅ 완료"
    elif script:
        state = f"▶ 진행 중 (PID {script}" + (f" · 경과 {dur((now - t0).total_seconds())})" if t0 else ")")
    else:
        state = "⏹ 멈춤 — 완료 표시가 없다"
    print(f"상태  {state}")

    phases = list(PHASE.finditer(text))
    fails = len(re.findall(r"실패", text))
    errors = sum(1 for l in lines if "ERROR" in l)
    if phases:
        p = phases[-1]
        step = "1/2 라벨" if p.group(2) == "labels" else "2/2 이미지"
        print(f"단계  {step} (시도 {p.group(3)})" + (f" · rclone {rclone}" if rclone else ""))
        seg = text[p.end():]
    else:
        seg = ""
    print(f"오류  rclone ERROR {errors}줄 · 재시도 {fails}회")

    pts = []
    for m in NOTICE.finditer(seg):
        t = datetime.strptime(m.group(1), "%Y/%m/%d %H:%M:%S")
        pts.append((t, float(m.group(2)) * UNITS.get(m.group(3), 1),
                    float(m.group(4)) * UNITS.get(m.group(5), 1), int(m.group(6)),
                    int(m.group(7)), int(m.group(8))))
    if pts:
        t, done_b, total_b, pct, xf, xt = pts[-1]
        print(f"진행  {human(done_b)} / {human(total_b)} ({pct}%) · 파일 {xf:,} / {xt:,}"
              f" · 마지막 기록 {int((now - t).total_seconds() // 60)}분 전")
        win = [q for q in pts if (t - q[0]).total_seconds() <= 600]
        if len(win) >= 2 and (win[-1][0] - win[0][0]).total_seconds() > 0:
            dt = (win[-1][0] - win[0][0]).total_seconds()
            bps = (win[-1][1] - win[0][1]) / dt
            fps = (win[-1][4] - win[0][4]) / dt
            eta_b = (total_b - done_b) / bps if bps > 0 else None
            eta_f = (xt - xf) / fps if fps > 0 else None
            eta = eta_b if p.group(2) == "images" else (eta_f or eta_b)
            print(f"속도  최근 {dt / 60:.0f}분 평균 {human(bps)}/s · 분당 파일 {fps * 60:,.0f}개"
                  + (f" · 이 단계 남은 시간 약 {dur(eta)}" if eta else " · 멈춰 있음"))
        if p.group(2) == "labels":
            print("      (라벨이 끝나면 이미지 약 140 GiB 를 받는다)")
    elif phases:
        print("진행  rclone 이 목록을 읽는 중 (첫 기록은 약 1분 뒤)")

    print("받은 것 (디스크)")
    for alt, (imgs, labels, size) in count_disk().items():
        print(f"  {alt}: 이미지 {imgs:,}장 ({human(size)}) · 라벨 {labels:,}개")
    free = shutil.disk_usage(RAW).free
    print(f"디스크 여유 {human(free)}")

    if DONE.exists():
        tail = text[text.rfind("완료"):].strip().splitlines()
        for l in tail[1:]:
            print(f"  {l.strip()}")
    elif not script:
        print(f"\n다시 시작 (받은 파일은 건너뛴다):\n  {RESTART}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", type=int, metavar="초", help="이 간격으로 계속 갱신")
    args = ap.parse_args()
    if not args.watch:
        report()
        return
    try:
        while True:
            sys.stdout.write("\033[2J\033[H")
            report()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
