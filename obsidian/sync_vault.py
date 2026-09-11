"""sync_vault.py — 로컬 Obsidian 볼트를 이 저장소로 복사한다.

  python obsidian/sync_vault.py

- 원본  캡스톤/obsidian/Drone  (Obsidian 에서 여는 볼트)
- 사본  _capstone_repo/obsidian/Drone
- 개인 UI 상태(workspace.json)와 백업 파일은 뺀다
- 공개 저장소라 팀 내부 배포 주소는 가린다
"""
import re
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
SRC = REPO.parent / "obsidian" / "Drone"
DST = REPO / "obsidian" / "Drone"

SKIP_NAMES = {"workspace.json", "workspace-mobile.json", "graph.json.bak"}
SKIP_DIRS = {".trash", "plugins"}
REDACT = [
    (re.compile(r"<?https://kumoh-mission-control[^\s>)]*>?"), "(팀 내부 링크 — 공개 저장소라 생략)"),
]


def main():
    if not SRC.is_dir():
        raise SystemExit(f"볼트 없음: {SRC}")
    if DST.exists():
        shutil.rmtree(DST)
    n = 0
    for p in SRC.rglob("*"):
        rel = p.relative_to(SRC)
        if p.is_dir() or p.name in SKIP_NAMES or SKIP_DIRS & set(rel.parts):
            continue
        out = DST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix in (".md", ".canvas", ".base"):
            t = p.read_text(encoding="utf-8")
            for pat, rep in REDACT:
                t = pat.sub(rep, t)
            out.write_text(t, encoding="utf-8")
        else:
            shutil.copy2(p, out)
        n += 1
    print(f"{n}개 파일 → {DST}")


if __name__ == "__main__":
    main()
