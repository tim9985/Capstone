# 비전 파트 지식 볼트 (Obsidian)

비전 파트의 프로젝트 조건 · 데이터셋 · 모델 · 실험 · 결정 · 이슈 · 타임라인을
서로 링크된 노트 56개로 정리한 [Obsidian](https://obsidian.md) 볼트.

## 여는 법

1. 이 저장소를 클론한다
2. Obsidian → **보관함 열기** → `obsidian/Drone` 폴더 선택
3. `00 홈` 노트에서 시작한다

| 볼 것 | 위치 |
|---|---|
| 전체 목차 · 현재 상태 | `00 홈.md` |
| 한 장 지도 | `프로젝트 지도.canvas` |
| 모델 계보도 | `가중치 계보.canvas` |
| 실험 비교표 | `04 실험/실험 비교.base` (Obsidian 1.9 이상) |
| 노트 연결 | 그래프 뷰 — 폴더별로 색이 다르다 |

GitHub 에서도 마크다운으로 읽을 수 있지만 `[[링크]]` · 캔버스 · 비교표는 Obsidian 에서만 동작한다.

## 갱신

로컬 볼트(`캡스톤/obsidian/Drone`)를 고친 뒤:

```bash
python obsidian/sync_vault.py
git add obsidian && git commit -m "볼트 갱신" && git push
```

## 주의

- 수치의 원본은 [drone_yolo](https://github.com/tim9985/drone_yolo) 의 `metrics/*.csv` 와 문서다. 이 볼트는 요약이다
- 연구용 데이터셋(NOMAD · WiSARD · Okutama)의 이미지는 들어 있지 않다
