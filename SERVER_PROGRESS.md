# 서버(3090) 이관 진행 현황

`SERVER.md` · `EXPERIMENTS.md` · `DATASETS.md` 대로 진행하면서 실제로 겪은 것들과
현재 상태를 기록한다. 로컬(노트북) 작업 이어가기 전에 이 파일부터 볼 것.

작성 2026-09-11 · 갱신 2026-09-12 02:50 KST (09-11 17:50 UTC) · Claude Code (se 계정 세션)

---

## 0. 요약

**GPU 복구(재부팅) 완료 → NOMAD 배우 60명으로 데이터 확장 + 회전 증강 반영 →
E1~E8 학습 체인 재시작 → 사용자 결정으로 E1만 마무리하고 E2·E3·E7·E8 중단.**

중단 사유: 목표가 사람 탐지를 넘어 **자세 판별**까지이므로, 1클래스 백본 스윕보다
**서버(3090) 사양에 맞는 자세 판별용 데이터셋·실험 계획을 먼저 세운다.**
모델 크기는 **m · l 을 기본으로** 검토한다 (6절). 계획 초안은 `POSE_PLAN.md`.

⚠ **전처리 전제(화각 · 촬영 각도)가 흔들린다** (8절). 화각은 지터 확대로 대응해 데이터를 재생성했고,
촬영 각도 필터는 아직 적용하지 않았다.

**⚠ 09-12 07:16 KST 무렵 GPU 장애 재발 — M1 `m1_11m_1280` (yolo11m @1280) 은 13에폭에서 중단.**
`nvidia-smi` → `No devices were found`, 재부팅 필요. 24시간 안에 두 번째(E3 와 같은 에러) → 하드웨어(전원·발열) 의심.
중단 시점 모델 평가는 CPU 로 완료했다 (4-1절 결과).

**09-12 ~ 09-14 경과**

- 09-12 09:49 KST 서버가 내려가 이틀간 꺼져 있었다. 09-14 09:20 KST 관리자가 수동으로 켰다.
  - `sudo shutdown`/`reboot` 로 끄면 다시 켜지지 않는다 — 7절 재시작 방법.
- 회의용 짧은 마무리 `finish_m1.sh` 는 **실행되지 않았다** (마감 경과로 폐기).
- 09-14 부팅 후 GPU 정상 (CUDA 테스트 통과).

**▶ 09-14: M1 원래 스케줄로 이어 학습 (사용자 결정)** — `resume_m1.sh`, 로그 `resume_m1.log`
09:40 KST 시작: 전력 제한 300 W 확인 → "Resuming training ... from epoch 14 to 34", 시간 제한 해제 확인.
감시 스크립트 · `gpu_telemetry_20260914_0040.csv` 기록 중.

| 항목 | 값 |
|---|---|
| 방식 | `--resume` — 옵티마이저 상태 · 학습률 스케줄 · batch 3 을 그대로 이어 13 → 34에폭 (남은 21에폭 · 약 6시간) |
| `--time 0` | 체크포인트의 시간 제한(8.56h)을 끈다. 켜 두면 resume 시점부터 다시 재서 에폭 수가 줄어든다 |
| 전력 제한 | **300 W** (사용자가 `sudo nvidia-smi -pl 300` 실행, 스크립트는 적용될 때까지 대기) |
| 기록 | GPU 온도 · 전력 · 스로틀 원인 30초마다 `gpu_telemetry_*.csv` + 감시 스크립트 |
| 이후 | GPU 로 도메인 평가 → `metrics/eval_domain_m1_11m_1280_full.csv` |

**⚠ 10:09 KST GPU 장애 3회차** — 이어 학습 29분 만에 15에폭 84% 지점에서 `CUDA error: unspecified launch failure` → `No devices were found`.

- 저장된 것: `last.pt` = 14에폭 (09:57 KST), `best.pt` = 12에폭. 잃은 것은 15에폭 진행분(약 13분).
- 파이프라인은 설계대로 처리: 학습 rc=1 → GPU 없음 → 평가 생략 · 감시 스크립트 사고 기록.
- **장애 직전 30초 기록 (`gpu_telemetry_20260914_0040.csv`)**
  - 온도 64~65 °C 일정 · 전력 297~299 W (300 W 한도) · 사용률 94~100 % · 팬 75 %.
  - 스로틀 원인은 SW 전력 한도(0x4)뿐, 과열 · HW 감속 신호 없음.
  - 01:09:15 UTC 줄에서 값이 `[GPU is lost]` 로 바뀜.
  - → **과열이 원인은 아니다. 300 W 제한으로도 막지 못했다.**

| 장애 | 실행 | 부하 시작 후 경과 | 전력 제한 |
|---|---|---|---|
| 1회 09-11 | E3 yolo11s@1280 | 약 75분 (14에폭) | 350 W |
| 2회 09-12 07:16 KST | M1 yolo11m@1280 batch 3 | 약 3.6시간 (14에폭 검증) | 350 W |
| 3회 09-14 10:09 KST | M1 이어 학습 | 약 29분 (15에폭 84%) | **300 W** |

진행률 확인: `python train_status.py` (`--watch 60` 으로 자동 갱신) — 09-14 서버용으로 고침.

**▶ 다음: 재시작 후 GPU 안정성 테스트 (사용자 결정, 09-14)** — `gpu_stability_test.sh`

```bash
# 1) 재시작 — 관리자 매뉴얼 그대로 (sudo reboot · shutdown 금지, 7절)
echo 1 | sudo tee /proc/sys/kernel/sysrq
echo b | sudo tee /proc/sysrq-trigger

# 2) 재접속 후
cd ~/JupyterLAB/drone_dev/drone_yolo && nohup ./gpu_stability_test.sh > gpu_stability_test.log 2>&1 &
sudo nvidia-smi -pl 200        # 스크립트가 감지하면 30분 부하. 끝나면 250, 300 도 같은 식으로 실행
```

- 부하: 실제 학습 조건(yolo11m@1280 · batch 3 · 검증 포함). M1 체크포인트 보호용으로 별도 이름 `stabtest_<W>W`.
- GPU 가 사라지면 그 단계에서 종료. 결과 `gpu_stability_<시각>.txt`, 10초 간격 기록 `gpu_telemetry_stab_*.csv`.
- 한계: 장애 간격이 29분 ~ 3.6시간이라 30분 통과는 약한 증거다. 통과한 최저 단계로 더 길게 확인할 것.
- M1 은 `last.pt`(14에폭)에서 테스트 결과를 보고 `resume_m1.sh` 로 다시 이어간다.
  재실행 시 `POWER_TARGET` 을 통과한 전력으로 바꿔야 한다.

**안정성 테스트 결과 (09-14, 요약 `gpu_stability_20260914_0210.txt`)**

| 단계 | 시각 (KST) | 결과 | 부하 | 온도 평균 / 최고 | 전력 평균 | 속도 |
|---|---|---|---|---|---|---|
| 200 W | 11:21 ~ 11:53 | ✅ 통과 (rc=0) | 32분 · 2에폭 (검증 2회 포함) | 55.1 / 56 °C | 195 W | 5.1 it/s (중앙값) |
| 250 W | 12:08 ~ 12:40 | ✅ 통과 (rc=0) | 31분 · 2에폭 (검증 2회 포함) | 59.6 / 61 °C | 246 W | 6.9 it/s (중앙값) |
| 300 W | 12:46 ~ 13:17 | ✅ 통과 (rc=0) | 31분 · 2에폭 (검증 2회 포함) | 64.2 / 65 °C | 294 W | 7.5 it/s |

**결론 — 30분 테스트로는 장애가 재현되지 않았다.**
- 300 W 도 통과했다. 같은 300 W 에서 3회차 장애가 29분에 났으므로, 장애는 특정 전력을 넘으면 반드시 나는 게 아니라 **확률적으로** 난다.
- 전력 제한이 효과가 있는지는 이 테스트로 확인되지 않았다.
- 원인을 가리려면 커널 로그의 Xid 코드가 필요하다: `sudo grep -hE "NVRM|Xid|fallen off" /var/log/kern.log.1 /var/log/kern.log | tail -40`.

**▶ 09-14 13:20 KST: M1 이어 학습 재시작 — 250 W (사용자 결정)** — `resume_m1.sh`(`POWER_TARGET=250`), 로그 `resume_m1_250w.log`

- 시작점: `last.pt` = 14에폭 → 15 ~ 34에폭 (20에폭).
- 250 W 속도 약 7.0 it/s 기준 에폭당 약 17분 → **약 5.7시간**.
- 끝나면 GPU 로 도메인 평가 → `metrics/eval_domain_m1_11m_1280_full.csv`.
- 스크립트는 사용자의 `sudo nvidia-smi -pl 250` 적용을 기다린 뒤 시작한다.
- 테스트 스크립트 3개(`resume_m1.sh` · `gpu_stability_test.sh` · `train_status.py`)의 감시 스크립트 확인은 `^bash gpu_watchdog\.sh` 로 고정해 자기 매칭을 막았다.

**⚠ 13:48 KST GPU 장애 4회차 — 250 W 에서 부하 약 1분 만에**

- 13:47:52 전력 제한 250 W 확인 → 이어 학습 시작.
- 13:48:22 기록: 54 °C · 248.6 W · 사용률 95 % (정상).
- 13:48:46 `CUDA error: unspecified launch failure` → `No devices were found`. 15에폭 4 %(284/6307) 지점.
- `last.pt` 는 14에폭 그대로 (에폭 도중 장애라 덮어쓰지 않음). 기록 `gpu_telemetry_20260914_0447.csv`.

| 장애 | 실행 | 부하 후 경과 | 전력 제한 | 직전 온도 |
|---|---|---|---|---|
| 1 · 09-11 | E3 yolo11s@1280 | 약 75분 | 350 W | — |
| 2 · 09-12 07:16 | M1 yolo11m@1280 | 약 3.6시간 | 350 W | — |
| 3 · 09-14 10:09 | M1 이어 학습 | 약 29분 | 300 W | 65 °C |
| 4 · 09-14 13:48 | M1 이어 학습 | **약 1분** | **250 W** | **54 °C** |

같은 날 11:21 ~ 13:17 의 안정성 테스트(200/250/300 W 각 31분)는 모두 통과했다.

**결론: 전력 제한은 해결책이 아니다.** 과열도 아니고 부하 시간과도 무관하게 무작위로 난다 → 하드웨어(전원부 · PCIe · GPU) 또는 드라이버(GSP) 문제로 좁혀진다.
원인 코드(Xid)는 **재시작 전에** `sudo dmesg` 로 확보해야 한다 — sysrq `b` 는 디스크 동기화 없이 재시작하므로 `kern.log` 에 마지막 커널 메시지가 안 남을 수 있다.

### 원인 코드 확인 — Xid 79 "GPU has fallen off the bus" (4회차, 사용자가 재시작 전 `sudo dmesg` 로 확보)

```
[Mon Sep 14 04:48:45 2026] NVRM: Xid (PCI:0000:02:00): 79, pid=9962, name=python, GPU has fallen off the bus.
[Mon Sep 14 04:48:45 2026] NVRM: GPU0 GSP RPC buffer contains function 78 (DUMP_PROTOBUF_COMPONENT) ...
[Mon Sep 14 04:48:45 2026] NVRM: Xid (PCI:0000:02:00): 154, GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)
```

| 메시지 | 의미 |
|---|---|
| **Xid 79** | **1차 원인.** GPU 가 PCIe 버스에서 사라짐 |
| GSP RPC buffer / history | 결과. 사라지는 순간 처리 중이던 GSP 요청(`GSP_RM_CONTROL`) 기록. GSP 가 원인이 아니다 |
| Xid 154 Node Reboot Required | 결과. 버스에서 떨어진 GPU 는 재부팅으로만 복구된다는 드라이버 판정 (1회차에서도 봤던 메시지) |

**좁혀진 원인 (NVIDIA 기준 Xid 79 = 하드웨어 계열)**
- **배제됨: 과열** — 4회차는 부하 1분 · 54 °C 에서 났다. GDDR6X 메모리가 달궈질 시간도 없었다.
- **배제되지 않음: 전원 순간 피크** — `nvidia-smi -pl` 은 평균 전력만 제한한다. 3090 의 밀리초 단위 순간 피크(평균의 약 2배)는 못 막는다.
- **배제되지 않음: PCIe 연결** — Gen4 신호 품질 · 슬롯 · 라이저. `pci=noaer` 부팅 옵션이 PCIe 오류 보고를 숨겨 증거가 안 남는다.
- **배제되지 않음: GPU 카드 자체 결함**
- **우선순위 낮음: GSP 펌웨어 끄기** — GSP 메시지는 결과였다.

**관리자 점검 요청 목록 (하드웨어 · BIOS — sudo 로 할 수 없는 것)**
1. 파워서플라이 용량(3090 + i9 기준 850 W 이상 권장) · GPU 8핀 전원을 **케이블 2가닥으로 따로** 연결했는지 (한 케이블에서 갈라 쓰면 순간 피크에 취약)
2. GPU 재장착 · 라이저 케이블 사용 여부 · 슬롯 점검
3. BIOS 에서 해당 슬롯 **PCIe Gen3 고정** (Gen4 신호 불안정 대응, 학습 속도 손실 작음)
4. 진단용으로 `pci=noaer` 를 잠시 빼고 부팅 → PCIe 오류(AER)가 쌓이는지 확인
5. 가능하면 다른 슬롯 · 다른 파워 · 다른 GPU 로 교차 확인

**그 전까지:** 장시간 학습을 반복하지 않는다. 매번 재시작이 필요하고, 체크포인트로 잃는 건 적어도 진행이 안 된다.

**4회 전부 Xid 79 확인** (사용자가 `sudo grep -hE "Xid|fallen off" /var/log/kern.log.1 /var/log/kern.log` 로 확인)

| 회차 | 시각 (UTC) | 로그 |
|---|---|---|
| 1 | 2026-09-11 09:47:00 | Xid 79 → Xid 154 |
| 2 | 2026-09-11 22:16:44 | Xid 79 → Xid 154 |
| 3 | 2026-09-14 01:09:15 | Xid 79 → Xid 154 |
| 4 | 2026-09-14 04:48:45 | Xid 79 → Xid 154 |

→ 같은 하드웨어 원인이 반복되는 것으로 판단. **관리자 전달용 보고서: `GPU_XID79_REPORT.md`**.

**▶ 09-14 14:20 KST 무렵: 소프트웨어 완화 설정 + M1 이어 학습 (사용자 결정)** — 로그 `resume_m1_mitig.log`

관리자 점검 전, 남은 두 가설을 소프트웨어로 완화하고 M1 을 이어간다. 모든 설정은 재부팅하면 원래대로 돌아간다.

| 설정 | 명령 (사용자 실행, 이 순서) | 겨냥하는 가설 | 비용 |
|---|---|---|---|
| PCIe Gen3 고정 | `sudo setpci -s 00:01.0 CAP_EXP+30.w=0003:000f` → `sudo setpci -s 00:01.0 CAP_EXP+10.w=0020:0020` | PCIe Gen4 신호 불안정 | 거의 없음 |
| GPU 코어 클럭 상한 | `sudo nvidia-smi -lgc 300,1600` (최대 2130 MHz 의 약 75 %) | 전원 순간 피크 | 속도 −15~25 % 예상 |
| 전력 제한 | `sudo nvidia-smi -pl 250` (**마지막** — 스크립트가 이걸 감지해 시작) | (평균 전력) | −7~11 % |

- GPU 는 CPU 직결 슬롯(상위 포트 `00:01.0`, x16, 최대 Gen4). 쉬는 동안은 2.5 GT/s 로 내려가 있는 게 정상.
- 적용 확인은 부하 중 GPU 기록(`gpu_telemetry_*.csv`)으로 한다:
  - `pcie.link.gen.current` = 3 이면 Gen3 적용.
  - `clocks.current.graphics` ≤ 1600 MHz 이면 클럭 상한 적용.
- 되돌리기: `sudo nvidia-smi -rgc` · `sudo nvidia-smi -pl 350` · 링크는 재부팅.
- 한계: 둘을 함께 걸어서 버티더라도 무엇이 효과였는지는 가릴 수 없다. 끝까지 버티지 못하면 전원부 · 카드 결함 쪽으로 좁혀진다.

**적용 확인 (14:22:50 KST 학습 시작, 부하 중 측정)**

| 항목 | 결과 |
|---|---|
| PCIe | `current_link_speed` **8.0 GT/s x16**, `pcie.link.gen.current` = 3 → Gen3 적용 ✅ |
| 코어 클럭 | 1470 ~ 1590 MHz (상한 1600) → 적용 ✅ |
| 전력 | 235 ~ 249 W / 250 W |
| 온도 · 팬 | 56 ~ 59 °C · 63 ~ 66 % (과열 한계 95 °C) |
| 학습 | 15에폭부터 재개, 초기 6.5 it/s |

- 부하 시작 직후 기록에 팬 0 % 가 찍혔다. 이 카드는 약 50 °C 미만에서 팬을 멈추는 무소음 모드라, 온도가 오르기 전 값이다. 1분 뒤 63 % 로 정상 동작.
- 4회차 장애 시점(부하 약 1분)은 넘겼다.

### 관리자 답변과 GPU 자동 복구 (`autoheal/`, 09-14)

**관리자 답변 (09-14)**
- 서버 사양 문제라 하드웨어로는 해결 방법이 없다.
- 재부팅 명령을 자동화하면 복구가 빨라진다.
- 서버 사용자가 한 명뿐이니 작업을 나눠 학습해도 된다.
- 재부팅은 매뉴얼 방식(sysrq)으로 한다. `sudo reboot` 은 서버가 다시 켜지지 않는다.

→ 목표를 "장애를 막기"에서 **"죽어도 사람 없이 복구 · 이어 학습"** 으로 바꿨다 (사용자 결정: 전부 자동, `sync` 추가).

| 파일 | 역할 |
|---|---|
| `autoheal/gpu_autoheal.sh` | root 데몬. 시작 시 완화 설정 · 학습 재개, 감시 중 GPU 소실(Xid 79) 확정 시 sync → sysrq 1 → b |
| `autoheal/gpu-autoheal.service` | systemd 서비스 (`KillMode=process` — 서비스 재시작에도 학습은 유지). 관리자 `nvidia-powerlimit.service` 뒤에 실행 |
| `autoheal/job.conf` | 자동 재개할 학습 (`JOB_NAME=m1_11m_1280`, `JOB_SCRIPT=resume_m1.sh`, 평가 CSV) |
| `autoheal/install.sh` | 설치 · 제거 (`sudo bash autoheal/install.sh [--uninstall]`) |

- **안전장치**
  - `autoheal/ENABLED` 가 없으면 아무것도 안 한다.
  - 2시간에 자동 재부팅 3회면 멈추고 `autoheal/HALTED` 를 남긴다.
  - Xid 79 없이 nvidia-smi 만 실패하면 재부팅하지 않는다.
  - 학습 · 평가는 se 계정으로 실행한다.
  - `last.pt` 는 새로 저장될 때마다 검사 후 `last_backup.pt` 로 백업하고, 재개 전에 깨졌는지 확인한다.
  - 학습 중에는 PCIe 링크 재협상을 하지 않는다.
- **관리자 서비스와의 충돌 대응**
  - `nvidia-powerlimit.service` 가 부팅 때 `nvidia-smi -pl 350` 을 건다.
  - 대응 1: 서비스 순서를 그 뒤로 둔다.
  - 대응 2: 감시 중 전력 제한이 250 W 에서 벗어나면 다시 건다.
- 점검: `bash autoheal/gpu_autoheal.sh --selftest` (root 불필요 · 실제 조치 없음). 09-14 통과 — 실행 중 학습 건너뜀 · 백업 · 재부팅 3회 뒤 HALTED 확인.
- 기록: `autoheal/logs/autoheal.log`. 레포의 스크립트를 고치면 `install.sh` 를 다시 실행해야 반영된다.

**설치 점검 (09-14 15:08 KST, 사용자가 15:04 설치)**

| 항목 | 결과 |
|---|---|
| 서비스 | `enabled` · `active` · `KillMode=process` · `Restart=always` · `After=` 에 `nvidia-powerlimit.service` 포함 |
| 설치 파일 | `/usr/local/sbin/gpu_autoheal.sh` · `/etc/systemd/system/gpu-autoheal.service` 모두 레포와 동일 |
| PCIe | root 로 설정 레지스터를 직접 읽어 `LnkCtl2=0x0003` (Gen3) 확인 → 재협상 안 함 |
| 클럭 상한 | `GPU clocks set to (gpuClkMin 300, gpuClkMax 1600)` — GeForce 에서도 적용됨 |
| 학습 | 실행 중 감지 → 건드리지 않음 (PID 7060 그대로, 17에폭 진행) |
| 백업 | `last.pt` 읽기 검사 통과 → `last_backup.pt` 생성 |

- 아직 검증 못 한 것: 실제 장애 시 자동 재부팅 → 부팅 후 설정 · 재개 흐름 (장애가 나야만 확인 가능).
- GPU 소실 판정 정규식은 실제 커널 로그 줄로 시험했다.
  - 매칭: Xid 79 두 형식.
  - 불일치: Xid 154, 네트워크 카드의 `XID 641`.

### origin/main 병합 · 커밋 작성자 정정 · 계획 검토 (09-14)

- **병합**: 노트북 인계 `386a787`(docs/server_plan S0~S8 · CLAUDE.md 09-13 확정 · WEIGHTS.md · check_env.sh)을 병합했다 (사용자 승인).
  - 서버에 스테이징돼 있던 CLAUDE.md · REBOOT_PLAN.md (옛 사본) · DATASETS.md (main 과 동일)는 main 쪽을 채택.
- **작성자 정정**: d647e22 · 병합 커밋이 서버 계정 전역 git 설정 `rohyenwu` 로 기록돼 있었다 (사용자 아님).
  - 두 커밋 모두 미push 라 내용 · 날짜는 그대로 두고 작성자만 `Seoin Jung <timjjang0914@naver.com>` 으로 다시 만들었다 → 병합 커밋 **a7b43d5**.
  - 저장소 로컬 git 설정을 사용자로 바꿨다. 전역 `~/.gitconfig` 는 그대로 둠.
- **계획 검토**: `docs/server_plan_review_0914.md`.
  - 방향은 그대로 가능.
  - GPU 장애 → 재개 가능한 대기열 + autoheal 이 필요하다.
  - S0 재현에는 노트북 val 크롭이 필요하다.
  - S1 원본(NOMAD 50·70 m · SARD · Okutama)이 서버에 없다.
  - S4 11m · p2 는 batch 8 로 줄여야 한다.
  - 실험 1개는 약 4.5~5시간 걸린다.

참고: 2회차 장애가 검증 중에 났는데, 200 W 는 검증 2회를 넘겼다. 다만 30분 통과는 약한 증거다.
테스트 중 `gpu_watchdog.sh` 는 시작 시 자기 매칭 오판으로 꺼져 있다가 11:23 에 수동으로 켰다.

---

## 1. 환경

- 레포 경로가 `~/drone_dev` → **`~/JupyterLAB/drone_dev/drone_yolo`** 로 이동했다.
  `data/` 는 `~/JupyterLAB/drone_dev/data` 를 가리키는 심볼릭 링크 (레포 밖, `.gitignore`).
- conda env `drone` (python 3.10), `/home/se/miniconda3/envs/drone/bin/python`
- torch 2.13.0+cu130, ultralytics 8.4.102, opencv 5.0.0.93 — `requirements.txt`대로 설치됨
- **주의**: 이 머신은 `/home/se/JupyterLAB/bin`이 `PATH`에서 conda보다 앞에 있다.
  `pip install`을 conda env 활성화 없이 돌리면 JupyterLab 쪽 venv로 잘못 들어간다.
  반드시 `/home/se/miniconda3/envs/drone/bin/pip` 절대경로로 설치할 것.
- 드라이버 580.173.02 / CUDA 13.0 / RTX 3090 24GB. 서버 접속 초기엔 드라이버 커널모듈
  버전 불일치(580.159 로드됨 vs 580.173 유저스페이스)로 `nvidia-smi`가 죽어있었는데
  `sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia_uvm`로
  재부팅 없이 해결했었다 (6절의 Xid 154 장애와는 다른 문제).

## 2. 데이터

| 데이터셋 | 상태 | 방법 |
|---|---|---|
| **NOMAD** | **배우 60명** (1~30 전원 + 31~100 중 30명), 거리 a10/a30만 | `rclone gdrive:` + `--filter-from` (`nomad_filter_new30.txt`). `gdown --folder`는 API 레이트리밋으로 실패했었음 |
| **WiSARD** | VIS 전체(39개 비행), IR 삭제 | `gdown`으로 43.5GB zip, 압축해제 후 IR 폴더 삭제 |
| SARD | **미확보** | Roboflow API 키 필요. pose3(3클래스 자세) 실험엔 필수 |
| Okutama | **미확보** | 공식 Dropbox 막힘. 노트북에서 옮겨야 함 (평가 전용이라 탐지 학습은 안 막힘) |

가중치: `weights/` 에 `yolov8s_stage1_all.pt`(안전자산) · `yolov8s_pose3_sn_freeze.pt` ·
`yolov8s_pose3_nd_freeze.pt` · `yolov8s_sard2_pose6.pt` 반입 완료 (09-11 12:17 UTC).

### 현재 학습 데이터 (`configs/data_all.yaml`, nc=1 person)

| 배치 | train | val | val 배우 |
|---|---:|---:|---|
| `nomad_actor01_10` | 1,331 | 298 | 004 · 008 |
| `nomad_actor11_20` | 1,128 | 298 | 014 · 018 |
| `nomad_actor21_30` | 1,096 | 268 | 024 · 028 |
| `nomad_actor_sel31_100` (신규 30명) | 3,645 | 800 | 048 · 059 · 071 · 079 · 088 · 094 |
| `wisard` | 11,737 | 3,304 | (비행 단위 분할) |
| **합계** | **18,937** | **4,968** | |

신규 30명 = `nomad_new_actors.txt` (035 · 039 · 040 · 042 · 043 · 044 · 048 · 051 · 053 · 057 ·
059 · 065 · 068 · 070 · 071 · 073 · 076 · 077 · 078 · 079 · 084 · 087 · 088 · 090 · 092 · 093 ·
094 · 095 · 097 · 100).

### ⚠ 문서와 어긋나는 점 (확인 필요)

1. **배우 수가 42명이 아니라 60명이다.** `metrics/nomad_actor_selection.csv` 42명은
   1~30 중 12명 + 31~100 중 30명인데, 실제 학습에는 **1~30 전원 30명 + 신규 30명**이
   들어갔다. `data_all.yaml` 주석("선정 42명: 배우1~30 + 31~100 중 30명")도 숫자가 안 맞는다.
   `EXPERIMENTS.md` E4 (30→42명)와 조건이 다르다는 점을 기록해 둔다.
2. **검증 배우가 `CLAUDE.md` §4-1 시뮬레이션 값과 다르다.** 배치별로 따로 전처리해서
   1~30 쪽 val(004 · 008 · 014 · 018 · 024 · 028)이 기존 분할 그대로 유지됐다.
   노트북 분할과 같다면 `stage1_all` 을 이 val 로 재도 학습 배우 오염은 없다 — 노트북 쪽에서
   `val_actors` 대조 필요.

### NOMAD 전처리시 겪은 함정
- `nomad_prep.py`는 `--out` 하나만 받고 배우 범위 필터가 없다. 배치별로 다른 서브셋
  디렉토리를 가리키게 해야 `nomad_actor01_10/11_20/21_30/sel31_100` 구조가 나온다.
- 배치 디렉토리를 **심볼릭 링크**로 구성했더니 Python 3.10의 `Path.rglob('**')`가
  심볼릭 링크된 디렉토리를 따라가지 않아 이미지 0장으로 나왔다(3.13부터 해결된 이슈).
  **하드링크(`cp -al`)**로 바꿔서 해결 — 디스크 추가 사용 없음.

### WiSARD 원본 라벨 품질 이슈 (발견, 미수정)
`data/raw/WiSARD/**/*.txt` 전수조사 결과 **20,541개 중 5,599개(27.3%)**가 완전히 동일한
줄이 중복 저장돼 있다(좌표까지 일치). Ultralytics 데이터로더가 캐싱 시점에 자동으로
제거해줘서 학습엔 무해하지만, 원본 배포 자체의 결함으로 보인다.

## 3. 코드 수정

- `train_person.py`, `train_pose_cls.py`: 노트북 하드코딩 `workers=4`, `cache=False`를
  인자로 뺐다(`--workers` 기본 8). `--batch`가 **소수(0~1)면 그 비율만큼 VRAM 목표로
  자동산정** (ultralytics `check_train_batch_size`의 `fraction`).
  `--batch -1`(목표 60%)은 너무 보수적이라 **`--batch 0.85`** 를 쓴다.
- **stage1 증강 프로필: `degrees=180.0`, `flipud=0.5`** (하향 90° 시점엔 위쪽이 없다).
  → 이전 E1·E2·E3 결과와 비교 불가, 전부 재실행.
- `cache='disk'`: 데이터가 RAM(62GB)을 넘어 `ram` 은 어차피 disk 로 폴백되므로 명시.
- `make_configs.py`, `nomad_prep.py`, `wisard_prep.py`: 이동한 경로 · 신규 배치 반영.
- `nomad_prep.py` · `wisard_prep.py`: `mp.Pool` 병렬화 + 이미지별 결정적 시드(SEED+순번) + `--workers`.
  `--limit 200` 두 번 실행해 md5 동일 확인 (Drone-우분투 세션).
- `train_person.py`: 가중치 복사 파일명의 고정 `yolov8s_` 접두사 제거 → `weights/{name}.pt`.
- `wait_and_preprocess.sh`: `while` + `<(cat)` 무한루프(하드링크 중복) 수정.

### 자동화 스크립트 (서버에서 동작 중/완료)

| 스크립트 | 역할 | 상태 |
|---|---|---|
| `wait_and_preprocess.sh` | 신규 배우 다운로드 완료 대기 → 전처리 → `make_configs.py` | ✅ 09-11 14:55 UTC 완료 |
| `wait_and_train.sh` | 전처리 결과 검증 → 기존 runs 백업 → 학습 체인 시작 | ✅ 14:56 UTC 검증 통과 |
| `run_remaining_experiments.sh` | E1→E2→E3→E7→E8 순차 실행 (로그 `train_chain_final2.log`) | ⏹ 17:47 UTC 스크립트만 종료 (E1 프로세스는 계속 실행, 이후 실험 미실행) |
| `gpu_watchdog.sh` | `nvidia-smi` 30초 주기 확인, 연속 2회 실패 시 체인 종료 + `gpu_watchdog_incident.log` 기록. 재부팅/모듈 리로드는 절대 자동으로 안 함 | ▶ 12:02 UTC부터 실행 중, 사고 0건 |

## 4. 학습 — 2차 스윕 (현재)

체인 시작 2026-09-11 14:58 UTC (23:58 KST). 모두 COCO 사전학습에서 시작, 60에폭 · patience 15 ·
batch 0.85 (AutoBatch → 13).

| 실험 | 조건 | 상태 | best mAP50 | best mAP50-95 |
|---|---|---|---:|---:|
| **E1** | yolo11s / 960 | ✅ 42에폭 조기종료 (3.2시간, 09-11 18:12 UTC) | 0.618 (27ep) | 0.324 (27ep) |
| E2 | yolov8s / 1280 | ❌ 중단 (사용자 결정) | - | - |
| E3 | yolo11s / 1280 | ❌ 중단 (사용자 결정) | - | - |
| E7 | yolo11m / 1280 | ❌ 중단 (사용자 결정) | - | - |
| E8 | yolo11l / 1280 | ❌ 중단 (사용자 결정) | - | - |

best 는 ultralytics fitness(0.1·mAP50 + 0.9·mAP50-95) 기준. E1 경과:

| 에폭 | mAP50 | mAP50-95 | fitness |
|---:|---:|---:|---:|
| 9 | 0.605 | 0.281 | 0.313 |
| 17 | 0.632 | 0.321 | 0.352 |
| **27** | **0.618** | **0.324** | **0.353** |
| 30 | 0.611 | 0.319 | 0.348 |

- 로그에 CUDA OOM 2건이 보이지만 AutoBatch 탐색 중 발생한 것으로 **정상**이다.
- AutoBatch 는 batch 13 을 19.2GB 로 예측했지만 실제 사용은 약 7.2GB 다. GPU 사용률은 99%라
  병목은 아니나, 메모리 여유가 크다 — 다음 스윕에서 batch 를 수동으로 올려볼 여지.

### E1 종료 시 크래시 (학습 결과는 정상)

42에폭 학습을 마친 뒤 `best.pt` 최종 검증에서 PR 곡선을 그리다 죽었다.

```
ImportError: Cannot load backend 'tkagg' which requires the 'tk' interactive framework, as 'headless' is currently running
```

- `runs_person/e1_11s_960/weights/best.pt` 는 정상 저장됐다.
- `weights/` 복사만 빠졌다 → 수동으로 `weights/e1_11s_960.pt` 에 복사했다.
- 재발 방지: 학습을 띄우는 셸에서 `export MPLBACKEND=Agg`.

## 4-1. 밤사이 1회 학습 — M1 `m1_11m_1280` (09-12)

회의 전 하룻밤에 모델 하나를 학습한다. 스크립트 `overnight_m1280.sh`, 로그 `overnight_m1280.log`.

| 항목 | 값 |
|---|---|
| 모델 | yolo11m (COCO 사전학습) · 1클래스 · imgsz 1280 · batch 0.85 |
| 데이터 | NOMAD 60명 + WiSARD VIS 전체. **재생성**: 화각 지터 (0.65, 1.45), 촬영 각도 필터 없음 |
| 분할 | 기존과 동일. 스크립트가 val 배우 · val 비행을 백업본과 대조해 검증 |
| 학습 시간 | ultralytics `time` — 마감 12:15 KST (약 8.5시간). 에폭 수 · 학습률 스케줄은 자동 조정 |
| 증강 | stage1 프로필 그대로 (degrees 180 · flipud 0.5) + `close_mosaic 5` (에폭이 적어 10 → 5) |
| 이후 | `eval_domain.py` 도메인별 평가 2건 → `metrics/eval_domain_*.csv` · M1 @1280 · E1 @960 (둘 다 새 val 크롭) |

### 진행 순서와 안전장치

1. `data/det` → `data/det_v1_fov60` 백업 이동. 옛 `.npy` 캐시가 새 라벨에 섞이는 것을 막는다.
2. NOMAD 4개 배치 + WiSARD 재생성 → `make_configs.py`.
3. 검증 실패(빈 폴더 · `.npy` 잔존 · val 분할 불일치) 시 학습하지 않고 `overnight_incident.log` 에 기록한다.
4. 학습 → 평가.

### 준비 중 발견 · 수정한 것

| 문제 | 영향 | 조치 |
|---|---|---|
| `data/raw/NOMAD_b*` 의 json 심볼릭 링크가 옛 경로(`~/drone_dev`)를 가리켜 끊김 | NOMAD 1~30 재전처리 불가 | 링크 재연결 + 스크립트가 매번 재연결 |
| `eval_domain.py` 가 `--imgsz` 를 받고도 960 고정으로 평가 | 1280 모델을 960 으로 잼 | 전역값 반영 + `--out` 추가 (기존 CSV 덮어쓰기 방지) |
| `train_person.py` 일반 분기의 가중치 복사명에 `yolov8s_` 고정 접두사 | 파일명 오해 | `weights/{name}.pt` 로 통일 |
| 스크립트의 `pgrep -f train_person.py` 가 **스크립트를 띄운 셸 명령줄**까지 잡음 | 거짓 "중복 실행" 으로 두 번 중단 (데이터 무손상) | python 실행 파일만 검사, E1 대기는 PID 기준 |
| `train_person.py` 에 시간 제한 옵션 없음 | — | `--time`, `--close-mosaic` 추가 |

### 결과 — GPU 장애로 13에폭에서 중단 (09-12)

| 항목 | 값 |
|---|---|
| 배치 | AutoBatch → **3** (16 GB 예측 · 실사용 6 GB). 약 15.5분/에폭, 시간 예산 기준 34에폭 예정 |
| 중단 | 14에폭 학습 후 검증 중 `CUDA error: unspecified launch failure` (07:16 KST 무렵). 감시 스크립트 07:17 감지 |
| 이후 | `nvidia-smi` → `No devices were found`. **재부팅 필요** |
| 남은 가중치 | `runs_person/m1_11m_1280/weights/best.pt` (12에폭) · `last.pt` (13에폭) |
| 평가 | GPU 없이 **CPU 로** 수행 (값은 같고 시간만 약 100분) |

학습 중 검증(data_all val 4,968장): 12에폭 mAP50 0.616 / mAP50-95 0.316. 중단 직전까지 계속 오르는 중이었다.

**도메인별 (새 val 크롭 · `eval_domain.py` · 쓰러짐 재현율 conf 0.15)**

| 도메인 | 장수 | M1 (11m·1280·12ep) mAP50 / 50-95 | E1 (11s·960·27ep) mAP50 / 50-95 |
|---|---:|---:|---:|
| nomad_summer (4명) | 593 | 0.607 / 0.360 | **0.686 / 0.423** |
| nomad_holdout10 | 1,396 | 0.614 / 0.350 | **0.665 / 0.386** |
| wisard_sept (실제로는 1월 외 전체) | 2,614 | **0.516 / 0.196** | 0.489 / 0.188 |
| wisard_jan | 642 | **0.871 / 0.590** | 0.860 / 0.569 |
| combined | 3,849 | **0.618 / 0.313** | 0.601 / 0.306 |

| 활동별 재현율 (holdout10) | 표본 | M1 | E1 |
|---|---:|---:|---:|
| Hiding (Laying) — 쓰러짐 | 276 | 0.638 | **0.746** |
| Walking | 311 | 0.836 | **0.878** |
| Hiding | 806 | 0.495 | **0.566** |

해석
- M1 은 34에폭 스케줄의 12에폭(학습률 높음 · 모자이크 켜짐 · 미수렴)에서 멈춘 모델이다.
  **"m·1280 이 s·960 보다 못하다"는 결론은 낼 수 없다.**
- 그 상태로도 WiSARD(겨울 · 기타)는 E1 을 앞섰고, NOMAD(여름 농장)와 쓰러짐 재현율은 뒤졌다.
- 크기 · 해상도 · 지터 세 변수가 동시에 바뀐 비교라 원인은 분리할 수 없다.

원자료: `metrics/eval_domain_m1_11m_1280.csv`, `metrics/eval_domain_e1_11s_960_newval.csv`,
로그 `overnight_m1280.log` · `eval_m1_11m_1280.log` · `eval_e1_newval.log` · `gpu_watchdog_incident.log`.

### 참고 — 1차 스윕 결과 (증강 변경 전 · 배우 30명 · val 3,900장, **지금 수치와 비교 불가**)

| 실험 | 조건 | 결과 | mAP50 | mAP50-95 |
|---|---|---|---:|---:|
| E1 | yolo11s / 960 | 54ep 조기종료 | 0.647 | 0.340 |
| E2 | yolov8s / 1280 | 31ep 조기종료 | 0.639 | 0.350 |
| E3 | yolo11s / 1280 | 14ep에서 GPU 크래시 | 0.641 (13ep) | 0.341 |

백업 위치: `runs_person/*_before_20260911_1456/`, `runs_person/e3_11s_1280_crashed_13ep/`,
`runs_person/e1_11s_960_batch9_old/`.

## 5. 예상 일정

| 실험 | 예상 |
|---|---|
| 단계 (M1) | 예상 (09-12 KST) |
|---|---|
| 전처리 재생성 · 검증 | 03:18 ~ 03:40 |
| M1 학습 | ~03:40 ~ 12:15 (마감) |
| 최종 검증 · 도메인 평가 2건 | ~12:15 ~ 12:50 |

## 6. 다음에 할 일

1. E1 종료 후 `runs_person/e1_11s_960/` 의 `results.csv` · `best.pt` 백업 (대여 서버).
2. **자세 판별(person / fallen / ambiguous)까지 포함한 데이터셋·실험 계획 수립** — 3090 사양 기준.
   - 모델 크기는 m · l 을 기본으로 검토 (사용자 방향).
   - 함께 따져볼 것:
     - 쓰러짐 판별이 SARD 밖에서 무너지는 문제(SARD 0.974 → Okutama 0.091)는 데이터 다양성 문제다.
     - 자세 데이터가 작으면 큰 모델이 과적합할 수 있다.
     - 추론은 중앙 서버에서 하므로, 그 서버에서 NFR-V01(1280×720 ≤30 ms)을 지키는지 확인해야 한다.
   - 선행 조건: SARD 확보 (Roboflow 키), Okutama 서버 반입 여부 결정.
3. NOMAD 배우 수를 42명(선정 목록)으로 할지 60명(현재)으로 할지 결정 (2절 ⚠).
4. 노트북 기준선과 공정 비교용 홀드아웃 배우 확정 (`CLAUDE.md` §4-1).

## 7. 해결됨 — GPU 하드웨어 장애 (09-11)

E3 14에폭 시작 직후 `torch.AcceleratorError: CUDA error: unspecified launch failure`
발생, 직후 `nvidia-smi`가 `No devices were found`로 GPU를 완전히 잃음.

```
NVRM: Xid (PCI:0000:02:00): 154, GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)
```

드라이버가 "재부팅 필요"로 판정(GSP RPC 타임아웃 계열). `rmmod`도 걸린 채 `SIGKILL`에도
안 죽어 재부팅 외엔 복구 불가. 서버를 공유하는 `bz2149` 계정의 장기 CPU 작업 때문에
재부팅을 조율해야 했다.

**2026-09-11 11:31 UTC 재부팅으로 복구.** 재발 대비로 `gpu_watchdog.sh` 를 붙였다 (3절).
재발 시: `gpu_watchdog_incident.log` 확인 → 동시 사용자와 재부팅 조율 →
`python train_person.py --resume --name <죽은 실험 이름>` 으로 이어서 학습.

### 서버 재시작 방법 — `sudo reboot` · `sudo shutdown` 금지 (관리자 공지, 09-14 확인)

- **알려진 문제**: 무거운 프로그램을 실행하면 GPU 가 프로세스 진행을 멈추고 인식되지 않는다. 09-11 E3 · 09-12 M1 장애가 이것이다.
- **`sudo shutdown now` 등으로 끄면 서버를 사람이 직접 다시 켜야 한다.** 09-12 에 이렇게 이틀간 꺼져 있었다.
- 공지된 재시작 방법 — 커널 수준 강제 재시작:

```bash
echo 1 | sudo tee /proc/sys/kernel/sysrq
echo b | sudo tee /proc/sysrq-trigger
```

> `b` 는 디스크 동기화 없이 즉시 재시작한다. 학습이 체크포인트를 쓰던 중이면 마지막 기록이 날아갈 수 있으므로,
> 그 전에 `sync` 와 `echo s | sudo tee /proc/sysrq-trigger`(동기화)를 먼저 하는 것을 권장한다. (공지에는 없는 이 세션의 권고)

### 중복 학습 사고 (09-11, Drone-우분투 세션)

대기 스크립트를 kill 해도 **이미 띄운 `train_person.py` 는 고아로 남아** 같은 `--name` 으로 두 개가 돌았다.
전부 종료 · 손상 폴더 삭제 후 단일 재시작했다.
**학습을 다시 띄우기 전에 `pgrep -af train_person.py` 로 단일 인스턴스인지 확인할 것.**
(17:47 UTC 체인 중단 때도 같은 원리로 E1 만 고아로 남겨 계속 돌렸다 — 의도된 것, 단일 인스턴스 확인함.)

---

## 8. 전처리 전제 재검토 — 미적용 · 결정 대기

Drone-우분투 세션이 발견하고 이 세션이 일부 확인한 내용.
**09-12: 1)(화각 지터 확대)만 적용해 데이터를 재생성했다 (4-1절 M1). 2) 촬영 각도 필터는 미적용** —
결정되면 다시 재생성한다.

### 1) 화각 — 목표 픽셀 94 의 근거가 틀렸다

- `nomad_prep.py` · `wisard_prep.py` 의 `TARGET_PERSON_PX = 94` 주석은 "고도 20m, **FOV 60°**, 1280px" (코드에서 확인).
- `드론_사양_및_예산_기준서.md` 기준 예산안 v2.0 의 60° 는 **대각** 화각이고, 수평은 **54.0°**
  (IMX415 5.60×3.18 mm + 5.5 mm 계산값). 54° 면 20 m 에서 사람 1.7 m = **106.8 px**.
- 화각 자체도 미확정: 요구명세서 NFR-H02 "화각·보정값은 실측", 기준서 §8(실측 후 크게 벗어나면 4.8~5.2 mm 로 교체), 렌즈 "(잠정)".
- 1280 px 입력 · 누운 사람 1.7 m · 고도 16~30 m 에서: 5.5 mm/54° → **71~133 px**, 4.8 mm/60.5° → **62~117 px**.
- **사용자와 합의한 방향**: 목표 94 유지, `SCALE_JITTER` (0.75, 1.35) → **(0.65, 1.45)** = 61~136 px 로
  두 렌즈 경우를 모두 덮는다. 주석 근거도 고친다.

### 2) 촬영 각도가 섞여 있다 — 수직 하향 전제가 상당 부분 안 맞는다

- **NOMAD: 비스듬(oblique).** 표본 Actor001 a10/a30, Actor090 a10 — 얼굴 · 트럭 옆면 · 나무 줄기 · 원근이 보인다.
  (Actor001 a30 f0001 은 이 세션에서도 눈으로 확인)
- **WiSARD: 비행마다 다르다** (짐벌 기체라 조종자가 각도를 바꿈).

  | 각도 | 비행 |
  |---|---|
  | 수직 | `200402_Carnation_Inspire_VIS`, `200614_SuddenValley_Phantom_VIS_0005` |
  | 비스듬 | `210924_FHL_Enterprise_VIS_0403` |
  | 거의 수평 | `210327_Airfield_FLIR_VIS_1`(f75), `220109_Baker_Enterprise_VIS_1`(f0, 이륙 중일 수 있음) |

- **영향**:
  - `degrees=180` / `flipud=0.5` 증강, 누운/선 사람 픽셀 계산이 모두 수직 전제다.
  - `CLAUDE.md` §6 "마스크 기하가 반대 방향(선 사람이 더 길쭉)"도 비스듬 시점으로 설명된다.
- **검토 중 (사용자 결정 전)**:
  - 원본 프레임을 CLIP 제로샷으로 각도 분류해 걸러낸다 (1회성, 수십 분).
  - 기준은 수직 ±15° (`ANGLE_MAX` 8° 감안).
  - 정확도는 수작업 표본으로 검증해야 한다. 패키지 설치가 필요하므로 사용자에게 먼저 설명한다.

### 3) 마운트 각도 — 문서 간 불일치 (팀 확인 필요)

- 요구명세서 v1.1 NFR-H02 만 "무짐벌 하향 90° 기준 … **45°/60° 마운트는 검토 옵션**"이라고 적었다.
- 예산안 v2.0 · 기준서는 고정 90° 다.
- 45°/60° 를 채택하면 비스듬 데이터가 오히려 운용 조건과 맞는다 → 2)의 필터 결정과 묶여 있다.

### 4) 해상도 — 명세 확인 결과

- NFR-P06: 1080p 수신 ≥10 FPS · 누락 ≤5 %, **분석 입력 1280×720**.
- NFR-V01: 1280×720 ≤30 ms. NFR-V02: 분석 5 FPS.
- 축소는 통신 손실이 아니라 연산 예산 때문이다. **imgsz 1280 이 명세 기준이고, E1(960)은 대조군.**

### 5) 알려진 문서 오류 (미수정)

- NOMAD 인원 "42명" → 실제 **60명**:
  - `configs/data_all.yaml` 주석 (`make_configs.py` SETS note)
  - `CLAUDE.md` §2 (main 브랜치)
- 노트북 세션(Drone-local)에도 42명으로 전달됐었다 → 09-11 정정 메시지 전송.
