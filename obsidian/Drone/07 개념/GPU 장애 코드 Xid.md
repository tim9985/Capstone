---
tags: [개념, 서버]
---

> [!abstract] 한 줄
> **Xid** 는 NVIDIA 드라이버가 GPU 에 문제가 생겼을 때 커널 로그에 남기는 **오류 번호**다. 번호를 보면 원인 계열(프로그램 · 메모리 · 하드웨어)을 좁힐 수 있다.

## 어디서 보나

```bash
sudo dmesg | grep -E "NVRM|Xid"                                        # 이번 부팅 이후
sudo grep -hE "Xid|fallen off" /var/log/kern.log.1 /var/log/kern.log   # 지난 부팅까지
```

- 강제 재시작(sysrq `b`)은 디스크에 쓸 틈이 없어 마지막 로그가 파일에 안 남을 수 있다 → 원인 코드는 재시작 **전에** `dmesg` 로 확보한다 → [[sysrq 강제 재시작]]

## 우리 서버에서 본 번호

| 로그 | 뜻 | 원인인가 결과인가 |
|---|---|---|
| **Xid 79** | GPU has fallen off the bus — GPU 가 PCIe 통로에서 **사라졌다** | **원인 쪽.** NVIDIA 분류로 하드웨어 계열 (전원 · 연결 · 카드) |
| Xid 154 | Node Reboot Required — "재부팅해야만 복구된다"는 드라이버 판정 | 결과 |
| GSP RPC 덤프 | GPU 안의 관리용 프로세서(GSP)가 처리 중이던 요청 기록 | 결과 — 사라지는 순간의 기록일 뿐 |

## 증상으로 구별하기

| 증상 | 뜻 |
|---|---|
| 프로그램만 `CUDA error` 로 죽고 `nvidia-smi` 는 정상 | 프로그램 · 메모리 부족 쪽 |
| `nvidia-smi` → "No devices were found" | **장치 자체가 사라짐** → 재부팅해야 한다 |

우리 장애는 4번 모두 아래쪽 + Xid 79 → 154 였다 → [[GPU 장애 Xid 79]]

## 자동 복구 서비스가 쓰는 법

- `nvidia-smi` 연속 2번 실패 **그리고** 로그에 Xid 79 가 있을 때만 재부팅한다. 한쪽만이면 오판일 수 있다
- 같은 서버의 네트워크 카드도 `XID 641` 처럼 비슷한 글자를 남긴다 → 판정 규칙에서 걸러 냈다

## 연결

[[PCIe 세대와 ASPM·AER]] · [[전력 제한과 클럭 상한]] · [[결정 - GPU 완화 설정과 자동 복구]]
