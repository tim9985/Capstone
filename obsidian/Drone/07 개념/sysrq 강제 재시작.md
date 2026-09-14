---
tags: [개념, 서버]
---

> [!abstract] 한 줄
> **sysrq**(Magic SysRq)는 리눅스 커널에 **바로** 내리는 비상 명령이다. 우리 서버는 관리자 공지에 따라 `sudo reboot` 대신 이것으로 재시작한다.

## 명령

```bash
echo 1 | sudo tee /proc/sys/kernel/sysrq    # sysrq 기능 켜기
echo s | sudo tee /proc/sysrq-trigger       # (권장) 디스크 쓰기 마무리
echo b | sudo tee /proc/sysrq-trigger       # 즉시 재시작
```

| 글자 | 하는 일 |
|---|---|
| `s` | 메모리에 남은 쓰기를 디스크에 내린다 (sync) |
| `b` | 정리 없이 **즉시** 재시작 — 전원 버튼을 누른 것과 비슷하다 |

## 왜 `sudo reboot` 을 안 쓰나

- 이 서버는 `reboot` · `shutdown` 으로 끄면 **다시 켜지지 않아** 사람이 가서 켜야 한다 (관리자 공지). 09-12 에 이렇게 **이틀간** 꺼져 있었다
- GPU 가 사라지면 드라이버를 내리는 과정이 멈춰 버려 일반 종료 절차가 끝나지 않기도 한다 (09-11 확인). `b` 는 그 절차를 건너뛴다

## 위험과 대책

| 위험 | 대책 |
|---|---|
| 쓰던 파일을 마무리하지 않는다 → 체크포인트 `last.pt` 가 깨질 수 있다 | 자동 복구 서비스는 `sync` (최대 30초) 뒤에 `b`. 재개 전 `last.pt` 를 읽어 보고 깨졌으면 `last_backup.pt` 로 → [[결정 - GPU 완화 설정과 자동 복구]] |
| 마지막 커널 로그가 파일에 안 남을 수 있다 | 원인 코드는 재시작 전에 `sudo dmesg` → [[GPU 장애 코드 Xid]] |

## 연결

[[학과 서버]] · [[GPU 장애 Xid 79]]
