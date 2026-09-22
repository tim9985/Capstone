"""
coord_error.py — 마운트각별 좌표 산정 오차 (2026-09-22)

방식 (사용자 제안과 동일)
  1. 박스 **아래 변** = 발끝 = 지면 접촉점으로 본다 (몸 중심은 공중이라 더 멀게 나온다)
  2. 그 화소를 지나는 광선의 **하향각** φ 를 구한다
        f = (W/2) / tan(HFOV/2)
        φ = θ + atan( (v - H/2) / f )          θ = 마운트각 (90° = 수직 하향)
  3. 평지 가정으로 교차
        수평거리 d = h / tan φ          사선거리 R = h / sin φ
  4. 기체 GPS + 기수방위로 회전해 실좌표

오차 전파 (편미분)
  ∂d/∂φ = −h / sin²φ        마운트각·기체자세 오차
  ∂d/∂h = 1 / tan φ         고도 오차 (GPS 고도 · 기압계)
  지형 기복 Δz → Δd = Δz / tan φ
"""
import math

D = math.radians
W, H = 1920, 1080


def f_px(fov):
    return (W / 2) / math.tan(D(fov) / 2)


def solve(fov, h, theta, v_off=0.0):
    """v_off: 화면 중심에서 아래로 몇 px (양수 = 더 아래 = 더 가깝게 보임)"""
    f = f_px(fov)
    phi = D(theta) + math.atan(v_off / f)
    phi = min(phi, D(89.999))
    return h / math.tan(phi), h / math.sin(phi), math.degrees(phi)


print(f"화각 75° · f={f_px(75):.0f}px · 화면 중앙 대상 · 평지 가정\n")
print(f"{'마운트각':>8} {'고도':>5} {'수평거리':>8} {'사선거리':>8} │ "
      f"{'각도 1°':>8} {'고도 1m':>8} {'기복 5m':>8} {'발끝 10px':>10}")
print("─" * 86)
for th in (90, 75, 60, 45, 30):
    for h in (25,):
        d, R, phi = solve(75, h, th)
        dd_dphi = abs(-h / math.sin(D(phi))**2) * D(1)      # 1° 당 m
        dd_dh = 1 / math.tan(D(phi))                         # 1 m 당 m
        relief = 5.0 / math.tan(D(phi))                      # 기복 5 m
        d2, _, _ = solve(75, h, th, v_off=10)
        foot = abs(d2 - d)
        print(f"{th:>7}° {h:>4}m {d:>7.1f}m {R:>7.1f}m │ "
              f"{dd_dphi:>7.2f}m {dd_dh:>7.2f}m {relief:>7.1f}m {foot:>9.2f}m")

print("\n※ NFR-V04 예산: 수평 오차 10~15 m")
print("\n=== 실제 오차 합성 (제곱합 제곱근) — 고도 25 m · 화각 75°")
print(f"{'마운트각':>8} {'GPS수평':>8} {'짐벌각 2°':>9} {'고도 3m':>8} {'기복 5m':>8} {'발끝':>6} │ {'합성':>7} 판정")
print("─" * 82)
for th in (90, 75, 60, 45, 30):
    d, R, phi = solve(75, 25, th)
    e_gps = 3.0
    e_ang = abs(-25 / math.sin(D(phi))**2) * D(2)
    e_alt = 3.0 / math.tan(D(phi))
    e_rel = 5.0 / math.tan(D(phi))
    d2, _, _ = solve(75, 25, th, v_off=10); e_foot = abs(d2 - d)
    tot = math.sqrt(e_gps**2 + e_ang**2 + e_alt**2 + e_rel**2 + e_foot**2)
    mark = "✅ 통과" if tot <= 15 else ("⚠ 경계" if tot <= 20 else "❌ 초과")
    print(f"{th:>7}° {e_gps:>7.1f}m {e_ang:>8.1f}m {e_alt:>7.1f}m {e_rel:>7.1f}m {e_foot:>5.1f}m │ {tot:>6.1f}m {mark}")
