#!/usr/bin/env python3
r"""run_ladder_mms.py — СТУПЕНЬ 3 лестницы: метод изготовленных решений (MMS).

Задаём ``w`` аналитически, считаем ``q = D·Δ²w`` СИМВОЛЬНО, решаем защемлённой
структурой ``ω²·Φ`` и сверяем с ``w``. Два теста:

  • прямоугольник [−a,a]×[−b,b], ``ω = (a²−x²)(b²−y²)`` (полином!), ``w = ω² =
    (x²−a²)²(y²−b²)²``. Решение ЛЕЖИТ в структуре, квадратура на bbox ТОЧНА ⇒
    L²-погрешность машинная. Это изолирует корректность сборки билинейной формы и
    квадратур ОТ ошибок аппроксимации границы.
  • круг (R=1, q=1), ``w = (R²−r²)²/(64D)`` (проверка геометрии R-функции). Здесь
    граница криволинейна ⇒ погрешность задаёт СТУПЕНЧАТАЯ маска квадратуры
    (~1/Q, NOTES §4), а не аппроксимация поля.
"""

from __future__ import annotations

import numpy as np
from plates import ladder as L
from plates.clamped import ClampedPlate
from plates.config import Config
from plates.geometry import Domain
from plates.geometry import x as sx
from plates.geometry import y as sy

NU, E, H = 0.3, 2.1e6, 1.0


def _l2(cp, c, wf, mask_eps=0.0):
    X, Y, W = cp.quad.x, cp.quad.y, cp.quad.w
    if mask_eps > 0.0:
        keep = cp.domain.omega(X, Y) > mask_eps
        X, Y, W = X[keep], Y[keep], W[keep]
    wn = cp.deflection(c, X, Y)
    we = wf(X, Y)
    return float(np.sqrt(np.sum(W * (wn - we) ** 2)) / np.sqrt(np.sum(W * we**2)))


def run_ladder_mms():
    """-> dict: L²-погрешность MMS на прямоугольнике (машинная) и круге (квадратурная)."""
    D = E * H**3 / (12 * (1 - NU**2))

    # -- прямоугольник: omega = (a^2-x^2)(b^2-y^2),  w = omega^2 --
    a, b = 1.0, 1.0
    dom_r = Domain((a**2 - sx**2) * (b**2 - sy**2), (-a, a, -b, b))
    qf_r, wf_r = L.mms_load_and_exact(L.mms_clamped_rect_w(a, b), D)
    rect_rows = {}
    for p in (4, 6, 8, 10):
        cp = ClampedPlate.from_config(dom_r, Config(q0=1.0, nu=NU, h=H, E=E, p=p, Q=64))
        c = cp.solve(qf_r(cp.quad.x, cp.quad.y))
        rect_rows[p] = _l2(cp, c, wf_r)

    # -- круг: w = (R^2-r^2)^2/64D, q=1 (квадратурно-ограничено) --
    from plates import geometry
    R = 1.0
    dom_c = geometry.make_circle(R)
    qf_c, wf_c = L.mms_load_and_exact(L.mms_clamped_disk_w(R, D), D)
    disk_rows = {}
    for Q in (128, 256, 512, 1024):
        cp = ClampedPlate.from_config(dom_c, Config(q0=1.0, nu=NU, h=H, E=E, p=8, Q=Q))
        c = cp.solve(qf_c(cp.quad.x, cp.quad.y))
        disk_rows[Q] = _l2(cp, c, wf_c, mask_eps=0.02)

    return {"rect_rows": rect_rows, "disk_rows": disk_rows}


def main() -> None:
    d = run_ladder_mms()
    print("=== СТУПЕНЬ 3: MMS (изготовленные решения, защемление ω²Φ) ===")
    print("-- прямоугольник ω=(a²−x²)(b²−y²), w=ω²: решение в структуре, квадратура точна")
    for p, l2 in d["rect_rows"].items():
        print(f"   p={p:2d}: L²-погрешность = {l2:.3e}")
    print("-- круг w=(R²−r²)²/64D (q=1): квадратурно-ограничено (ступенчатая маска ~1/Q)")
    for Q, l2 in d["disk_rows"].items():
        print(f"   Q={Q:4d}: L²-погрешность = {l2:.3e}")


if __name__ == "__main__":
    main()
