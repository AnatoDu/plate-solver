#!/usr/bin/env python3
r"""run_ladder_rect_hinge.py — СТУПЕНЬ 4 лестницы: прямоугольная пластина, ШАРНИР (SSSS).

Квадрат a=b=1, ν=0.3. Мягкий шарнир ``ω·Φ`` (на ПРЯМЫХ краях совпадает с истинным
Кирхгофом — парадокса НЕТ, выпуклая полигональная область). Два типа нагрузки:

  (4a) синус ``q=q0 sin(πx/a)sin(πy/b)`` — ТОЧНОЕ одночленное решение;
  (4b) равномерная ``q0`` — эталон ряд Навье + табличные константы Тимошенко
       (квадрат, ν=0.3): w_max=0.00406235 q0 a⁴/D, M_max=0.0478864 q0 a².

Контроль: модельной погрешности нет ни при каком разрешении (прямые края).
"""

from __future__ import annotations

import numpy as np

from plate_solver import geometry
from plate_solver import ladder as L
from plate_solver.config import Config
from plate_solver.plate import PlateBending

A, B, NU, E, H = 1.0, 1.0, 0.3, 2.1e6, 1.0
Q0 = 4.0
P_SWEEP = (4, 6, 8, 10, 12)
W_COEF, M_COEF = 0.00406235, 0.0478864   # Тимошенко Табл.8 (квадрат, ν=0.3)


def run_ladder_rect_hinge():
    """-> dict: синус (к точному) и равномерная (к ряду Навье + табличным константам)."""
    D = E * H**3 / (12 * (1 - NU**2))
    dom = geometry.make_rectangle(0.0, A, 0.0, B)
    center = (np.array([A / 2]), np.array([B / 2]))

    # (4a) синус — точное решение
    w_sin_ex = L.rect_sin_wmax(A, B, D, Q0)
    sin_rows = {}
    for p in P_SWEEP:
        cfg = Config(q0=Q0, nu=NU, h=H, E=E, p=p, Q=160)
        pb = PlateBending.from_config(dom, cfg)
        _, cw = pb.solve(L.rect_sin_load(pb.quad.x, pb.quad.y, A, B, Q0))
        wc = float(pb.deflection(cw, A / 2, B / 2))
        sin_rows[p] = {"w_max": wc, "err": abs(wc - w_sin_ex) / w_sin_ex}

    # (4b) равномерная — ряд Навье + табличные константы
    w_navier, m_navier = L.navier_uniform_center(A, B, D, Q0, NU, n_terms=50)  # 99×99
    w_table = W_COEF * Q0 * A**4 / D
    m_table = M_COEF * Q0 * A**2
    unif_rows = {}
    for p in P_SWEEP:
        cfg = Config(q0=Q0, nu=NU, h=H, E=E, p=p, Q=160)
        pb = PlateBending.from_config(dom, cfg)
        _, cw = pb.solve_uniform(Q0)
        wc = float(pb.deflection(cw, A / 2, B / 2))
        Mx, _ = L.bending_moments(dom, pb.basis, cw, 1, D, NU, *center)
        unif_rows[p] = {
            "w_max": wc, "err_w": abs(wc - w_table) / w_table,
            "M": float(Mx[0]), "err_M": abs(float(Mx[0]) - m_table) / m_table,
        }

    return {
        "D": D, "w_sin_ex": w_sin_ex, "sin_rows": sin_rows,
        "w_navier": w_navier, "m_navier": m_navier, "w_table": w_table, "m_table": m_table,
        "unif_rows": unif_rows,
    }


def main() -> None:
    d = run_ladder_rect_hinge()
    print("=== СТУПЕНЬ 4: ПРЯМОУГОЛЬНИК, ШАРНИР (SSSS) ===")
    print(f"квадрат a=b=1, ν={NU}, q0={Q0}, D={d['D']:.6g}")
    print(f"-- (4a) синус: точное w_max = {d['w_sin_ex']:.8e}")
    for p, r in d["sin_rows"].items():
        print(f"   p={p:2d}: w_max={r['w_max']:.8e}  отн.err={r['err']:.2e}")
    print(f"-- (4b) равномерная: Навье(99×99) w={d['w_navier']:.8e}, M={d['m_navier']:.6e}")
    print(f"        табл. Тимошенко: w=0.00406235 q a⁴/D={d['w_table']:.8e}, "
          f"M=0.0478864 q a²={d['m_table']:.6e}")
    for p, r in d["unif_rows"].items():
        print(f"   p={p:2d}: w_max={r['w_max']:.8e} (err {r['err_w']:.2e})  "
              f"M={r['M']:.6e} (err {r['err_M']:.2e})")
    print("вывод: парадокса нет (прямые края) — модельной погрешности нет при любом p.")


if __name__ == "__main__":
    main()
