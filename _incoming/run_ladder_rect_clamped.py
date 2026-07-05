#!/usr/bin/env python3
r"""run_ladder_rect_clamped.py — СТУПЕНЬ 5 лестницы: прямоугольная пластина, ЗАЩЕМЛЕНИЕ (CCCC).

Квадрат a=b=1, ν=0.3, равномерная нагрузка. Замкнутого элементарного решения нет —
эталон табличный (Тимошенко Табл.35 / Batista): w_max=0.00126532 qa⁴/D,
M_центр=0.0229051 qa², M_край=−0.0513338 qa². Структура защемления ``ω²·Φ``;
домен-аппроксимация некритична (парадокса нет при любой границе, выпуклая область).
"""

from __future__ import annotations

import numpy as np
from plates import geometry
from plates import ladder as L
from plates.clamped import ClampedPlate
from plates.config import Config

A, B, NU, E, H = 1.0, 1.0, 0.3, 2.1e6, 1.0
Q0 = 4.0
P_SWEEP = (6, 8, 10, 12)
W_COEF, MC_COEF, ME_COEF = 0.00126532, 0.0229051, -0.0513338   # Тимошенко Табл.35


def run_ladder_rect_clamped():
    """-> dict: p-sweep w_max и моментов (центр, середина защемлённого края)."""
    D = E * H**3 / (12 * (1 - NU**2))
    dom = geometry.make_rectangle(0.0, A, 0.0, B)
    w_table = W_COEF * Q0 * A**4 / D
    mc_table = MC_COEF * Q0 * A**2
    me_table = ME_COEF * Q0 * A**2

    rows = {}
    for p in P_SWEEP:
        cfg = Config(q0=Q0, nu=NU, h=H, E=E, p=p, Q=200)
        cp = ClampedPlate.from_config(dom, cfg)
        c = cp.solve_uniform(Q0)
        wc = float(cp.deflection(c, A / 2, B / 2))
        ctr = (np.array([A / 2]), np.array([B / 2]))
        Mc_x, _ = L.bending_moments(dom, cp.basis, c, 2, D, NU, *ctr)
        # середина защемлённого нижнего края y=0: нормаль — y ⇒ M_край = M_y
        _, Me_y = L.bending_moments(dom, cp.basis, c, 2, D, NU, np.array([A / 2]), np.array([1e-9]))
        rows[p] = {
            "w_max": wc, "err_w": abs(wc - w_table) / w_table,
            "M_c": float(Mc_x[0]), "err_Mc": abs(float(Mc_x[0]) - mc_table) / abs(mc_table),
            "M_e": float(Me_y[0]), "err_Me": abs(float(Me_y[0]) - me_table) / abs(me_table),
            "cond": cp.cond,
        }
    return {"D": D, "w_table": w_table, "mc_table": mc_table, "me_table": me_table, "rows": rows}


def main() -> None:
    d = run_ladder_rect_clamped()
    print("=== СТУПЕНЬ 5: ПРЯМОУГОЛЬНИК, ЗАЩЕМЛЕНИЕ (CCCC) ===")
    print(f"квадрат a=b=1, ν={NU}, q0={Q0}, D={d['D']:.6g}")
    print(f"табл. Тимошенко: w_max=0.00126532 qa⁴/D={d['w_table']:.8e}, "
          f"M_c=0.0229051 qa²={d['mc_table']:.6e}, M_край=−0.0513338 qa²={d['me_table']:.6e}")
    for p, r in d["rows"].items():
        print(f"   p={p:2d}: w_max={r['w_max']:.8e}(e{r['err_w']:.1e})  "
              f"M_c={r['M_c']:.6e}(e{r['err_Mc']:.1e})  "
              f"M_край={r['M_e']:.6e}(e{r['err_Me']:.1e})  cond={r['cond']:.1e}")


if __name__ == "__main__":
    main()
