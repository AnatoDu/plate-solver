#!/usr/bin/env python3
r"""run_ladder_1d.py — СТУПЕНЬ 1 лестницы: одномерный (цилиндрический) изгиб полосы.

Полоса единичной ширины, ``D w⁗ = q``. RFM + Ритц в 1D (структура: шарнир ``ω·Φ``,
защемление ``ω²·Φ``; ``ω = x(a−x)/a``). Точные замкнутые решения программируются
напрямую. Решение полиномиально ⇒ ложится в базис ПОЧТИ ТОЧНО (ожидается машинная
точность).
"""

from __future__ import annotations

import numpy as np
from plates import ladder as L

A, D, Q = 1.0, 1.0, 1.0
P_SWEEP = (2, 4, 6, 8, 10)


def run_ladder_1d():
    """-> dict со сводкой по обоим закреплениям (p-sweep, отн. погрешность w_max)."""
    out = {}
    for support, wmax_fn, exact_fn in (
        ("hinge", L.strip_hinge_wmax, L.strip_hinge_exact),
        ("clamped", L.strip_clamped_wmax, L.strip_clamped_exact),
    ):
        w_ex = wmax_fn(A, D, Q)
        rows = {}
        xs = np.linspace(0.0, A, 401)
        we = exact_fn(xs, A, D, Q)
        for p in P_SWEEP:
            res = L.solve_strip_1d(A, D, Q, support, p)
            wc = res.w_center()
            wn = res.deflection(xs)
            l2 = float(np.sqrt(np.trapezoid((wn - we) ** 2, xs)) / np.sqrt(np.trapezoid(we**2, xs)))
            rows[p] = {"w_max": wc, "err": abs(wc - w_ex) / w_ex, "l2": l2, "cond": res.cond}
        out[support] = {"w_exact": w_ex, "rows": rows}
    return out


def main() -> None:
    d = run_ladder_1d()
    print("=== СТУПЕНЬ 1: 1D ПОЛОСА (RFM+Ритц, без МОР) ===")
    labels = {"hinge": "шарнир  (5qa⁴/384D)", "clamped": "защемл. (qa⁴/384D) "}
    for support in ("hinge", "clamped"):
        s = d[support]
        print(f"-- {labels[support]}  эталон w_max = {s['w_exact']:.10e}")
        for p, r in s["rows"].items():
            print(f"   p={p:2d}: w_max={r['w_max']:.10e}  отн.err={r['err']:.2e}  "
                  f"L²={r['l2']:.2e}  cond={r['cond']:.1e}")


if __name__ == "__main__":
    main()
