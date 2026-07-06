#!/usr/bin/env python3
r"""run_clamped_circle.py — ЗАЩЕМЛЁННЫЙ круг (контрольный случай).

Жёстко защемлённая круглая пластина под равномерной нагрузкой. Точное решение
Кирхгофа (формула 4.1):

    w(ρ) = q (a² − ρ²)² / (64 D),    w_max = q a⁴ / (64 D).

Структура защемления ``w = ω²Φ`` ТОЖДЕСТВЕННО даёт ``w=0`` и ``∂w/∂n=0``. Для круга
точное решение само лежит в этой структуре (``w = [q a²/(16D)]·ω²``), поэтому RFM
воспроизводит (4.1) с ВЫСОКОЙ точностью и БЕЗ модельной погрешности — в отличие от
«мягкого шарнира», где на криволинейной границе возникает модельный разрыв 26.4 %
(NOTES.md §8). Это прямой контраст: при защемлении расщепление корректно.

Отдаёт: таблицу сходимости p → w_max(RFM)/отн.погрешность/cond(S); контроль по Q
(погрешность падает к нулю — нет модельного «пола»); независимый МКЭ (Аргирис);
поверхность прогиба ``clamped_circle.png``.
"""

from __future__ import annotations

import os

import numpy as np

from plate_solver import analytic, geometry
from plate_solver.clamped import ClampedPlate, clamped_fem_circle
from plate_solver.config import Config

FIG = os.path.join("figures", "clamped_circle.png")

A, Q0, NU, E, H = 1.0, 4.0, 0.3, 2.1e6, 1.0
P_SWEEP = (2, 4, 6, 8, 10)
Q_QUAD = 1024


def run_clamped_circle():
    """-> dict со сводкой (p-sweep, Q-контроль, МКЭ) для защемлённого круга."""
    D = E * H**3 / (12 * (1 - NU**2))
    dom = geometry.make_circle(A)
    w_exact = float(analytic.clamped_uniform_wmax(A, Q0, D))            # (4.1)
    w_soft = float(analytic.circular_plate_soft_hinge_wmax(Q0, A, D))   # мягкий шарнир (контраст)
    w_ss = float(analytic.circular_plate_simply_supported(0.0, Q0, A, NU, D))

    # p-sweep при Q=1024
    rows = {}
    c10 = None
    for p in P_SWEEP:
        cfg = Config(a=A, q0=Q0, nu=NU, h=H, E=E, p=p, Q=Q_QUAD)
        cp = ClampedPlate.from_config(dom, cfg)
        c = cp.solve_uniform(Q0)
        w0 = float(cp.deflection(c, 0.0, 0.0))
        rows[p] = {"w_max": w0, "err": abs(w0 - w_exact) / w_exact, "cond": cp.cond}
        if p == 10:
            c10, cp10, cfg10 = c, cp, cfg

    # Q-контроль (нет модельного «пола»: погрешность падает с Q при фиксированном p)
    q_rows = {}
    for Q in (64, 128, 256, 512, 1024):
        cfg = Config(a=A, q0=Q0, nu=NU, h=H, E=E, p=6, Q=Q)
        cp = ClampedPlate.from_config(dom, cfg)
        w0 = float(cp.deflection(cp.solve_uniform(Q0), 0.0, 0.0))
        q_rows[Q] = abs(w0 - w_exact) / w_exact

    # независимый МКЭ (Аргирис, C¹); максимум защемлённого круга — в центре
    fem = clamped_fem_circle(A, D, Q0, NU, nref=4)
    w_fem = fem.at_point(0.0, 0.0)

    return {
        "D": D, "w_exact": w_exact, "w_soft": w_soft, "w_ss": w_ss,
        "soft_gap_pct": 100.0 * abs(w_soft - w_ss) / w_ss,
        "rows": rows, "q_rows": q_rows,
        "w_fem": w_fem, "fem_err": abs(w_fem - w_exact) / w_exact,
        "c10": c10, "plate10": cp10, "cfg10": cfg10,
    }


def make_figure(data, save: str = FIG):
    """Поверхность прогиба защемлённого круга (p=10)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cp, c = data["plate10"], data["c10"]
    x0, x1, y0, y1 = cp.domain.bbox
    g = 120
    Xg, Yg = np.meshgrid(np.linspace(x0, x1, g), np.linspace(y0, y1, g))
    inside = cp.domain.omega(Xg, Yg) > 0.0
    W = np.full(Xg.shape, np.nan)
    W[inside] = cp.deflection(c, Xg[inside], Yg[inside])

    fig = plt.figure(figsize=(7, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xg, Yg, W, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("w")
    ax.set_title("Защемлённый круг: прогиб w(x,y), p=10")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    fig.savefig(save, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return save


def main() -> None:
    d = run_clamped_circle()
    path = make_figure(d)
    print("=== ЧАСТЬ 2A: КРУГ, ЗАЩЕМЛЕНИЕ ===")
    print(f"a={A}, q0={Q0}, ν={NU}, D={d['D']:.6g}, Q={Q_QUAD}")
    print(f"эталон (4.1): w_max = q a⁴/(64D) = {d['w_exact']:.6e}")
    print("p-sweep  w_max(RFM) / отн.погрешность к (4.1) / cond(S):")
    for p, r in d["rows"].items():
        print(f"  p={p:>2}: {r['w_max']:.6e} / {100*r['err']:.4f} % / cond={r['cond']:.2e}")
    print("контроль по Q (p=6) — погрешность падает к нулю (нет модельного «пола»):")
    for Q, e in d["q_rows"].items():
        print(f"  Q={Q:>4}: отн. {100*e:.4f} %")
    print(f"МКЭ (Аргирис, C¹): w_max = {d['w_fem']:.6e}, отн. = {100*d['fem_err']:.4f} %")
    print(f"модельная погрешность: ОТСУТСТВУЕТ (контраст с мягким шарниром "
          f"{d['soft_gap_pct']:.2f} % к (4.2))")
    print(f"файл: {path}")


if __name__ == "__main__":
    main()
