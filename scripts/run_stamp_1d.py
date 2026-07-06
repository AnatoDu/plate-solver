#!/usr/bin/env python3
r"""run_stamp_1d.py — 1D контактная задача со ШТАМПОМ (задел к 2D).

Полоса-пластина [0,L] на жёстком штампе фиксированной ширины [m,L] с зазором Δ;
односторонний контакт, метод обобщённой реакции (функция Грина балки). Физика и
числа — точно как в ``lab/Задачи Python/fix_base2.py`` (E=2.1·10⁶, L=100, m=45,
Δ=1, q0=4, n=100, β=0.01). Эталон — аналитика Maple (data/stamp_maple_xy.txt).

Отдаёт чистый PNG (прогиб w(x): МОР vs Maple | реакция r(x)) и числовую сводку.
Расчётная часть переиспользует ``plate_solver.mor1d`` через ``plate_solver.stamp``.
"""

from __future__ import annotations

import os

from plate_solver.mor1d import ContactStrip1D
from plate_solver.stamp import load_maple_reference, maple_agreement, solve_stamp

FIG = os.path.join("figures", "stamp_1d.png")


def run_stamp(problem: ContactStrip1D | None = None):
    """-> (StampResult, (xa, ya) эталон Maple, (max|Δw|, отн.L²%))."""
    problem = problem or ContactStrip1D()
    res = solve_stamp(problem)
    xa, ya = load_maple_reference()
    agree = maple_agreement(res.x, res.w, xa, ya)
    return res, (xa, ya), agree


def make_figure(res, maple, save: str = FIG):
    """Двухпанельный рисунок: слева w(x) (МОР vs Maple), справа реакция r(x)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xa, ya = maple
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(res.x, res.w, "r-", lw=2.0, label="МОР (численно)")
    ax1.plot(xa, ya, "b--", lw=1.5, label="аналитика (Maple)")
    ax1.axhline(ContactStrip1D().gap, color="0.5", ls=":", lw=1.0, label="зазор Δ")
    ax1.set_title("Прогиб пластины $w(x)$")
    ax1.set_xlabel("$x$, см")
    ax1.set_ylabel("$w$, см")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(res.x, res.r, "g-", lw=2.0)
    xs, xe = res.contact_span
    ax2.axvspan(xs, xe, color="green", alpha=0.08, label=f"зона контакта [{xs:.0f}, {xe:.0f}]")
    ax2.plot(res.x_rmax, res.r_max, "k*", ms=12, label=f"пик $r$={res.r_max:.1f}")
    ax2.set_title("Контактная реакция $r(x)$")
    ax2.set_xlabel("$x$, см")
    ax2.set_ylabel("$r$")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save), exist_ok=True)
    fig.savefig(save, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return save


def main() -> None:
    res, maple, (max_abs, rel_l2) = run_stamp()
    xa, ya = maple
    xs, xe = res.contact_span
    path = make_figure(res, maple)

    print("=== ЧАСТЬ 1: ШТАМП (1D) ===")
    print("единицы: размерные (x в см, w в см); m/L = 45/100 = 0.45")
    print(f"зона контакта:           x ∈ [{xs:.0f}, {xe:.0f}]  ({res.n_contact} узлов)")
    print(f"пик реакции r_max:       {res.r_max:.4f} в точке x={res.x_rmax:.0f} (кромка штампа)")
    print(f"w_max:                   {res.w_max:.5f} в точке x={res.x_wmax:.0f}")
    print(f"согласие с Maple:        max|Δw|={max_abs:.5f}, отн. L²={rel_l2:.3f} %")
    print(f"итераций до сходимости:  {res.iters}  (критерий max|Δw|<1e-9, "
          f"сошлось={res.converged})")
    print(f"невязка ‖Δr‖:            {res.dr_first:.3e} → {res.dr_last:.3e}")
    print(f"файл: {path}")


if __name__ == "__main__":
    main()
