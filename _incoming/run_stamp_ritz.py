#!/usr/bin/env python3
r"""run_stamp_ritz.py — ЧАСТЬ 1: штамп методом Ритца (вариант B), тройная сходимость.

Та же 1D-контактная задача о штампе, но изгиб решается РИТЦЕМ (структура под пару
ГУ «шарнир(0)+скользящая заделка(L)») с тем же МОР. Показываем: Грин = Ритц =
аналитика Maple — штамп решается и точным оператором (функция Грина), и Ритцем.

Грин-результат переиспользуется из ``plates.stamp`` (как есть). Отдаёт PNG
``stamp_ritz.png`` (w: Ритц/Грин/Maple; r: Ритц/Грин) и числовую сводку.
"""

from __future__ import annotations

import os

import numpy as np
from plate_solver.contact.mor1d import ContactStrip1D
from plates.stamp import load_maple_reference, maple_agreement, solve_stamp
from plates.stamp_ritz import solve_stamp_ritz

FIG = os.path.join("figures", "stamp_ritz.png")
P_RITZ = 16


def _rel_l2(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return 100.0 * np.sqrt(np.sum((a - b) ** 2)) / np.sqrt(np.sum(b**2))


def run_stamp_ritz(p: int = P_RITZ):
    """-> dict: результаты Грин/Ритц + попарные L²-отклонения и согласие с Maple."""
    problem = ContactStrip1D()
    green = solve_stamp(problem)                      # как есть (функция Грина)
    ritz, condS = solve_stamp_ritz(problem, p=p)
    xa, ya = load_maple_reference()

    _, l2_green_maple = maple_agreement(green.x, green.w, xa, ya)
    _, l2_ritz_maple = maple_agreement(ritz.x, ritz.w, xa, ya)
    return {
        "green": green, "ritz": ritz, "maple": (xa, ya), "cond_S": condS, "p": p,
        "l2_ritz_green": _rel_l2(ritz.w, green.w),
        "l2_ritz_maple": l2_ritz_maple,
        "l2_green_maple": l2_green_maple,
    }


def make_figure(d, save: str = FIG):
    """w(x): Ритц/Грин/Maple | r(x): Ритц vs Грин."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g, rz = d["green"], d["ritz"]
    xa, ya = d["maple"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(g.x, g.w, "r-", lw=2.2, label="функция Грина")
    ax1.plot(rz.x, rz.w, color="tab:green", ls="--", lw=2.0, label=f"Ритц (p={d['p']})")
    ax1.plot(xa, ya, "b:", lw=1.8, label="аналитика (Maple)")
    ax1.axhline(ContactStrip1D().gap, color="0.6", ls=":", lw=1.0)
    ax1.set_title("Прогиб w(x): Грин = Ритц = Maple")
    ax1.set_xlabel("x, см")
    ax1.set_ylabel("w, см")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(g.x, g.r, "r-", lw=2.2, label="реакция (Грин)")
    ax2.plot(rz.x, rz.r, color="tab:green", ls="--", lw=2.0, label="реакция (Ритц)")
    xs, xe = g.contact_span
    ax2.axvspan(xs, xe, color="green", alpha=0.07)
    ax2.set_title("Контактная реакция r(x): Ритц vs Грин")
    ax2.set_xlabel("x, см")
    ax2.set_ylabel("r")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save), exist_ok=True)
    fig.savefig(save, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return save


def main() -> None:
    d = run_stamp_ritz()
    path = make_figure(d)
    g, rz = d["green"], d["ritz"]
    print("=== ЧАСТЬ 1: ШТАМП МЕТОДОМ РИТЦА (вариант B) ===")
    print(f"структура Ритца: w=Σ B·T (ГУ: шарнир x=0 + скольз.заделка x=L), p={d['p']}, "
          f"cond(S)={d['cond_S']:.1e}")
    print(f"L²-отклонения:  Ритц↔Грин={d['l2_ritz_green']:.2f} %  ;  "
          f"Ритц↔Maple={d['l2_ritz_maple']:.2f} %  ;  Грин↔Maple={d['l2_green_maple']:.2f} %")
    print(f"зона контакта (Ритц): [{rz.contact_span[0]:.0f}, {rz.contact_span[1]:.0f}]  "
          f"(Грин: [{g.contact_span[0]:.0f}, {g.contact_span[1]:.0f}])")
    print(f"пик реакции: Ритц r={rz.r_max:.1f} (x={rz.x_rmax:.0f})  ;  Грин r={g.r_max:.1f}")
    print(f"w_max: Ритц={rz.w_max:.4f}  ;  Грин={g.w_max:.4f}")
    print(f"итераций: Ритц={rz.iters}  ;  Грин={g.iters}")
    print("вывод: Грин = Ритц = Maple → штамп решается и точным оператором, и Ритцем")
    print(f"файл: {path}")


if __name__ == "__main__":
    main()
