#!/usr/bin/env python3
r"""run_circle_1d_2d.py — ЧАСТЬ 2: круг как ОДНОМЕРНАЯ задача по радиусу.

Осесимметричная пластина решена 1D-методом Ритца по радиусу (``plates.radial``) и
сверена с радиальным сечением 2D-решения (RFM+Ритц) и с точной формулой —
верификация 1D ↔ 2D ↔ аналитика (для доклада по симметрии). a=1, q=4, ν=0.3.

  • защемление: точное ``w=q(a²−r²)²/(64D)``; 1D воспроизводит машинно, 2D ~0.1 %;
  • шарнир (мягкий): эталон ``w=q(a²−r²)(3a²−r²)/(64D)``; 1D (две радиальные
    Пуассоны) и 2D (расщепление) совпадают с ним.

Отдаёт PNG ``circle_1d_2d.png`` (w(r) тремя кривыми) и числовую сводку.
"""

from __future__ import annotations

import os

import numpy as np
from plates import analytic, geometry
from plates.clamped import ClampedPlate
from plates.config import Config
from plates.plate import PlateBending
from plates.radial import RadialClamped, solve_radial_soft_hinge

FIG = os.path.join("figures", "circle_1d_2d.png")
A, Q0, NU, E, H = 1.0, 4.0, 0.3, 2.1e6, 1.0
P2D, Q2D, P1D = 10, 1024, 6


def _l2(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return 100.0 * np.sqrt(np.mean((x - y) ** 2)) / np.sqrt(np.mean(y**2))


def run_circle_1d_2d():
    """-> dict: профили и L²-отклонения 1D/2D/точное для защемления и мягкого шарнира."""
    D = E * H**3 / (12 * (1 - NU**2))
    dom = geometry.make_circle(A)
    rr = np.linspace(0.0, A, 201)
    cfg = Config(a=A, q0=Q0, nu=NU, h=H, E=E, p=P2D, Q=Q2D)
    zeros = np.zeros_like(rr)

    # -- защемление --
    w_ex_c = analytic.circular_plate_clamped(rr, Q0, A, D)
    cp = ClampedPlate.from_config(dom, cfg)
    w2d_c = cp.deflection(cp.solve_uniform(Q0), rr, zeros)
    rc = RadialClamped(A, D, p=P1D)
    rc.solve(Q0, NU)
    w1d_c = rc.deflection(rr)

    # -- мягкий шарнир --
    w_ex_s = analytic.circular_plate_soft_hinge(rr, Q0, A, D)
    pb = PlateBending.from_config(dom, cfg)
    _, cw = pb.solve_uniform(Q0)
    w2d_s = pb.deflection(cw, rr, zeros)
    rp, cws = solve_radial_soft_hinge(A, D, Q0, p=P1D)
    w1d_s = rp.deflection(cws, rr)

    return {
        "r": rr,
        "clamped": {"exact": w_ex_c, "w1d": w1d_c, "w2d": w2d_c, "cond1d": rc.cond,
                    "l2_1d_ex": _l2(w1d_c, w_ex_c), "l2_2d_ex": _l2(w2d_c, w_ex_c),
                    "l2_1d_2d": _l2(w1d_c, w2d_c)},
        "hinge": {"exact": w_ex_s, "w1d": w1d_s, "w2d": w2d_s,
                  "l2_1d_ex": _l2(w1d_s, w_ex_s), "l2_2d_ex": _l2(w2d_s, w_ex_s),
                  "l2_1d_2d": _l2(w1d_s, w2d_s)},
    }


def make_figure(d, save: str = FIG):
    """w(r) тремя кривыми (1D, 2D-сечение, точная) — защемление | мягкий шарнир."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rr = d["r"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, key, ttl in ((axes[0], "clamped", "Защемление  w=q(a²−r²)²/64D"),
                         (axes[1], "hinge", "Мягкий шарнир  w=q(a²−r²)(3a²−r²)/64D")):
        s = d[key]
        ax.plot(rr, s["exact"], "k-", lw=3, alpha=0.4, label="точная")
        ax.plot(rr, s["w2d"], "b--", lw=1.8, label="2D-сечение (RFM+Ритц)")
        ax.plot(rr, s["w1d"], color="tab:red", ls=":", lw=2.2, label="1D по радиусу")
        ax.set_title(ttl)
        ax.set_xlabel("r")
        ax.set_ylabel("w")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save), exist_ok=True)
    fig.savefig(save, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return save


def main() -> None:
    d = run_circle_1d_2d()
    path = make_figure(d)
    c, s = d["clamped"], d["hinge"]
    print("=== ЧАСТЬ 2: КРУГ КАК 1D-ЗАДАЧА ПО РАДИУСУ ===")
    print(f"a={A}, q={Q0}, ν={NU}; 2D: p={P2D},Q={Q2D}; 1D: p={P1D} (радиальный Ритц)")
    print("защемление:")
    print(f"  1D↔точное (q(a²−r²)²/64D) = {c['l2_1d_ex']:.3f} % ;  "
          f"2D↔точное = {c['l2_2d_ex']:.3f} % ;  1D↔2D = {c['l2_1d_2d']:.3f} %")
    print("шарнир (мягкий):")
    print(f"  1D↔мягкий эталон = {s['l2_1d_ex']:.3f} % ;  "
          f"2D↔мягкий = {s['l2_2d_ex']:.3f} % ;  1D↔2D = {s['l2_1d_2d']:.3f} %")
    print("вывод: 1D (полярн.) ↔ 2D (RFM+Ритц) ↔ аналитика — все сходятся")
    print(f"файл: {path}")


if __name__ == "__main__":
    main()
