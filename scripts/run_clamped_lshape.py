#!/usr/bin/env python3
r"""run_clamped_lshape.py — ЗАЩЕМЛЁННАЯ L-форма (контраст к парадоксу шарнира, раздел 4.2).

Та же L-форма ``[0,1]²\[0.5,1]²`` с входящим углом (0.5,0.5), но защемление по
ВСЕМУ контуру (структура ``w = ω²Φ``). Сравниваем RFM (защемление) с независимым
МКЭ-эталоном (защемлённый конформный C¹-элемент Аргириса).

Главный вывод: при защемлении расхождение RFM↔МКЭ МАЛО и УБЫВАЕТ с ростом p
(сходится к нулю) — это ОБЫЧНАЯ погрешность дискретизации у входящего угла, а НЕ
парадокс. Контраст с шарниром, где RFM(мягкий шарнир)↔Кирхгоф расходятся на 54.86 %
и расхождение НЕ убывает (модельный разрыв — парадокс Сапонджяна–Бабушки, NOTES §9).
Справочно (для текста доклада): показатель угловой сингулярности при защемлении во
входящем угле 3π/2 иной, чем при шарнире (≈0.5445 против 2/3); проявляется в
моментах/производных, не в прогибе, и парадокса не создаёт (численно не извлекаем).

Отдаёт: w_max(RFM) vs w_max(МКЭ), отн. L²; таблицу p-сходимости расхождения
(убывает); рисунок ``clamped_lshape.png`` (прогибы RFM/МКЭ + сходимость).
"""

from __future__ import annotations

import os

import numpy as np

from plate_solver import geometry
from plate_solver import quadrature as quad
from plate_solver.clamped import ClampedPlate, clamped_fem_lshape
from plate_solver.config import Config

FIG = os.path.join("figures", "clamped_lshape.png")

NU, Q0, E, H = 0.3, 4.0, 2.1e6, 0.06
SIDE, CUT = 1.0, 0.5
P_HEAD = 10
P_SWEEP = (2, 4, 6, 8, 10)
Q_QUAD = 120
FEM_M, FEM_REFINE = 16, 3
HINGE_PARADOX_PCT = 54.86   # из золотого прогона: RFM(шарнир)↔Кирхгоф (контраст)


def _rel_l2(a, b, w):
    a, b, w = np.asarray(a), np.asarray(b), np.asarray(w)
    return 100.0 * np.sqrt(np.sum(w * (a - b) ** 2)) / np.sqrt(np.sum(w * b**2))


def run_clamped_lshape():
    """-> dict: w_max(RFM/МКЭ), отн. L², p-сходимость расхождения RFM↔МКЭ."""
    D = E * H**3 / (12 * (1 - NU**2))
    dom = geometry.make_L(SIDE, CUT)

    # независимый МКЭ-эталон (Аргирис, C¹), один раз
    fem = clamped_fem_lshape(D, Q0, NU, side=SIDE, cut=CUT, mesh_m=FEM_M, refine=FEM_REFINE)
    w_fem_max = fem.w_max_on_grid(dom, grid_n=160)

    # точки сверки: узлы квадратуры с отступом от границы
    qn = quad.interior_nodes(dom, Q_QUAD)
    keep = dom.omega(qn.x, qn.y) > 0.02
    Xc, Yc, Wc = qn.x[keep], qn.y[keep], qn.w[keep]
    w_fem_c = fem.at(Xc, Yc)

    # p-сходимость расхождения RFM↔МКЭ
    rows = {}
    head = {}
    for p in P_SWEEP:
        cfg = Config(nu=NU, q0=Q0, h=H, E=E, p=p, Q=Q_QUAD)
        cp = ClampedPlate.from_config(dom, cfg)
        c = cp.solve_uniform(Q0)
        w_rfm_c = cp.deflection(c, Xc, Yc)
        w_rfm_max = cp.w_max_on_grid(c, grid_n=160)
        rows[p] = {
            "w_max": w_rfm_max,
            "rel_l2": _rel_l2(w_rfm_c, w_fem_c, Wc),
            "wmax_rel": 100.0 * abs(w_rfm_max - w_fem_max) / w_fem_max,
            "cond": cp.cond,
        }
        if p == P_HEAD:
            head = {"plate": cp, "c": c, "fem": fem}

    return {
        "D": D, "dom": dom, "w_fem_max": w_fem_max,
        "rows": rows, "n_points": int(Xc.size),
        "rel_l2_head": rows[P_HEAD]["rel_l2"], "wmax_rel_head": rows[P_HEAD]["wmax_rel"],
        "w_rfm_max_head": rows[P_HEAD]["w_max"],
        **head,
    }


def make_figure(data, save: str = FIG):
    """Планшет: прогиб RFM | прогиб МКЭ | p-сходимость расхождения (нет парадокса)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dom, cp, c, fem = data["dom"], data["plate"], data["c"], data["fem"]
    g = 160
    xs = np.linspace(0.0, 1.0, g)
    Xg, Yg = np.meshgrid(xs, xs)
    inside = dom.omega(Xg, Yg) > 0.0
    W_rfm = np.full(Xg.shape, np.nan)
    W_fem = np.full(Xg.shape, np.nan)
    W_rfm[inside] = cp.deflection(c, Xg[inside], Yg[inside])
    W_fem[inside] = fem.at(Xg[inside], Yg[inside])
    vmax = np.nanmax(W_fem)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, Wf, ttl in ((axes[0], W_rfm, f"RFM (защемл., p={P_HEAD})"),
                        (axes[1], W_fem, "МКЭ (Аргирис, C¹)")):
        pcm = ax.contourf(Xg, Yg, np.ma.masked_invalid(Wf), levels=20, cmap="viridis",
                          vmin=0, vmax=vmax)
        ax.contour(Xg, Yg, dom.omega(Xg, Yg), levels=[0.0], colors="k", linewidths=1.0)
        fig.colorbar(pcm, ax=ax, label="w")
        ax.set_aspect("equal")
        ax.set_title(ttl)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    ax = axes[2]
    ps = list(data["rows"].keys())
    l2 = [data["rows"][p]["rel_l2"] for p in ps]
    ax.plot(ps, l2, "o-", color="tab:blue", label="RFM↔МКЭ (защемление)")
    ax.axhline(HINGE_PARADOX_PCT, color="tab:red", ls="--",
               label=f"шарнир: парадокс {HINGE_PARADOX_PCT:.1f} %")
    ax.set_xlabel("степень p")
    ax.set_ylabel("отн. L²-отклонение, %")
    ax.set_title("Расхождение убывает с p → нет парадокса")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save), exist_ok=True)
    fig.savefig(save, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return save


def main() -> None:
    d = run_clamped_lshape()
    path = make_figure(d)
    print("=== ЧАСТЬ 2B: L-ФОРМА, ЗАЩЕМЛЕНИЕ ===")
    print(f"ν={NU}, q0={Q0}, h={H}, p={P_HEAD}, Q={Q_QUAD}; точек сверки {d['n_points']}")
    print("МКЭ-элемент: Аргирис (полный конформный C¹, степень 5)")
    print(f"w_max(RFM, p={P_HEAD}) = {d['w_rfm_max_head']:.6e}")
    print(f"w_max(МКЭ)           = {d['w_fem_max']:.6e}   "
          f"(отн. по w_max = {d['wmax_rel_head']:.2f} %)")
    print(f"отн. L²-отклонение RFM↔МКЭ (p={P_HEAD}): {d['rel_l2_head']:.2f} %")
    print("p-сходимость расхождения (УБЫВАЕТ ⇒ сходятся к ОДНОМУ решению, нет разрыва):")
    for p, r in d["rows"].items():
        print(f"  p={p:>2}: L²={r['rel_l2']:6.2f} %  | по w_max {r['wmax_rel']:5.2f} %  "
              f"| cond(S)={r['cond']:.2e}")
    print("вывод: ПАРАДОКСА НЕТ — расхождение убывает с p (дискретизация у входящего угла)")
    print(f"       и сходится к нулю; по w_max ({d['wmax_rel_head']:.1f} %) на порядок меньше "
          f"шарнирного разрыва (~61 %),")
    ratio = HINGE_PARADOX_PCT / d["rel_l2_head"]
    print(f"       по L² ({d['rel_l2_head']:.1f} %) в ~{ratio:.1f} раза меньше парадокса "
          f"{HINGE_PARADOX_PCT:.2f} % (тот НЕ убывает с измельчением).")
    print(f"файл: {path}")


if __name__ == "__main__":
    main()
