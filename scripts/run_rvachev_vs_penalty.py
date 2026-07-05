#!/usr/bin/env python3
r"""run_rvachev_vs_penalty.py — БЛОК Р: структура Рвачёва (ω^p·Φ) против ШТРАФА.

Один метод Ритца, один базис Чебышёва, круг (шарнир, равномерная нагрузка); ГУ
учтены двумя способами: СТРУКТУРОЙ ``w=ω·Φ`` (ГУ тождественно) и ШТРАФОМ
``w=Σcᵢφᵢ`` + ``γ∮w²ds`` (ГУ приближённо). Эталон — «мягкий» шарнир
``w=3qa⁴/64D``. Показываем: структура достигает целевой точности при существенно
меньшем числе функций N и с лучшей обусловленностью, чем штраф.

Интерпретация (для текста): это «Ритц со структурой Рвачёва против Ритца со
штрафным учётом ГУ» — оба используют один решатель, разница лишь в способе учёта
краевых условий; так изолируется вклад именно R-функций.
"""

from __future__ import annotations

import os

from plate_solver.penalty import penalty_vs_structure_circle

FIG = os.path.join("figures", "rvachev_vs_penalty.png")
A, Q0, NU, E, H = 1.0, 4.0, 0.3, 2.1e6, 1.0
P_LIST = (2, 4, 6, 8, 10, 12, 14)
Q_QUAD, N_BND = 1024, 600
GAMMA, GAMMA_SMALL = 1.0e4, 1.0e2     # рабочий γ и «слишком малый» (ГУ не выполнены)
TARGET_PCT = 1.0


def run_rvachev_vs_penalty():
    """-> (data при рабочем γ, data при малом γ)."""
    D = E * H**3 / (12 * (1 - NU**2))
    kw = dict(p_list=P_LIST, Q=Q_QUAD, n_bnd=N_BND, target_pct=TARGET_PCT)
    d = penalty_vs_structure_circle(A, Q0, D, NU, E, H, gamma=GAMMA, **kw)
    d_small = penalty_vs_structure_circle(A, Q0, D, NU, E, H, gamma=GAMMA_SMALL, **kw)
    return d, d_small


def make_figure(d, d_small, save: str = FIG):
    """err(N): структура vs штраф (рабочий γ) vs штраф (малый γ, «пол» точности)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = [r["N"] for r in d["rows"]]
    err_s = [100 * r["err_struct"] for r in d["rows"]]
    err_p = [100 * r["err_pen"] for r in d["rows"]]
    err_p2 = [100 * r["err_pen"] for r in d_small["rows"]]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.semilogy(N, err_s, "o-", color="tab:blue", lw=2, label="структура Рвачёва (ω·Φ)")
    ax.semilogy(N, err_p, "s-", color="tab:red", lw=2, label=f"штраф (γ={GAMMA:.0e})")
    ax.semilogy(N, err_p2, "s--", color="tab:orange", lw=1.5,
                label=f"штраф (γ={GAMMA_SMALL:.0e}, ГУ не выполнены)")
    ax.axhline(TARGET_PCT, color="0.4", ls=":", label=f"цель {TARGET_PCT:.0f} %")
    if d["N_struct"]:
        ax.axvline(d["N_struct"], color="tab:blue", ls=":", alpha=0.5)
    if d["N_pen"]:
        ax.axvline(d["N_pen"], color="tab:red", ls=":", alpha=0.5)
    ax.set_xlabel("число базисных функций N = (p+1)²")
    ax.set_ylabel("отн. погрешность w_max, %")
    ax.set_title("Вклад структуры Рвачёва: структура vs штрафной учёт ГУ (круг, шарнир)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save), exist_ok=True)
    fig.savefig(save, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return save


def main() -> None:
    d, d_small = run_rvachev_vs_penalty()
    path = make_figure(d, d_small)
    print("=== БЛОК Р: СТРУКТУРА РВАЧЁВА vs ШТРАФ (круг, шарнир) ===")
    print(f"a={A}, q={Q0}, ν={NU}, Q={Q_QUAD}, γ={GAMMA:.0e}; эталон w_soft={d['w_soft']:.6e}")
    print("   N    err_структ %   cond_структ    err_штраф %   cond_штраф")
    for r in d["rows"]:
        print(f"  {r['N']:4d}   {100*r['err_struct']:9.4f}    {r['cond_struct']:.1e}    "
              f"{100*r['err_pen']:9.4f}    {r['cond_pen']:.1e}")
    cs = d["cond_struct_target"] or 0.0
    cp = d["cond_pen_target"] or 0.0
    print(f"целевая точность {TARGET_PCT:.0f} %:  "
          f"N_структ={d['N_struct']} (cond={cs:.1e})  vs  N_штраф={d['N_pen']} (cond={cp:.1e})")
    print(f"при малом γ={GAMMA_SMALL:.0e}: штраф упирается в «пол» "
          f"{100*min(r['err_pen'] for r in d_small['rows']):.2f} % (ГУ не выполнены)")
    print(f"файл: {path}")


if __name__ == "__main__":
    main()
