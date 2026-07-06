#!/usr/bin/env python3
"""run_circle.py — верификация 1D↔2D на круге (эталонная таблица).

Двухстрочная верификация: численный метод ↔ аналитика мягкого шарнира (< 0.1 %),
и модельный разрыв мягкого шарнира с Кирхгофом (4.2) ≈ 26.4 %. Круг при h=1.0.

Расчётная часть вынесена в :func:`run_circle_golden` (принимает GoldenConfig).
"""

from __future__ import annotations

from plate_solver import analytic, geometry, viz
from plate_solver.plate import PlateBending


def run_circle_golden(cfg):
    """-> (dict, fig). Перебор p из cfg.p_sweep; поверхность прогиба при p=cfg.p."""
    a, q, nu, D = cfg.a, cfg.q0, cfg.nu, cfg.D
    dom = geometry.make_circle(a)
    w_soft = analytic.circular_plate_soft_hinge_wmax(q, a, D)
    w_ss = float(analytic.circular_plate_simply_supported(0.0, q, a, nu, D))

    def _solve(p):
        lab = cfg.to_config(h=cfg.h_circle, Q=cfg.Q_circle, p=p)
        pb = PlateBending.from_config(dom, lab)
        _, cw = pb.solve_uniform(q)
        return lab, pb, cw

    rows = {}
    fig = None
    for p in cfg.p_sweep:
        lab, pb, cw = _solve(p)
        w0 = float(pb.deflection(cw, 0.0, 0.0))
        rows[p] = {"w_max": w0, "err": abs(w0 - w_soft) / w_soft, "cond": pb.poisson.cond}
        if p == cfg.p:
            fig = viz.plot_deflection_surface(lab, pb, cw)
    if fig is None:  # рабочий p не в p_sweep — построить поверхность отдельно
        lab, pb, cw = _solve(cfg.p)
        fig = viz.plot_deflection_surface(lab, pb, cw)

    data = {
        "rows": rows, "w_soft": w_soft, "w_SS": w_ss,
        "ratio": w_soft / w_ss, "model_gap_pct": 100.0 * abs(w_soft - w_ss) / w_ss,
        "D": D,
    }
    return data, fig


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from golden_config import GoldenConfig

    cfg = GoldenConfig()
    data, fig = run_circle_golden(cfg)
    print(f"Круг: a={cfg.a}, q0={cfg.q0}, ν={cfg.nu}, D={data['D']:.6g}, Q={cfg.Q_circle}")
    print(f"w_soft={data['w_soft']:.6e}  w_SS(4.2)={data['w_SS']:.6e}")
    print(f"{'p':>3} | {'w_max':>14} | {'отн.ошибка':>11} | {'cond(A)':>10}")
    for p, r in data["rows"].items():
        print(f"{p:>3} | {r['w_max']:>14.8g} | {r['err']:>11.2e} | {r['cond']:>10.2e}")
    print(f"модельный разрыв с (4.2): {data['model_gap_pct']:.2f} %")
    plt.close(fig)


if __name__ == "__main__":
    main()
