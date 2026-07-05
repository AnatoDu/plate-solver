#!/usr/bin/env python3
"""run_lshape_contact.py — контакт L-формы с жёстким основанием (МОР) → Табл. 4.2.

L-форма прижимается к плоскому жёсткому основанию под всей Ω с зазором Δ
(передаётся снаружи — единый для контакта и КТН). L-серия при h=cfg.h_ktn.

Расчётная часть — :func:`run_lshape_contact_golden(cfg, Delta)`.
"""

from __future__ import annotations

import numpy as np

from plate_solver import geometry, viz
from plate_solver.config import Config
from plate_solver.contact import ContactMOR
from plate_solver.plate import PlateBending


def lshape_lab_config(cfg) -> Config:
    """Лабораторный Config для L-серии из золотого конфига (единая толщина h_ktn)."""
    return Config(nu=cfg.nu, q0=cfg.q0, h=cfg.h_ktn, E=cfg.E, p=cfg.p, Q=cfg.Q_lshape,
                  grid_n=cfg.grid_n, beta=cfg.beta, max_iter=cfg.mor_iter, tol=cfg.mor_tol)


def compute_w_free_lshape(cfg) -> float:
    """Свободный прогиб L-формы без контакта: max|w| при q0, p, Q, h_ktn (опора для Δ)."""
    dom = geometry.make_L(cfg.L_side, cfg.L_cut)
    pb = PlateBending.from_config(dom, lshape_lab_config(cfg))
    q = pb.quad
    _, cw = pb.solve_uniform(cfg.q0)
    return float(np.max(np.abs(pb.deflection(cw, q.x, q.y))))


def run_lshape_contact_golden(cfg, Delta):
    """-> (dict, fig, ContactResult). Контакт при едином Δ, основание под всей Ω."""
    dom = geometry.make_L(cfg.L_side, cfg.L_cut)
    lab = lshape_lab_config(cfg)
    pb = PlateBending.from_config(dom, lab)
    res = ContactMOR(pb, lab, gap=Delta).solve()

    h = res.residual_history
    drop = float(h[0] / h[-1]) if h[-1] > 0 else float("inf")
    px, py = res.peak_xy
    data = {
        "iters": res.iters, "converged": res.converged,
        "residual_first": float(h[0]), "residual_last": float(h[-1]), "residual_drop": drop,
        "r_min": float(res.r_nodes.min()), "r_max": float(res.r_nodes.max()),
        "n_contact": int((res.r_nodes > 0).sum()), "n_total": int(res.r_nodes.size),
        "peak_xy": (float(px), float(py)), "d_corner": float(np.hypot(px - 0.5, py - 0.5)),
        "w_under_max": float(res.w_nodes.max()),
        "comp_residual": float(res.comp_residual), "gap_overshoot": float(res.gap_overshoot),
    }
    fig = viz.plot_contact_summary(lab, res)
    return data, fig, res


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from golden_config import GoldenConfig

    cfg = GoldenConfig()
    w_free = compute_w_free_lshape(cfg)
    Delta = cfg.gap_factor * w_free
    data, fig, _ = run_lshape_contact_golden(cfg, Delta)
    print(f"L-форма (h={cfg.h_ktn}): w_free={w_free:.4e}, Δ={Delta:.4e}")
    print(f"МОР: итераций {data['iters']}, невязка {data['residual_first']:.2e}→"
          f"{data['residual_last']:.2e} (×{data['residual_drop']:.0f})")
    print(f"контакт: {data['n_contact']}/{data['n_total']} узлов, r∈[{data['r_min']:.2e},"
          f"{data['r_max']:.3e}], пик {data['peak_xy']}, d={data['d_corner']:.3f}")
    print(f"max w под контактом = {data['w_under_max']:.4e} (≈ Δ = {Delta:.4e})")
    print(f"комплементарность: max|r·(w−Δ)|/(q0·Δ) = {data['comp_residual']:.2e}, "
          f"перелёт зазора (max w_cont − Δ)/Δ = {data['gap_overshoot']:.2e}")
    plt.close(fig)


if __name__ == "__main__":
    main()
