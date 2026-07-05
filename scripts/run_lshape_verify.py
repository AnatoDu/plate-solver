#!/usr/bin/env python3
"""run_lshape_verify.py — независимая сверка изгиба L-формы через scikit-fem.

Две колонки: RFM ↔ FEM-Marcus (та же модель, ~2.6 %) и RFM ↔ FEM-Kirchhoff
(парадокс Сапонджяна, ~40–55 %). L-серия при h=cfg.h_ktn (единая толщина).

Расчётная часть — :func:`run_lshape_verify_golden` (принимает GoldenConfig).
"""

from __future__ import annotations

from plate_solver import verify_fem as vf
from plate_solver.config import Config


def run_lshape_verify_golden(cfg):
    """-> dict. RFM против FEM-Marcus и FEM-Kirchhoff на L-форме (h=cfg.h_ktn)."""
    lab = Config(nu=cfg.nu, q0=cfg.q0, h=cfg.h_ktn, E=cfg.E, p=cfg.p,
                 Q=cfg.Q_lshape, grid_n=cfg.grid_n)
    c = vf.compare_rfm_vs_fem(lab, side=cfg.L_side, cut=cfg.L_cut,
                              mesh_m=cfg.fem_mesh_m, refine=cfg.fem_refine)
    return {
        "rfm_w_max": c.w_rfm_max,
        "marcus_w_max": c.w_marcus_max,
        "kirch_w_max": c.w_kirchhoff_max,
        "rfm_vs_marcus_pct": c.rel_marcus_pct,
        "rfm_vs_kirchhoff_pct": c.rel_kirchhoff_pct,
        "n_points": c.n_points,
    }


def main() -> None:
    from golden_config import GoldenConfig

    cfg = GoldenConfig()
    d = run_lshape_verify_golden(cfg)
    print(f"L-форма (h={cfg.h_ktn}), точек сверки {d['n_points']}")
    print(f"  RFM           w_max = {d['rfm_w_max']:.4e}")
    print(f"  FEM-Marcus    w_max = {d['marcus_w_max']:.4e}   RFM↔Marcus = {d['rfm_vs_marcus_pct']:.3f} %")
    print(f"  FEM-Kirchhoff w_max = {d['kirch_w_max']:.4e}   RFM↔Kirchhoff = {d['rfm_vs_kirchhoff_pct']:.3f} %")


if __name__ == "__main__":
    main()
