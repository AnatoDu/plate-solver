#!/usr/bin/env python3
"""run_ktn.py — поправки КТН на контакте L-формы: классика ↔ уточнённая → Табл. 4.3.

При едином Δ (как в контакте) и h=cfg.h_ktn сравниваются классика (Кирхгоф) и
КТН. Главные эффекты: КТН снимает сингулярность давления под штампом (сглаживает
реакцию) и даёт поправку w_max (теория мягче).

Расчётная часть — :func:`run_ktn_golden(cfg, Delta, classic_res)`.
"""

from __future__ import annotations

import numpy as np
from run_lshape_contact import lshape_lab_config

from plate_solver import geometry
from plate_solver.contact import ContactMOR
from plate_solver.ktn import KTNParams
from plate_solver.plate import PlateBending


def _roughness(r) -> float:
    return float(np.sqrt(np.sum(np.diff(r) ** 2)))


def run_ktn_golden(cfg, Delta, classic_res):
    """-> (dict, fig). КТН-контакт при едином Δ; ``classic_res`` — классика из Табл. 4.2.

    Классическая колонка переиспользуется из контактного прогона (тот же Δ, h,
    основание) ⇒ Табл. 4.3.классика == Табл. 4.2. Здесь считается только КТН-колонка.
    """
    import matplotlib.pyplot as plt

    dom = geometry.make_L(cfg.L_side, cfg.L_cut)
    lab = lshape_lab_config(cfg)
    pb = PlateBending.from_config(dom, lab)
    ktn_res = ContactMOR(pb, lab, gap=Delta, ktn=KTNParams.from_config(lab)).solve()
    kp = KTNParams.from_config(lab)

    rc, rk = classic_res, ktn_res
    data = {
        "h": cfg.h_ktn, "h_psi2": kp.h_psi2, "h_star2": kp.h_star2, "c_curv": kp.c_curv,
        "classic": {"peak": float(rc.r_nodes.max()), "nodes": int((rc.r_nodes > 0).sum()),
                    "rough": _roughness(rc.r_nodes), "w_max": float(rc.w_nodes.max())},
        "ktn": {"peak": float(rk.r_nodes.max()), "nodes": int((rk.r_nodes > 0).sum()),
                "rough": _roughness(rk.r_nodes), "w_max": float(rk.w_ktn_nodes.max())},
    }
    data["peak_ratio"] = data["ktn"]["peak"] / data["classic"]["peak"]
    data["rough_ratio"] = data["ktn"]["rough"] / data["classic"]["rough"]
    data["wmax_corr_pct"] = 100.0 * (data["ktn"]["w_max"] - data["classic"]["w_max"]) / data["classic"]["w_max"]

    # Фигура: классика vs КТН, общая цветовая шкала (vmax по классике).
    vmax = float(np.nanmax(rc.r_grid))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, res, title in [(axes[0], rc, "Классика (Кирхгоф)"), (axes[1], rk, "КТН (уточнённая)")]:
        pcm = ax.pcolormesh(res.Xg, res.Yg, np.ma.masked_invalid(res.r_grid),
                            cmap="magma", vmin=0, vmax=vmax, shading="auto")
        if res.contact_zone.any():
            ax.contour(res.Xg, res.Yg, res.contact_zone.astype(float),
                       levels=[0.5], colors="cyan", linewidths=1.2)
        ax.contour(res.Xg, res.Yg, dom.omega(res.Xg, res.Yg), levels=[0], colors="k", linewidths=1)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nпик r = {res.r_nodes.max():.1f}, узлов {int((res.r_nodes > 0).sum())}")
    fig.colorbar(pcm, ax=axes, label="реакция r", shrink=0.8)
    fig.suptitle(f"Контактная реакция: классика vs КТН (h={cfg.h_ktn}, Δ={Delta:.2e}) — "
                 "КТН снимает сингулярность штампа", fontsize=12)
    return data, fig


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from golden_config import GoldenConfig
    from run_lshape_contact import compute_w_free_lshape, run_lshape_contact_golden

    cfg = GoldenConfig()
    Delta = cfg.gap_factor * compute_w_free_lshape(cfg)
    _, _, classic_res = run_lshape_contact_golden(cfg, Delta)
    data, fig = run_ktn_golden(cfg, Delta, classic_res)
    c, k = data["classic"], data["ktn"]
    print(f"КТН (h={data['h']}): пик {c['peak']:.1f}→{k['peak']:.1f} (×{data['peak_ratio']:.3f}), "
          f"узлы {c['nodes']}→{k['nodes']}, w_max {c['w_max']:.4e}→{k['w_max']:.4e} "
          f"({data['wmax_corr_pct']:+.1f} %)")
    plt.close(fig)


if __name__ == "__main__":
    main()
