#!/usr/bin/env python3
r"""run_ladder_circle.py — СТУПЕНЬ 2 лестницы: осесимметричная круглая пластина.

a=1, q=4, ν=0.3. RFM + Ритц на круге (структура шарнир ``ω·Φ`` / защемление ``ω²·Φ``),
базис Чебышёва, p-sweep. Переиспользует готовые решатели ``PlateBending`` (мягкий
шарнир) и ``ClampedPlate`` (защемление).

  • защемление — сверка с ТОЧНОЙ формулой ``w_max = q a⁴/(64D)`` (модельной
    погрешности нет; ошибка падает с Q — нет «пола»);
  • шарнир — реализован МЯГКИЙ шарнир; сверка с ЕГО точным значением
    ``w = q(a²−r²)(3a²−r²)/(64D)``; отклонение от Кирхгофа (≈26 %) — модельная
    погрешность (парадокс), как в 4.1.
"""

from __future__ import annotations

from plate_solver import analytic, geometry
from plate_solver.clamped import ClampedPlate
from plate_solver.config import Config
from plate_solver.plate import PlateBending

A, Q0, NU, E, H = 1.0, 4.0, 0.3, 2.1e6, 1.0
P_SWEEP = (2, 4, 6, 8, 10)
Q_QUAD = 1024


def run_ladder_circle():
    """-> dict: p-sweep защемления (к точной 4.1) и мягкого шарнира (к мягкому эталону)."""
    D = E * H**3 / (12 * (1 - NU**2))
    dom = geometry.make_circle(A)
    w_clamped_ex = float(analytic.clamped_uniform_wmax(A, Q0, D))            # q a⁴/64D
    w_soft_ex = float(analytic.circular_plate_soft_hinge_wmax(Q0, A, D))     # мягкий шарнир
    w_ss_kirch = float(analytic.circular_plate_simply_supported(0.0, Q0, A, NU, D))  # Кирхгоф

    clamped_rows, hinge_rows = {}, {}
    for p in P_SWEEP:
        cfg = Config(a=A, q0=Q0, nu=NU, h=H, E=E, p=p, Q=Q_QUAD)
        cp = ClampedPlate.from_config(dom, cfg)
        w0c = float(cp.deflection(cp.solve_uniform(Q0), 0.0, 0.0))
        clamped_rows[p] = {"w_max": w0c, "err": abs(w0c - w_clamped_ex) / w_clamped_ex,
                           "cond": cp.cond}
        pb = PlateBending.from_config(dom, cfg)
        _, cw = pb.solve_uniform(Q0)
        w0h = float(pb.deflection(cw, 0.0, 0.0))
        hinge_rows[p] = {"w_max": w0h, "err": abs(w0h - w_soft_ex) / w_soft_ex,
                         "cond": pb.poisson.cond}

    # контроль по Q для защемления (нет модельного «пола»)
    q_rows = {}
    for Q in (64, 128, 256, 512, 1024):
        cp = ClampedPlate.from_config(dom, Config(a=A, q0=Q0, nu=NU, h=H, E=E, p=6, Q=Q))
        w0 = float(cp.deflection(cp.solve_uniform(Q0), 0.0, 0.0))
        q_rows[Q] = abs(w0 - w_clamped_ex) / w_clamped_ex

    return {
        "D": D, "w_clamped_ex": w_clamped_ex, "w_soft_ex": w_soft_ex, "w_ss_kirch": w_ss_kirch,
        "soft_model_gap_pct": 100.0 * abs(w_soft_ex - w_ss_kirch) / w_ss_kirch,
        "clamped_rows": clamped_rows, "hinge_rows": hinge_rows, "q_rows": q_rows,
    }


def main() -> None:
    d = run_ladder_circle()
    print("=== СТУПЕНЬ 2: КРУГ (RFM+Ритц, без МОР) ===")
    print(f"a={A}, q={Q0}, ν={NU}, D={d['D']:.6g}, Q={Q_QUAD}")
    print(f"-- защемление: эталон (4.1) w_max = q a⁴/64D = {d['w_clamped_ex']:.8e}")
    for p, r in d["clamped_rows"].items():
        print(f"   p={p:2d}: w_max={r['w_max']:.8e}  отн.err={100*r['err']:.4f} %  "
              f"cond={r['cond']:.1e}")
    print("   контроль по Q (p=6): " + " → ".join(f"{100*e:.3f}%" for e in d["q_rows"].values())
          + "  (нет модельного «пола»)")
    print(f"-- шарнир (МЯГКИЙ): эталон w(0)=3qa⁴/64D = {d['w_soft_ex']:.8e}")
    for p, r in d["hinge_rows"].items():
        print(f"   p={p:2d}: w_max={r['w_max']:.8e}  отн.err(мягк.)={100*r['err']:.4f} %  "
              f"cond={r['cond']:.1e}")
    print(f"   модельная погрешность мягкого шарнира к Кирхгофу (4.2): "
          f"{d['soft_model_gap_pct']:.2f} % (парадокс)")


if __name__ == "__main__":
    main()
