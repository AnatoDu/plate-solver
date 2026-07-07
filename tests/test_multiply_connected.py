r"""Ворота многосвязных/неканонических областей (N2 v0.6.0, §5).

Многосвязность берётся RFM естественно: R-операции над ω (вырезы/отверстия).
Нелинейный решатель КТН работает на многосвязной (кольцо, пластина с
отверстием) и неканонической (L-форма) области; редукция КТН→Карман точна и
на них (расширение R1). ⚠️ Gate R5 «вырез→0 = односвязная» требует СВОБОДНОГО
внутреннего края (смешанные КУ §5.2 — задел); при ЗАЩЕМЛЁННОМ отверстии
точечное защемление в центре сохраняется.
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver import theory
from plate_solver.config import Config
from plate_solver.geometry import make_annulus, make_plate_with_hole
from plate_solver.ktn_solver import KTNSolver


def _cfg(h=0.1, q0=1e-4, ns=2):
    return Config(E=1.0, h=h, nu=0.3, a=1.0, q0=q0, p=10, Q=140,
                  n_load_steps=ns, karman_tol=1e-8, karman_max_iter=200)


# --------------------------------------------------------------------------- #
#  Конструкторы многосвязных областей (библиотека §5.3)
# --------------------------------------------------------------------------- #
def test_plate_with_hole_geometry():
    """make_plate_with_hole: ω>0 в теле, ω<0 в отверстии и снаружи (R-разность)."""
    dom = make_plate_with_hole(-1.0, 1.0, -1.0, 1.0, hole_a=0.3)
    assert dom.omega(0.6, 0.0) > 0.0        # в теле пластины
    assert dom.omega(0.0, 0.0) < 0.0        # в отверстии
    assert dom.omega(2.0, 0.0) < 0.0        # снаружи


def test_plate_with_hole_validation():
    """Отверстие обязано лежать целиком внутри прямоугольника."""
    with pytest.raises(ValueError):
        make_plate_with_hole(-1.0, 1.0, -1.0, 1.0, hole_a=1.5)   # больше пластины
    with pytest.raises(ValueError):
        make_plate_with_hole(-1.0, 1.0, -1.0, 1.0, hole_a=0.3, hole_cx=0.9)  # у края


# --------------------------------------------------------------------------- #
#  Нелинейный решатель на многосвязной области + редукция (расширение R1)
# --------------------------------------------------------------------------- #
def test_ktn_solves_on_annulus_multiply_connected():
    """КТН решается на КОЛЬЦЕ (многосвязная, защемление обеих кромок)."""
    r = KTNSolver.from_theory_name(make_annulus(1.0, 0.4), _cfg(), "ktn_full",
                                   bc_type="clamped", inplane_bc="immovable").solve_uniform()
    assert r.converged and np.isfinite(r.w_max) and r.w_max > 0.0


def test_ktn_solves_on_plate_with_hole():
    """КТН решается на ПРЯМОУГОЛЬНИКЕ С ОТВЕРСТИЕМ (многосвязная)."""
    dom = make_plate_with_hole(-1.0, 1.0, -1.0, 1.0, hole_a=0.3)
    r = KTNSolver.from_theory_name(dom, _cfg(q0=1e-5), "ktn_full",
                                   bc_type="clamped", inplane_bc="immovable").solve_uniform()
    assert r.converged and np.isfinite(r.w_max)


def test_reduction_ktn_to_karman_on_annulus():
    """R5 (многосвязная редукция): КТН→Карман точна и на КОЛЬЦЕ (расширение R1)."""
    dom = make_annulus(1.0, 0.4)
    cfg = _cfg()
    rk = KTNSolver.from_theory_name(dom, cfg, "karman", bc_type="clamped",
                                    inplane_bc="immovable").solve_uniform()
    # полная КТН при refinement→0 == Карман (тот же путь, машинная точность)
    kf0 = theory.ktn_full(0.3, cfg.h).with_refinement_scale(0.0)
    r0 = KTNSolver.from_config(dom, cfg, kf0, bc_type="clamped",
                               inplane_bc="immovable").solve_uniform()
    assert abs(r0.w_max - rk.w_max) / rk.w_max < 1e-12
