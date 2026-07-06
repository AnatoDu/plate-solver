"""fem-эталон в резолвере: кольцо + существующие пути.

Кольцо (структурированная сетка n_r × n_θ): потолок 2 % пройден, допуски
заморожены «факт × 3»: clamped 5.373e-3 → 1.62e-2, soft 5.443e-3 → 1.64e-2.
Существующие пути подключены без изменений (L/soft: Marcus + Kirchhoff-Морли;
clamped: Аргирис — круг/квадрат/L). Требует scikit-fem (маркер fem).
"""

from __future__ import annotations

import copy

import pytest

pytest.importorskip("skfem")

from plate_solver.dispatch import solve  # noqa: E402
from plate_solver.problem import Problem  # noqa: E402
from plate_solver.references import verify_result  # noqa: E402

pytestmark = pytest.mark.fem

BASE = {
    "load": {"type": "uniform", "q0": 4.0},
    "model": {"h": 0.06},
    "discretization": {"p": 10, "Q": 256, "grid_n": 24},
}


def _report(tol, **over):
    d = copy.deepcopy(BASE)
    d["verify"] = {"reference": "fem", "tol": tol}
    for k, v in over.items():
        d[k] = v
    return verify_result(solve(Problem.from_dict(d)))


def test_gate_fem_annulus_clamped():
    """ВОРОТА: кольцо/Аргирис на структурированной сетке (факт 5.373e-3 × 3)."""
    rep = _report(1.62e-2, geometry={"kind": "annulus", "a": 1.0, "b": 0.4},
                  bc={"type": "clamped"})
    assert rep.ok, "\n" + rep.table()


def test_gate_fem_annulus_soft():
    """ВОРОТА: кольцо/Marcus-P2, шарнир на обеих окружностях (факт 5.443e-3 × 3)."""
    rep = _report(1.64e-2, geometry={"kind": "annulus", "a": 1.0, "b": 0.4},
                  bc={"type": "soft_hinge"})
    assert rep.ok, "\n" + rep.table()


def test_fem_lshape_soft_two_columns():
    """Существующий путь: L/soft — Marcus гейтится, Kirchhoff-парадокс — инфо."""
    rep = _report(5.0e-2, geometry={"kind": "L", "side": 1.0, "cut": 0.5},
                  bc={"type": "soft_hinge"},
                  discretization={"p": 10, "Q": 120, "grid_n": 24})
    assert rep.ok, "\n" + rep.table()          # Marcus ~2.6 % < 5 %
    info = [r for r in rep.rows if not r.gated]
    assert len(info) == 1 and "Морли" in info[0].name
    assert info[0].rel > 0.3                    # парадокс ~53 % — вне допуска


@pytest.mark.parametrize("geometry, disc, tol", [
    ({"kind": "circle", "a": 1.0}, {"p": 10, "Q": 256, "grid_n": 24}, 1.0e-2),
    ({"kind": "rectangle", "x1": 0.0, "x2": 1.0, "y1": 0.0, "y2": 1.0},
     {"p": 8, "Q": 120, "grid_n": 24}, 1.0e-2),
], ids=["circle", "rectangle"])
def test_fem_clamped_existing_paths(geometry, disc, tol):
    """Существующие пути: Аргирис на круге (~0.16 %) и квадрате (~0.05 %)."""
    rep = _report(tol, geometry=geometry, bc={"type": "clamped"},
                  discretization=disc)
    assert rep.ok, "\n" + rep.table()
