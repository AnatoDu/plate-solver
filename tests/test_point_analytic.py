"""Аналитика точечной силы в центре круга (фаза 2, P3.5).

Ворота: sympy-подстановка вне r = 0 (бигармоника однородна; для soft:
−ΔM = 0 при r > 0, M(a) = 0, −Δw = M/D), краевые условия, контроль
«предел ν → 1 формулы Тимошенко для опёртой пластины» (паттерн NOTES §8).
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from plate_solver import analytic

A, P, D = 1.0, 1.0, 1.832e4

r = sp.symbols("r", positive=True)


def _lap(f):
    return sp.diff(f, r, 2) + sp.diff(f, r) / r


def test_point_clamped_biharmonic_and_bc():
    w = P / (16 * sp.pi * D) * (A**2 - r**2 + 2 * r**2 * sp.log(r / A))
    assert sp.simplify(_lap(_lap(w))) == 0          # однородна при r > 0
    assert sp.simplify(w.subs(r, A)) == 0           # w(a) = 0
    assert sp.simplify(sp.diff(w, r).subs(r, A)) == 0   # w'(a) = 0 (защемление)
    assert analytic.circle_point_clamped_wmax(A, P, D) == \
        pytest.approx(P * A**2 / (16 * np.pi * D))


def test_point_soft_split_identities():
    """Расщепление: −ΔM = 0 (r>0), M(a)=0; −Δw = M/D; w(a)=0."""
    M = P / (2 * sp.pi) * sp.log(A / r)
    w = P / (8 * sp.pi * D) * (A**2 - r**2 * (1 + sp.log(A / r)))
    assert sp.simplify(_lap(M)) == 0
    assert sp.simplify(M.subs(r, A)) == 0
    assert sp.simplify(_lap(w) + M / D) == 0        # −Δw = M/D
    assert sp.simplify(w.subs(r, A)) == 0
    assert analytic.circle_point_soft_wmax(A, P, D) == \
        pytest.approx(P * A**2 / (8 * np.pi * D))


def test_point_soft_equals_timoshenko_nu1_limit():
    """Контроль: w_soft = предел ν→1 формулы Тимошенко для опёртой пластины."""
    nu = sp.symbols("nu")
    w_timo = P / (16 * sp.pi * D) * ((3 + nu) / (1 + nu) * (A**2 - r**2)
                                     + 2 * r**2 * sp.log(r / A))
    w_soft = P / (8 * sp.pi * D) * (A**2 - r**2 * (1 + sp.log(A / r)))
    assert sp.simplify(w_timo.subs(nu, 1) - w_soft) == 0


def test_point_fields_finite_at_center():
    """Прогиб конечен в центре (r² ln r → 0), поле M логарифмически растёт."""
    w_c = analytic.circle_point_clamped(np.array([0.0, 0.5, 1.0]), A, P, D)
    w_s = analytic.circle_point_soft(np.array([0.0, 0.5, 1.0]), A, P, D)
    assert np.all(np.isfinite(w_c)) and np.all(np.isfinite(w_s))
    assert w_c[0] == pytest.approx(analytic.circle_point_clamped_wmax(A, P, D))
    assert w_s[0] == pytest.approx(analytic.circle_point_soft_wmax(A, P, D))
    assert abs(w_c[2]) < 1e-15 and abs(w_s[2]) < 1e-15
    assert w_s[0] == pytest.approx(2.0 * w_c[0])    # мягче защемления ровно вдвое
