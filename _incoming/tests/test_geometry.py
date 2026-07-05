"""Тесты геометрии: знак ω, граница, входящий угол L-формы, R-операции, ∇ω.

Проверяют реализованную часть: символьная ω (SymPy) и её точный градиент.
Требуют sympy и numpy.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
from plates import geometry as geo


def test_circle_sign_inside_boundary_outside():
    dom = geo.make_circle(1.0)
    assert dom.omega(0.0, 0.0) > 0          # центр внутри
    assert abs(float(dom.omega(1.0, 0.0))) < 1e-12  # точка на границе r=1
    assert dom.omega(2.0, 0.0) < 0          # снаружи


def test_circle_value_center():
    # ω = (a² − r²)/(2a); в центре = a/2.
    dom = geo.make_circle(2.0)
    assert float(dom.omega(0.0, 0.0)) == pytest.approx(1.0)


def test_circle_normalized_first_order():
    # |∇ω| = 1 на границе. Для круга ∇ω = (−x/a, −y/a); в (a,0) даёт (−1, 0).
    dom = geo.make_circle(2.0)
    gx, gy = dom.grad_omega(2.0, 0.0)
    assert np.hypot(float(gx), float(gy)) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
#  Тест-ворота шага (критерий «зелёного»)
# --------------------------------------------------------------------------- #
def test_gate_circle_sign_inside_boundary_outside():
    """ω > 0 внутри, ≈ 0 на границе, < 0 снаружи — набором точек (вектор)."""
    a = 1.5
    dom = geo.make_circle(a)
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)

    # Внутри (r = 0.5 a) — строго положительно.
    rin = 0.5 * a
    assert np.all(dom.omega(rin * np.cos(theta), rin * np.sin(theta)) > 0)
    # На границе (r = a) — ноль с точностью до округления.
    on = dom.omega(a * np.cos(theta), a * np.sin(theta))
    assert np.max(np.abs(on)) < 1e-12
    # Снаружи (r = 1.5 a) — строго отрицательно.
    rout = 1.5 * a
    assert np.all(dom.omega(rout * np.cos(theta), rout * np.sin(theta)) < 0)


def test_gate_circle_gradient_equals_analytic():
    """∇ω круга совпадает с аналитическим (−x/a, −y/a) в произвольных точках."""
    a = 1.5
    dom = geo.make_circle(a)
    X = np.array([0.0, 0.3, -0.7, 1.0, 0.5])
    Y = np.array([0.0, -0.4, 0.2, -1.0, 0.9])
    gx, gy = dom.grad_omega(X, Y)
    np.testing.assert_allclose(gx, -X / a, atol=1e-12)
    np.testing.assert_allclose(gy, -Y / a, atol=1e-12)


def test_rectangle_inside_outside():
    dom = geo.make_rectangle(0.0, 2.0, 0.0, 2.0)
    assert dom.omega(1.0, 1.0) > 0           # центр
    assert dom.omega(-0.5, 1.0) < 0          # слева снаружи
    assert dom.omega(1.0, 3.0) < 0           # сверху снаружи


def test_lshape_sign_and_reentrant():
    # L = [0,1]² без выреза [0.5,1]×[0.5,1]; входящий угол (0.5, 0.5).
    dom = geo.make_L(side=1.0, cut=0.5)
    assert dom.bbox == (0.0, 1.0, 0.0, 1.0)
    assert dom.omega(0.25, 0.25) > 0         # ядро
    assert dom.omega(0.75, 0.25) > 0         # нижняя полоса
    assert dom.omega(0.25, 0.75) > 0         # левая полоса
    assert dom.omega(0.75, 0.75) < 0         # вырезанный угол — снаружи


def test_r_operations_symbolic():
    one, minus_one = sp.Float(1.0), sp.Float(-1.0)
    assert float(geo.r_and(one, minus_one)) < 0     # пересечение: одна снаружи
    assert float(geo.r_and(one, sp.Float(2.0))) > 0
    assert float(geo.r_or(one, minus_one)) > 0      # объединение: одна внутри
    assert float(geo.r_or(minus_one, sp.Float(-2.0))) < 0


def test_omega_vectorized():
    dom = geo.make_circle(1.0)
    X = np.array([0.0, 0.5, 1.0])
    Y = np.zeros_like(X)
    w = dom.omega(X, Y)
    assert w.shape == (3,)
    assert w[0] > 0 and abs(w[2]) < 1e-12


def test_grad_matches_finite_difference():
    # Точный символьный градиент должен совпасть с центральной разностью
    # в гладких внутренних точках (вдали от углов, где sqrt негладок).
    dom = geo.make_L(side=1.0, cut=0.5)
    eps = 1e-6
    for px, py in [(0.30, 0.20), (0.20, 0.70), (0.60, 0.10)]:
        gx, gy = dom.grad_omega(px, py)
        fdx = (float(dom.omega(px + eps, py)) - float(dom.omega(px - eps, py))) / (2 * eps)
        fdy = (float(dom.omega(px, py + eps)) - float(dom.omega(px, py - eps))) / (2 * eps)
        assert float(gx) == pytest.approx(fdx, abs=1e-4)
        assert float(gy) == pytest.approx(fdy, abs=1e-4)
