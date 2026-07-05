"""Реестр геометрий фазы 2 (P1): annulus, r_not/r_diff, мини-язык compose.

Ворота P1.3: (а) площадь по маске квадратуры против точной; (б) знак ω;
(в) символьный ∇ω против центральной разности; (г) smoke compose-области
через решатель Пуассона; (д) ограда глубины.
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver import geometry as geo
from plate_solver import quadrature as quad
from plate_solver.problem import CaseError

A, B = 1.0, 0.4

# difference(union(rect, rect), circle) — глубина 3, 4 примитива + 2 op = 6 узлов
TREE_D3 = {"op": "difference", "children": [
    {"op": "union", "children": [
        {"kind": "rectangle", "x1": 0.0, "x2": 2.0, "y1": 0.0, "y2": 1.0},
        {"kind": "rectangle", "x1": 0.0, "x2": 1.0, "y1": 0.0, "y2": 2.0},
    ]},
    {"kind": "circle", "a": 0.25, "cx": 0.5, "cy": 0.5},
]}


# --------------------------------------------------------------------------- #
#  (а) площадь по маске против точной: кольцо π(a² − b²), допуск ~1/Q
# --------------------------------------------------------------------------- #
def test_gate_annulus_area_vs_exact():
    dom = geo.make_annulus(A, B)
    exact = np.pi * (A**2 - B**2)
    errs = []
    for Q in (128, 256):
        area = float(quad.interior_nodes(dom, Q).w.sum())
        rel = abs(area - exact) / exact
        assert rel < 3.0 / Q, (Q, rel)      # ступенчатая маска двух границ ~1/Q
        errs.append(rel)
    assert errs[1] < errs[0]                # и убывает с Q


# --------------------------------------------------------------------------- #
#  (б) знак ω: внутри / снаружи / в дырке
# --------------------------------------------------------------------------- #
def test_annulus_signs():
    dom = geo.make_annulus(A, B)
    mid = 0.5 * (A + B)
    assert dom.omega(mid, 0.0) > 0          # в теле кольца
    assert dom.omega(0.0, 0.0) < 0          # в дырке
    assert dom.omega(0.0, -1.5 * A) < 0     # снаружи
    assert abs(dom.omega(A, 0.0)) < 1e-14   # внешняя граница
    assert abs(dom.omega(0.0, B)) < 1e-14   # внутренняя граница


def test_compose_signs():
    dom = geo.make_compose(TREE_D3)
    assert dom.omega(1.5, 0.5) > 0          # в горизонтальной полосе
    assert dom.omega(0.5, 1.5) > 0          # в вертикальной полосе
    assert dom.omega(0.5, 0.5) < 0          # в круглом вырезе
    assert dom.omega(1.5, 1.5) < 0          # вне L-объединения
    assert dom.bbox == (0.0, 2.0, 0.0, 2.0)  # bbox difference = bbox первого


def test_compose_bbox_rules():
    union = geo.make_compose({"op": "union", "children": [
        {"kind": "circle", "a": 1.0},
        {"kind": "circle", "a": 1.0, "cx": 3.0, "cy": 0.0}]})
    assert union.bbox == (-1.0, 4.0, -1.0, 1.0)
    inter = geo.make_compose({"op": "intersect", "children": [
        {"kind": "rectangle", "x1": 0.0, "x2": 2.0, "y1": 0.0, "y2": 2.0},
        {"kind": "rectangle", "x1": 1.0, "x2": 3.0, "y1": 1.0, "y2": 3.0}]})
    assert inter.bbox == (1.0, 2.0, 1.0, 2.0)
    with pytest.raises(ValueError, match="пусто"):
        geo.make_compose({"op": "intersect", "children": [
            {"kind": "circle", "a": 0.5},
            {"kind": "circle", "a": 0.5, "cx": 5.0, "cy": 0.0}]})


# --------------------------------------------------------------------------- #
#  (в) точный (символьный) ∇ω против центральной разности
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("domain", [geo.make_annulus(A, B), geo.make_compose(TREE_D3)],
                         ids=["annulus", "compose"])
def test_grad_omega_vs_central_difference(domain):
    rng = np.random.default_rng(42)
    x0, x1, y0, y1 = domain.bbox
    pts_x, pts_y = [], []
    while len(pts_x) < 20:                   # 20 случайных строго внутренних точек
        X = rng.uniform(x0, x1, 200)
        Y = rng.uniform(y0, y1, 200)
        keep = domain.omega(X, Y) > 0.05     # вдали от границы (там ω гладкая)
        pts_x.extend(X[keep])
        pts_y.extend(Y[keep])
    X = np.array(pts_x[:20])
    Y = np.array(pts_y[:20])
    gx, gy = domain.grad_omega(X, Y)
    h = 1e-5
    gx_fd = (domain.omega(X + h, Y) - domain.omega(X - h, Y)) / (2 * h)
    gy_fd = (domain.omega(X, Y + h) - domain.omega(X, Y - h)) / (2 * h)
    scale = np.maximum(np.hypot(gx, gy), 1e-12)
    assert np.max(np.abs(gx - gx_fd) / scale) < 1e-6
    assert np.max(np.abs(gy - gy_fd) / scale) < 1e-6


# --------------------------------------------------------------------------- #
#  (г) smoke: compose глубины 3 работает в решателе Пуассона
# --------------------------------------------------------------------------- #
def test_compose_depth3_poisson_smoke():
    from plate_solver import basis as B_
    from plate_solver.poisson import PoissonSolver

    dom = geo.make_compose(TREE_D3)
    sol = PoissonSolver(dom, B_.ChebyshevBasis(6, dom.bbox), quad.interior_nodes(dom, 64))
    c = sol.solve(np.ones(sol.quad.x.size))
    v = sol.evaluate(c, 1.5, 0.5)            # точка в теле области
    assert np.isfinite(sol.cond)
    assert np.isfinite(v) and v > 0.0        # −Δv = 1 ⇒ v > 0 внутри


# --------------------------------------------------------------------------- #
#  (д) ограда: глубина 4 отклоняется (единый валидатор problem.py)
# --------------------------------------------------------------------------- #
def test_compose_depth4_rejected():
    deep = {"op": "union", "children": [
        {"op": "union", "children": [
            {"op": "union", "children": [
                {"kind": "circle", "a": 1.0},
                {"kind": "circle", "a": 0.5}]},
            {"kind": "circle", "a": 0.4}]},
        {"kind": "circle", "a": 0.3}]}
    with pytest.raises(CaseError, match="глубина"):
        geo.make_compose(deep)


def test_r_not_r_diff_formulas():
    """r_not — смена знака; r_diff(f1, f2) = r_and(f1, −f2) (символьно)."""
    import sympy as sp

    f1 = geo.circle_expr(1.0)
    f2 = geo.circle_expr(0.4)
    assert sp.simplify(geo.r_not(f1) + f1) == 0
    assert sp.simplify(geo.r_diff(f1, f2) - geo.r_and(f1, geo.r_not(f2))) == 0
