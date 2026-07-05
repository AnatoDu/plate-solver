"""Тест-ворота квадратуры: ∫_Ω 1 dΩ для круга ≈ π·a², ошибка падает с Q.

Также проверяем точную нормировку весов (сумма по bbox = площадь bbox) и
точность на прямоугольнике (все узлы Гаусса строго внутри ⇒ площадь точна).
"""

from __future__ import annotations

import numpy as np
from plates import geometry as geo
from plates import quadrature as quad


def test_weight_sum_equals_bbox_area():
    # Σ W по всему bbox = площадь bbox (якобиан зашит в веса).
    bbox = (-1.0, 2.0, 0.5, 3.0)
    _, _, W = quad.gauss_legendre_grid(24, bbox)
    area_bbox = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])
    assert abs(W.sum() - area_bbox) < 1e-12


def test_rectangle_area_exact():
    # Прямоугольник: узлы Гаусса строго внутри ⇒ маска оставляет все ⇒ площадь точна.
    dom = geo.make_rectangle(0.0, 2.0, 0.0, 3.0)
    X, Y, W = quad.gauss_legendre_grid(20, dom.bbox)
    m = quad.interior_mask(dom, X, Y)
    assert abs(W[m].sum() - 6.0) < 1e-10


def test_gate_circle_area_converges():
    """∫_Ω 1 dΩ для круга → π·a², и ошибка монотонно падает с ростом Q."""
    a = 1.0
    dom = geo.make_circle(a)
    exact = np.pi * a**2

    errs = []
    for Q in (8, 16, 32, 64, 128):
        X, Y, W = quad.gauss_legendre_grid(Q, dom.bbox)
        area = W[quad.interior_mask(dom, X, Y)].sum()
        errs.append(abs(area - exact))

    errs = np.array(errs)
    # 1) ошибка строго убывает с Q (ступенчатая маска ⇒ ~O(1/Q));
    assert np.all(np.diff(errs) < 0), f"ошибка не убывает монотонно: {errs}"
    # 2) заметное падение (≈ деление пополам при удвоении Q): err(8) >> err(128);
    assert errs[0] > 5.0 * errs[-1]
    # 3) на Q=128 относительная погрешность < 1 %.
    assert errs[-1] / exact < 1e-2


def test_interior_nodes_helper_matches_manual():
    dom = geo.make_circle(1.0)
    X, Y, W = quad.gauss_legendre_grid(32, dom.bbox)
    m = quad.interior_mask(dom, X, Y)
    xi, yi, wi = quad.interior_nodes(dom, 32)
    assert xi.size == int(m.sum())
    assert abs(wi.sum() - W[m].sum()) < 1e-14
