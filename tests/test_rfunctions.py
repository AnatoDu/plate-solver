"""Проверка R-функций: ω>0 внутри, ω≈0 на границе, ω<0 снаружи; вырезы."""

import numpy as np

from plate_solver import geometry as g


def test_circle_signs():
    # единичный круг в начале координат
    assert g.circle(0.0, 0.0, radius=1.0) > 0          # центр — внутри
    assert np.isclose(g.circle(1.0, 0.0, radius=1.0), 0.0)  # точка на границе
    assert g.circle(2.0, 0.0, radius=1.0) < 0          # снаружи


def test_rectangle_signs():
    # прямоугольник |x|<=2, |y|<=1
    assert g.rectangle(0.0, 0.0, ax=2.0, ay=1.0) > 0   # центр
    assert g.rectangle(3.0, 0.0, ax=2.0, ay=1.0) < 0   # за пределом по x
    assert g.rectangle(0.0, 2.0, ax=2.0, ay=1.0) < 0   # за пределом по y


def test_conjunction_disjunction_logic():
    # пересечение двух кругов: точка внутри обоих -> >0
    w1 = g.circle(0.3, 0.0, cx=-0.5, radius=1.0)
    w2 = g.circle(0.3, 0.0, cx=0.5, radius=1.0)
    assert g.r_conjunction(w1, w2) > 0
    # объединение: точка внутри хотя бы одного -> >0
    w1o = g.circle(-1.2, 0.0, cx=-0.5, radius=1.0)   # внутри левого только
    w2o = g.circle(-1.2, 0.0, cx=0.5, radius=1.0)
    assert g.r_disjunction(w1o, w2o) > 0


def test_difference_creates_hole():
    # большой круг с круглым вырезом в центре
    outer = g.circle(0.0, 0.0, radius=2.0)
    hole = g.circle(0.0, 0.0, radius=0.5)
    omega = g.difference(outer, hole)
    assert omega < 0          # точка в дырке — вне области
    # точка в кольце — внутри
    outer2 = g.circle(1.0, 0.0, radius=2.0)
    hole2 = g.circle(1.0, 0.0, radius=0.5)
    assert g.difference(outer2, hole2) > 0


def test_vectorized():
    x = np.linspace(-2, 2, 11)
    y = np.zeros_like(x)
    w = g.circle(x, y, radius=1.0)
    assert w.shape == x.shape
    assert w[5] > 0 and w[0] < 0    # центр внутри, край массива снаружи
