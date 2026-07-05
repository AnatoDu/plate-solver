r"""R-функции (система R0 В.Л. Рвачёва).

Каждая область задаётся функцией ω(x, y) такой, что:
    ω > 0  — внутри области,
    ω = 0  — на границе,
    ω < 0  — снаружи.

Примитивы нормированы до первого порядка (|∇ω| = 1 на границе), что важно
для корректной аппроксимации краевых условий. Операции R0:
    конъюнкция (пересечение):  ω₁ ∧ ω₂ = ω₁ + ω₂ − √(ω₁² + ω₂²)
    дизъюнкция (объединение):  ω₁ ∨ ω₂ = ω₁ + ω₂ + √(ω₁² + ω₂²)
Разность (вырез/отверстие): A \ B = A ∧ (−B).

Все функции numpy-векторизованы: x, y могут быть скалярами или массивами.
"""

from __future__ import annotations

import numpy as np


def r_conjunction(w1, w2):
    """R-конъюнкция (пересечение областей), система R0."""
    w1 = np.asarray(w1, float)
    w2 = np.asarray(w2, float)
    return w1 + w2 - np.sqrt(w1**2 + w2**2)


def r_disjunction(w1, w2):
    """R-дизъюнкция (объединение областей), система R0."""
    w1 = np.asarray(w1, float)
    w2 = np.asarray(w2, float)
    return w1 + w2 + np.sqrt(w1**2 + w2**2)


def difference(outer, hole):
    """Разность областей: outer \\ hole = outer ∧ (−hole). Создаёт вырез/отверстие."""
    return r_conjunction(outer, -np.asarray(hole, float))


def circle(x, y, cx: float = 0.0, cy: float = 0.0, radius: float = 1.0):
    r"""Круг радиуса ``radius`` с центром (cx, cy), нормированный до 1-го порядка.

    .. math:: \omega = \frac{radius^2 - (x-c_x)^2 - (y-c_y)^2}{2\,radius}
    """
    if radius <= 0:
        raise ValueError("radius должен быть положительным.")
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return (radius**2 - (x - cx) ** 2 - (y - cy) ** 2) / (2.0 * radius)


def half_plane(x, y, nx: float, ny: float, c: float = 0.0):
    r"""Полуплоскость ``nx·x + ny·y - c >= 0``. Вектор (nx, ny) нормируется."""
    n = float(np.hypot(nx, ny))
    if n == 0:
        raise ValueError("Нормаль (nx, ny) не должна быть нулевой.")
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return (nx * x + ny * y - c) / n


def strip(coord, half_width: float):
    r"""Нормированная полоса ``|coord| <= half_width``.

    .. math:: \omega = \frac{half\_width^2 - coord^2}{2\,half\_width}
    """
    if half_width <= 0:
        raise ValueError("half_width должен быть положительным.")
    coord = np.asarray(coord, float)
    return (half_width**2 - coord**2) / (2.0 * half_width)


def rectangle(x, y, ax: float, ay: float, cx: float = 0.0, cy: float = 0.0):
    r"""Прямоугольник ``|x-cx| <= ax`` и ``|y-cy| <= ay`` (конъюнкция двух полос)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return r_conjunction(strip(x - cx, ax), strip(y - cy, ay))
