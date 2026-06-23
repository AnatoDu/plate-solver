"""Подпакет geometry — задание границы произвольной области R-функциями.

Это защищаемый «способ задания границы» (положение на защиту № 1).
Реализована система R0 Рвачёва: примитивы (круг, полуплоскость, полоса,
прямоугольник) и операции (R-конъюнкция ∧, R-дизъюнкция ∨, разность),
из которых конструируется аналитическое уравнение границы ω(x, y) = 0
(ω > 0 внутри, ω < 0 снаружи).
"""

from .rfunctions import (
    circle,
    half_plane,
    strip,
    rectangle,
    r_conjunction,
    r_disjunction,
    difference,
)

__all__ = [
    "circle",
    "half_plane",
    "strip",
    "rectangle",
    "r_conjunction",
    "r_disjunction",
    "difference",
]
