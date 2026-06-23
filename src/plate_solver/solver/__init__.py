"""Подпакет solver — численное решение на произвольной области.

Здесь живёт численный метод (положение на защиту № 2): решение задач
Пуассона и бигармоники на области, заданной R-функцией ω. На старте —
заготовки с описанием интерфейса; реализация наполняется по главе 2,
вместе с теоремами о сходимости и оценками погрешности.
"""

from .biharmonic import solve_biharmonic, solve_clamped_circular
from .green1d import green_matrix, green_simply_supported
from .poisson import solve_poisson

__all__ = [
    "solve_biharmonic",
    "solve_clamped_circular",
    "solve_poisson",
    "green_simply_supported",
    "green_matrix",
]
