r"""Классический изгиб круглой пластины под равномерной нагрузкой q.

Решения по теории Кирхгофа (Тимошенко, «Пластинки и оболочки»). Радиус a,
цилиндрическая жёсткость D. Используются как эталон при верификации 2D-метода.

Защемлённый край (clamped):
    w(r) = q (a² − r²)² / (64 D),     w_max = q a⁴ / (64 D)

Шарнирное опирание (simply supported):
    w(r) = q/(64 D) (a² − r²) [ (5+ν)/(1+ν) a² − r² ]
    w_max = (5+ν)/(1+ν) · q a⁴ / (64 D)
"""

from __future__ import annotations

import numpy as np


def clamped_uniform(r, a: float, q: float, D: float):
    """Прогиб защемлённой круглой пластины (равномерная нагрузка)."""
    r = np.asarray(r, float)
    return q * (a**2 - r**2) ** 2 / (64.0 * D)


def clamped_uniform_wmax(a: float, q: float, D: float) -> float:
    """Максимальный прогиб (в центре) защемлённой круглой пластины."""
    return q * a**4 / (64.0 * D)


def simply_supported_uniform(r, a: float, q: float, D: float, nu: float):
    """Прогиб шарнирно опёртой круглой пластины (равномерная нагрузка)."""
    r = np.asarray(r, float)
    k = (5.0 + nu) / (1.0 + nu)
    return q / (64.0 * D) * (a**2 - r**2) * (k * a**2 - r**2)


def simply_supported_uniform_wmax(a: float, q: float, D: float, nu: float) -> float:
    """Максимальный прогиб (в центре) шарнирно опёртой круглой пластины."""
    return (5.0 + nu) / (1.0 + nu) * q * a**4 / (64.0 * D)
