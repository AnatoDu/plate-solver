r"""quadrature.py — тензорная квадратура Гаусса–Лежандра + маска «внутри Ω».

Интегралы по Ω берём по описанному прямоугольнику (bbox) тензорным правилом
Гаусса–Лежандра ``Q×Q``, оставляя только узлы внутри области по условию
``ω(x, y) > eps`` (маскирование). Интеграл ``∫_Ω f dΩ ≈ Σ_{внутри} W·f``.

Якобиан отображения [−1,1]² → bbox зашит в веса: на [−1,1] вес Гаусса даёт
``Σ w = 2``; после масштабирования ``Σ W = (x_max−x_min)(y_max−y_min)`` (площадь
bbox), а сумма ``W`` по маске круга → площадь круга ``π a²`` (тест-ворота).

Узлы квадратуры — «источник истины» и для контактной реакции ``r`` (NOTES.md §3).

Замечание (NOTES.md §4): маскирование даёт СТУПЕНЧАТУЮ границу интегрирования —
это основной источник погрешности у границы. Сходимость по Q умеренная
(~O(1/Q): погрешность ∝ периметр × шаг). Для высокой точности нужны адаптивные/
погранично-согласованные квадратуры (на будущее; соответствует замечанию доклада).
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from .geometry import Domain

BBox = tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)

# Узлы квадратуры внутри Ω: координаты x, y и веса w (1D-массивы равной длины).
# namedtuple ⇒ передаётся в сборщик/решатель одним аргументом ``quad`` и при этом
# распаковывается как обычный кортеж (x, y, w).
QuadNodes = namedtuple("QuadNodes", "x y w")


def gauss_legendre_grid(Q: int, bbox: BBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Тензорные узлы и веса Гаусса–Лежандра ``Q×Q`` на bbox (БЕЗ маски).

    Returns
    -------
    X, Y, W : 1D-массивы длины Q² — координаты узлов и веса (с якобианом
    отображения [−1,1]² → bbox), тензорно развёрнутые.
    """
    if Q < 1:
        raise ValueError("Q должно быть ≥ 1.")
    xmin, xmax, ymin, ymax = map(float, bbox)
    nodes, weights = np.polynomial.legendre.leggauss(Q)   # узлы/веса на [−1, 1]
    ax, bx = (xmax - xmin) / 2.0, (xmax + xmin) / 2.0
    ay, by = (ymax - ymin) / 2.0, (ymax + ymin) / 2.0
    xs, wx = ax * nodes + bx, ax * weights
    ys, wy = ay * nodes + by, ay * weights
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    WX, WY = np.meshgrid(wx, wy, indexing="ij")
    return X.ravel(), Y.ravel(), (WX * WY).ravel()


def interior_mask(domain: Domain, X, Y, eps: float = 0.0) -> np.ndarray:
    """Булева маска узлов внутри области: ``domain.omega(X, Y) > eps``."""
    return domain.omega(X, Y) > eps


def interior_nodes(domain: Domain, Q: int, eps: float = 0.0) -> QuadNodes:
    """Узлы и веса квадратуры, ОТФИЛЬТРОВАННЫЕ маской ``ω > eps`` (вход для сборки).

    Удобная связка ``gauss_legendre_grid`` + ``interior_mask``: возвращает
    :class:`QuadNodes` ``(x, y, w)`` только для точек внутри Ω. Это и есть
    аргумент ``quad`` для assembler/PoissonSolver.
    """
    X, Y, W = gauss_legendre_grid(Q, domain.bbox)
    m = interior_mask(domain, X, Y, eps)
    return QuadNodes(X[m], Y[m], W[m])


__all__ = ["BBox", "QuadNodes", "gauss_legendre_grid", "interior_mask", "interior_nodes"]
