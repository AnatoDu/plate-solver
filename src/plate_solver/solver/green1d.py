r"""Функция Грина шарнирно-опёртой балки (оба конца, классическая теория).

Для нормированной задачи (длина 1) изгиба балки под нагрузкой q(ξ):

.. math::

    D\,w'''' = q, \quad w(0) = w''(0) = w(1) = w''(1) = 0,

функция Грина G(x, ξ) такова, что

.. math::

    w(x) = \frac{L^4}{D} \int_0^1 G(x,\xi)\,q(\xi)\,d\xi.

Явная формула (Тимошенко, «Сопротивление материалов», т. 2):

.. math::

    G(x,\xi) = \begin{cases}
        \dfrac{x\,(1-\xi)}{6}\bigl(2\xi - \xi^2 - x^2\bigr), & x \le \xi,\\[6pt]
        \dfrac{\xi\,(1-x)}{6}\bigl(2x - x^2 - \xi^2\bigr),   & x >  \xi.
    \end{cases}

Верификация: свободная балка (r≡0) при q₀=1 даёт w_max = 5/(384·D) в точке x=0.5
(Тимошенко). Сравнение МОР с точным решением Maple — ``analytic.strip_contact``.
"""

from __future__ import annotations

import numpy as np


def green_simply_supported(x, xi: float) -> np.ndarray:
    r"""G(x, ξ) — функция Грина шарнирно-опёртой балки в точке (x, ξ).

    Parameters
    ----------
    x  : координата наблюдения (скаляр или массив), 0 ≤ x ≤ 1.
    xi : точка приложения единичной нагрузки, 0 ≤ ξ ≤ 1.
    """
    x = np.asarray(x, float)
    xi = float(xi)
    return np.where(
        x <= xi,
        x * (1.0 - xi) * (2.0 * xi - xi ** 2 - x ** 2) / 6.0,
        xi * (1.0 - x) * (2.0 * x - x ** 2 - xi ** 2) / 6.0,
    )


def green_matrix(n: int) -> np.ndarray:
    r"""Матрица G[i, j] = G(i/n, j/n) для равномерной сетки из n+1 узлов.

    Вычисляется векторно за O(n²) без циклов Python.

    Parameters
    ----------
    n : число разбиений отрезка [0, 1].

    Returns
    -------
    G : массив формы (n+1, n+1).
    """
    t = np.linspace(0.0, 1.0, n + 1)
    x = t[:, np.newaxis]   # (n+1, 1)
    xi = t[np.newaxis, :]  # (1, n+1)
    return np.where(
        x <= xi,
        x * (1.0 - xi) * (2.0 * xi - xi ** 2 - x ** 2) / 6.0,
        xi * (1.0 - x) * (2.0 * x - x ** 2 - xi ** 2) / 6.0,
    )
