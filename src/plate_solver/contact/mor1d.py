"""Метод обобщённой реакции (МОР) для 1D контактной задачи (балка-полоса).

Постановка: шарнирно-опёртая балка длиной L под равномерной нагрузкой q₀
опирается (односторонний контакт) на жёсткое основание высотой gap,
расположенное на участке [foundation_start, L].

Итерационная схема МОР (Михайловский–Тарасов):

    1. w = ∫ G(x,ξ) (q₀ − r(ξ)) dξ · L⁴/D   — прогиб при текущей реакции r
    2. r ← max(0,  r + β (w − gap))             — обновление реакции
    3. повторять до сходимости.

Параметр β — «жёсткость» итерации; при малом β сходимость гарантирована,
но медленнее. Рекомендуется β ∈ (0, 0.1] для задач настоящего класса.

Верификация 1D-решения: ``examples/contact_strip.py`` сравнивает результат
с точным решением Maple, хранящимся в ``analytic.strip_contact``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..solver.green1d import green_matrix


@dataclass
class ContactStrip1D:
    """Параметры 1D контактной задачи балки-полосы.

    Attributes
    ----------
    E : модуль Юнга.
    h : толщина балки/пластины-полосы.
    nu : коэффициент Пуассона.
    L : длина балки (размерная).
    q0 : интенсивность равномерной поперечной нагрузки.
    gap : зазор между балкой и основанием (Delta).
    foundation_start : координата левого края жёсткого основания.
    n : число равномерных разбиений по длине.
    beta : параметр итерации МОР.
    max_iter : максимальное число итераций.
    """

    E: float = 2.1e6
    h: float = 1.0
    nu: float = 0.3
    L: float = 100.0
    q0: float = 4.0
    gap: float = 1.0
    foundation_start: float = 45.0
    n: int = 100
    beta: float = 0.01
    max_iter: int = 50_000

    @property
    def D(self) -> float:
        """Цилиндрическая жёсткость D = E h³ / [12(1−ν²)]."""
        return self.E * self.h ** 3 / (12.0 * (1.0 - self.nu ** 2))


def solve_mor_1d(
    problem: ContactStrip1D,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Решить 1D контактную задачу методом обобщённой реакции.

    Parameters
    ----------
    problem : параметры задачи.

    Returns
    -------
    x_dim : координаты узлов (размерные), длина n+1.
    w : прогиб в узлах.
    r : контактная реакция в узлах.
    """
    p = problem
    tau = 1.0 / p.n
    scale = p.L ** 4 / p.D

    G = green_matrix(p.n)               # (n+1, n+1)
    x_norm = np.linspace(0.0, 1.0, p.n + 1)
    x_dim = p.L * x_norm

    r = np.zeros(p.n + 1)
    w = np.zeros(p.n + 1)
    contact_mask = x_dim > p.foundation_start

    for _ in range(p.max_iter):
        load = p.q0 - r                          # (n+1,)
        w = G[:, 1:] @ load[1:] * tau * scale   # суммирование по внутренним узлам
        r[contact_mask] = np.maximum(
            0.0,
            r[contact_mask] + p.beta * (w[contact_mask] - p.gap),
        )

    return x_dim, w, r
