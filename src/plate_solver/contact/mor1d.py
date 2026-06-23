"""Метод обобщённой реакции (МОР) для 1D контактной задачи (балка-полоса).

Постановка: балка-полоса длиной L под равномерной нагрузкой q₀, шарнирно
опёртая слева (x=0) и с плоскостью симметрии справа (x=L). Снизу — жёсткое
основание с зазором gap на участке [foundation_start, L]; контакт
односторонний (см. функцию Грина в ``solver.green1d``).

Итерационная схема МОР (Михайловский–Тарасов, ПММ 1993):

    1. w = ∫ G(x,ξ) (q₀ − r(ξ)) dξ · L⁴/D   — прогиб при текущей реакции r
    2. r ← max(0,  r + β (w − gap))             — обновление реакции
    3. повторять до сходимости.

Параметр β — «жёсткость» итерации. Сходимость линейная; устойчивость требует
β < β_кр (для эталонной задачи β_кр ≈ 0.025, при β ≥ 0.03 итерация расходится).
По умолчанию β = 0.01 — как в исходных программах научной школы.

Верификация: ``examples/contact_strip.py`` и ``tests/test_mor1d.py`` сравнивают
результат с точным решением Maple (``analytic.strip_contact``); расхождение —
O(1/n) (погрешность квадратуры на разрывной функции Грина).
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
    tol : критерий остановки по максимальному изменению прогиба между
        итерациями (max|wₖ₊₁ − wₖ| < tol). При tol=0 итерация идёт ровно
        max_iter шагов.
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
    max_iter: int = 500_000
    tol: float = 1e-9

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
        w_new = G[:, 1:] @ load[1:] * tau * scale   # суммирование по внутренним узлам
        r[contact_mask] = np.maximum(
            0.0,
            r[contact_mask] + p.beta * (w_new[contact_mask] - p.gap),
        )
        if p.tol > 0.0 and np.max(np.abs(w_new - w)) < p.tol:
            w = w_new
            break
        w = w_new

    return x_dim, w, r
