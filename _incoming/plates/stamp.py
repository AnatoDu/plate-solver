r"""stamp.py — 1D контактная задача со ШТАМПОМ фиксированной ширины (метод обобщённой реакции).

Полоса-пластина на ``[0, L]``; снизу — ЖЁСТКИЙ штамп (основание) фиксированной
ширины на участке ``[m, L]`` с зазором ``Δ``. Контакт ОДНОСТОРОННИЙ: реакция
``r ≥ 0`` включается только под штампом и только там, где прогиб достигает зазора
(``w > Δ``). Это 1D-ЗАДЕЛ, обобщаемый работой на 2D (та же схема МОР).

Физика и числа взяты БЕЗ ИЗМЕНЕНИЙ из ``lab/Задачи Python/fix_base2.py`` и
переиспользуют решатель ``plate_solver.contact.mor1d`` (функция Грина балочного
оператора ``green1d``): шарнир слева, плоскость симметрии справа,

    G(t,ξ) = (1/6)(t−ξ)³H(t−ξ) + (ξ − ξ²/2)·t − t³/6,
    w(x) = (L⁴/D) ∫ G(x,ξ)(q0 − r(ξ)) dξ.

Итерация МОР (как в fix_base2.py):

    w ← (L⁴/D)·G·(q0 − r) · τ
    r ← max(0, r + β(w − Δ))   только при  L·x > m

Здесь добавлено лишь ОТСЛЕЖИВАНИЕ невязки ‖Δr‖ и критерий остановки по сходимости
(в fix_base2.py было фиксированные 50000 итераций — НЕдосходимость: пик реакции
у кромки штампа — особенность давления под жёстким штампом — досходится медленно).
Сам решатель ``plate_solver`` не изменяется.

Эталон — аналитическое решение из Maple (``data/stamp_maple_xy.txt``); в исходных
данных значение с индексом 45 битое (стоит «25») и чинится усреднением соседей
``ya[45]=(ya[44]+ya[46])/2`` — обработка сохранена.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from plate_solver.contact.mor1d import ContactStrip1D
from plate_solver.solver.green1d import green_matrix

# Эталон Maple рядом с пакетом (копия исходного xy.txt).
_DEFAULT_MAPLE = os.path.join(os.path.dirname(__file__), "..", "data", "stamp_maple_xy.txt")


# --------------------------------------------------------------------------- #
#  Решение МОР с диагностикой сходимости
# --------------------------------------------------------------------------- #
@dataclass
class StampResult:
    """Результат 1D-задачи о штампе: поля + диагностика сходимости МОР."""

    x: np.ndarray                     # координаты узлов (размерные, см), 0..L
    w: np.ndarray                     # прогиб в узлах
    r: np.ndarray                     # контактная реакция в узлах
    iters: int                        # число итераций до остановки
    converged: bool                   # достигнут ли критерий сходимости
    dr_first: float                   # ‖Δr‖ на первой итерации
    dr_last: float                    # ‖Δr‖ на последней итерации
    res_iters: np.ndarray = field(default_factory=lambda: np.empty(0))   # отметки итераций
    res_hist: np.ndarray = field(default_factory=lambda: np.empty(0))    # ‖Δr‖ на них

    # -- производные характеристики ------------------------------------- #
    @property
    def w_max(self) -> float:
        return float(self.w.max())

    @property
    def x_wmax(self) -> float:
        return float(self.x[int(np.argmax(self.w))])

    @property
    def contact(self) -> np.ndarray:
        """Булева маска зоны контакта (r > 0)."""
        return self.r > 0.0

    @property
    def contact_span(self) -> tuple[float, float]:
        """Границы зоны контакта (x_start, x_end) в физических координатах."""
        xc = self.x[self.contact]
        return (float(xc.min()), float(xc.max())) if xc.size else (float("nan"), float("nan"))

    @property
    def n_contact(self) -> int:
        return int(self.contact.sum())

    @property
    def r_max(self) -> float:
        return float(self.r.max())

    @property
    def x_rmax(self) -> float:
        return float(self.x[int(np.argmax(self.r))])


def solve_stamp(
    problem: ContactStrip1D | None = None,
    *,
    max_iter: int = 2_000_000,
    tol_dw: float = 1e-9,
    record_every: int = 2000,
) -> StampResult:
    r"""Решить 1D-задачу о штампе методом обобщённой реакции с диагностикой.

    Алгоритм ТОЖДЕСТВЕН ``fix_base2.py`` / ``plate_solver.contact.mor1d`` (та же
    функция Грина, та же итерация); добавлены лишь критерий сходимости по
    ``max|Δw| < tol_dw`` и история невязки ‖Δr‖.

    Parameters
    ----------
    problem : параметры (по умолчанию — точно как в fix_base2.py).
    max_iter : предел итераций (по умолчанию с запасом до сходимости).
    tol_dw : критерий остановки по ``max|w_{k+1} − w_k|``.
    record_every : шаг записи истории ‖Δr‖.
    """
    p = problem if problem is not None else ContactStrip1D()
    n, tau = p.n, 1.0 / p.n
    scale = p.L**4 / p.D
    Gint = green_matrix(n)[:, 1:]                 # суммирование по внутренним узлам j=1..n
    x = p.L * np.linspace(0.0, 1.0, n + 1)
    mask = x > p.foundation_start                 # под штампом: L·x > m

    r = np.zeros(n + 1)
    w = np.zeros(n + 1)
    dr_first = 0.0
    res_iters: list[int] = []
    res_hist: list[float] = []
    converged = False
    k = 0
    for k in range(1, max_iter + 1):
        w_new = Gint @ (p.q0 - r)[1:] * tau * scale
        upd = r.copy()
        upd[mask] = r[mask] + p.beta * (w_new[mask] - p.gap)
        r_new = np.maximum(upd, 0.0)
        dr = float(np.linalg.norm(r_new - r))
        dw = float(np.max(np.abs(w_new - w)))
        if k == 1:
            dr_first = dr
        if k == 1 or k % record_every == 0:
            res_iters.append(k)
            res_hist.append(dr)
        r, w = r_new, w_new
        if dw < tol_dw:
            converged = True
            break

    return StampResult(
        x=x, w=w, r=r, iters=k, converged=converged,
        dr_first=dr_first, dr_last=dr,
        res_iters=np.array(res_iters), res_hist=np.array(res_hist),
    )


# --------------------------------------------------------------------------- #
#  Эталон Maple и согласие
# --------------------------------------------------------------------------- #
def load_maple_reference(path: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    r"""Загрузить аналитический эталон Maple (x целые 0..100; w).

    Сохраняет обработку битого значения с индексом 45:
    ``ya[45] = (ya[44] + ya[46]) / 2`` (в исходнике там стоит «25»).
    """
    path = path or _DEFAULT_MAPLE
    with open(path) as f:
        lines = f.readlines()
    xa = np.array([int(v) for v in lines[0].split(",")], dtype=float)
    ya = np.array([float(v) for v in lines[1].split(",")], dtype=float)
    ya[45] = (ya[44] + ya[46]) / 2.0
    return xa, ya


def maple_agreement(
    x: np.ndarray, w: np.ndarray, xa: np.ndarray, ya: np.ndarray
) -> tuple[float, float]:
    r"""Согласие численного ``w`` с эталоном Maple на целочисленных узлах x=0..100.

    Узлы МОР (n=100) совпадают с целочисленными узлами Maple, поэтому интерполяция
    тривиальна (при иной сетке — линейная интерполяция на xa).

    Returns
    -------
    (max|Δw|, отн. L²-отклонение в ПРОЦЕНТАХ).
    """
    wn = w if np.array_equal(x, xa) else np.interp(xa, x, w)
    max_abs = float(np.max(np.abs(wn - ya)))
    rel_l2 = 100.0 * float(np.sqrt(np.sum((wn - ya) ** 2)) / np.sqrt(np.sum(ya**2)))
    return max_abs, rel_l2


__all__ = ["StampResult", "solve_stamp", "load_maple_reference", "maple_agreement"]
