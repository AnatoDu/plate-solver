r"""stamp_ritz.py — задача о ШТАМПЕ методом Ритца (1D), вариант B лестницы.

Та же 1D-контактная задача, что и в ``plate_solver.stamp`` (функция Грина), но изгиб на
КАЖДОЙ итерации МОР решается методом РИТЦА (как ступень 1 лестницы), а не свёрткой
с функцией Грина. Цель — тройная сходимость Грин = Ритц = аналитика Maple: штамп
становится первой ступенью Ритца.

Краевые условия — ТЕ ЖЕ, что у функции Грина в ``fix_base2.py``: шарнир при x=0
(``w=0``, ``w''=0``) и скользящая (симметрийная) заделка при x=L (``w'=0``,
``w'''=0``). В энергии ``(D/2)∫(w'')²`` существенны условия на ``w`` и ``w'``:
``w(0)=0`` и ``w'(L)=0``; естественными (из минимума) выходят ``w''(0)=0`` и
``w'''(L)=0`` — ровно пара ГУ функции Грина.

Структура. Простой множитель ``ω^p`` НЕ годится: ``w(0)=0`` (значение) и ``w'(L)=0``
(производная) — смешанные ГУ. Берём БАЗИС, КАЖДАЯ функция которого удовлетворяет
обоим существенным условиям: ``ψ_j = Σ_i B_{ij} T_i(ξ)``, где столбцы ``B`` —
нуль-пространство двух линейных связей
    Σ_i c_i T_i(−1) = Σ_i c_i(−1)^i = 0     (w(0)=0),
    Σ_i c_i T_i'(1)·(2/L) = (2/L) Σ_i c_i i² = 0   (w'(L)=0).
Это вариант (б) из ТЗ — расширение базиса функциями, удовлетворяющими ГУ.

Изгиб линеен ⇒ Ритц задаёт ДИСКРЕТНЫЙ оператор отклика ``M_ritz`` (как функция
Грина — матрица): ``w_узлы = M_ritz·(q − r)``. МОР поверх — тот же, что в
``plate_solver.stamp`` (нодальная реакция ``r := max(0, r + β(w − Δ))`` под штампом).
"""

from __future__ import annotations

import numpy as np
import numpy.polynomial.chebyshev as _cheb
import scipy.linalg as sla

from .mor1d import ContactStrip1D
from .stamp import StampResult


# --------------------------------------------------------------------------- #
#  Структурный базис под пару ГУ «шарнир(0) + скользящая заделка(L)»
# --------------------------------------------------------------------------- #
def _constraint_basis(p: int) -> np.ndarray:
    """Нуль-пространство связей ``Σc_i(−1)^i=0`` и ``Σc_i i²=0``: матрица (p+1, p−1)."""
    i = np.arange(p + 1, dtype=float)
    C = np.vstack([(-1.0) ** i, i**2])              # (2, p+1): w(0)=0 и w'(L)=0
    return sla.null_space(C)                         # (p+1, p−1)


def build_ritz_beam_operator(L: float, D: float, n: int, p: int, n_quad: int = 600):
    r"""Дискретный оператор Ритца ``M`` (n+1, n+1): ``w_узлы = M·(q−r)_узлы``.

    Стержень-полоса ``D w⁗ = нагрузка`` с ГУ шарнир(0)+скользящая заделка(L).
    Жёсткость ``S_jk = ∫ ψ_j'' ψ_k'' dx`` (Гаусс), нагрузка интегрируется по сетке
    из ``n+1`` узлов (трапеции) — той же, что у функции Грина (прямое сравнение).
    """
    B = _constraint_basis(p)                         # (p+1, p−1)
    # --- жёсткость по Гауссу ---
    t, wt = np.polynomial.legendre.leggauss(n_quad)
    xq = 0.5 * L * (t + 1.0)
    Wq = 0.5 * L * wt
    xiq = (2.0 * xq - L) / L
    eye = np.eye(p + 1)
    T2 = np.empty((p + 1, xq.size))
    for k in range(p + 1):
        T2[k] = _cheb.chebval(xiq, _cheb.chebder(eye[k], 2)) if p >= 2 else 0.0
    T2 *= (2.0 / L) ** 2                              # T_i''(x)
    Psi2 = B.T @ T2                                   # (p−1, nq)
    S = (Psi2 * Wq) @ Psi2.T
    S = 0.5 * (S + S.T)
    # --- значения базиса на сетке + веса нагрузки (трапеции) ---
    xg = np.linspace(0.0, L, n + 1)
    xig = (2.0 * xg - L) / L
    T0g = np.moveaxis(_cheb.chebvander(xig, p), -1, 0)   # (p+1, n+1)
    Psi0g = B.T @ T0g                                    # (p−1, n+1)
    Wgrid = np.full(n + 1, L / n)
    Wgrid[0] *= 0.5
    Wgrid[-1] *= 0.5
    X = np.linalg.solve(D * S, Psi0g * Wgrid)            # (p−1, n+1)
    M = Psi0g.T @ X                                       # (n+1, n+1)
    return M, float(np.linalg.cond(S))


# --------------------------------------------------------------------------- #
#  МОР поверх оператора отклика (тот же алгоритм, что в plate_solver.stamp)
# --------------------------------------------------------------------------- #
def solve_stamp_ritz(
    problem: ContactStrip1D | None = None,
    *,
    p: int = 16,
    max_iter: int = 2_000_000,
    tol_dw: float = 1e-9,
    record_every: int = 2000,
) -> tuple[StampResult, float]:
    """Решить штамп методом Ритца + МОР. -> (StampResult, cond(S))."""
    pr = problem if problem is not None else ContactStrip1D()
    M, condS = build_ritz_beam_operator(pr.L, pr.D, pr.n, p)
    x = pr.L * np.linspace(0.0, 1.0, pr.n + 1)
    mask = x > pr.foundation_start

    r = np.zeros(pr.n + 1)
    w = np.zeros(pr.n + 1)
    dr_first = 0.0
    res_iters: list[int] = []
    res_hist: list[float] = []
    converged = False
    k = 0
    for k in range(1, max_iter + 1):
        w_new = M @ (pr.q0 - r)
        upd = r.copy()
        upd[mask] = r[mask] + pr.beta * (w_new[mask] - pr.gap)
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

    res = StampResult(
        x=x, w=w, r=r, iters=k, converged=converged,
        dr_first=dr_first, dr_last=dr,
        res_iters=np.array(res_iters), res_hist=np.array(res_hist),
    )
    return res, condS


__all__ = ["build_ritz_beam_operator", "solve_stamp_ritz"]
