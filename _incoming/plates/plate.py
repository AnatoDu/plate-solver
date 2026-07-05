r"""plate.py — изгиб пластины через расщепление бигармоники на две Пуассона.

Постановка (классика, шарнир в упрощении «мягкого шарнира»):
    D·Δ²w = q̃   в Ω,    w = 0 и M = 0 на ∂Ω,    q̃ = q0 − r.

Ввод ``M = −D·Δw`` даёт две задачи Дирихле с ОДНИМ оператором −Δ (NOTES.md §0):

    (P1)   −ΔM = q̃        в Ω,   M = 0 на ∂Ω
    (P2)   −Δw = M / D     в Ω,   w = 0 на ∂Ω

Обе решаются ОДНИМ :class:`~plates.poisson.PoissonSolver` (матрица ``A`` и её
факторизация общие) — меняется только правая часть.

Алгоритм :meth:`PlateBending.solve`:
    1. cM = solve(q̃)                       # (P1) → коэффициенты поля M
    2. M в узлах квадратуры = evaluate(cM)  # «источник истины» для (P2)
    3. cw = solve(M / D)                    # (P2) → коэффициенты прогиба w

Контроль (NOTES.md §0): на круге при q0>0 ⇒ w>0 всюду, максимум в центре.
"""

from __future__ import annotations

import numpy as np

from .basis import ChebyshevBasis
from .poisson import PoissonSolver
from .quadrature import interior_nodes


class PlateBending:
    """Изгиб пластины расщеплением (P1)+(P2) на одном решателе Пуассона."""

    def __init__(self, domain, basis, quad, cfg):
        self.domain = domain
        self.basis = basis
        self.quad = quad
        self.cfg = cfg
        self.D = float(cfg.D)
        self.poisson = PoissonSolver(domain, basis, quad)

    @classmethod
    def from_config(cls, domain, cfg) -> PlateBending:
        """Собрать решатель изгиба по конфигу: базис степени ``cfg.p``, квадратура ``cfg.Q``."""
        basis = ChebyshevBasis(cfg.p, domain.bbox)
        quad = interior_nodes(domain, cfg.Q)
        return cls(domain, basis, quad, cfg)

    def solve(self, qtilde_values):
        r"""Решить изгиб для правой части ``q̃`` в узлах квадратуры → ``(cM, cw)``.

        ``M`` вычисляется в узлах квадратуры (там же, где собирается нагрузка
        для (P2)). Возвращает коэффициенты полей ``M`` и ``w``.
        """
        cM = self.poisson.solve(qtilde_values)                       # (P1): −ΔM = q̃
        M_nodes = self.poisson.evaluate(cM, self.quad.x, self.quad.y)  # M в узлах
        cw = self.poisson.solve(M_nodes / self.D)                    # (P2): −Δw = M/D
        return cM, cw

    def solve_uniform(self, q: float | None = None):
        """Изгиб под равномерной нагрузкой ``q`` (по умолчанию ``cfg.q0``), без контакта."""
        q = self.cfg.q0 if q is None else q
        qtilde = np.full(self.quad.x.size, float(q))
        return self.solve(qtilde)

    def deflection(self, cw, X, Y) -> np.ndarray:
        """Прогиб ``w = ω·Σ cw_k T_k`` в точках (X, Y)."""
        return self.poisson.evaluate(cw, X, Y)

    def moment(self, cM, X, Y) -> np.ndarray:
        """Поле «суммарного момента» ``M = ω·Σ cM_k T_k`` в точках (X, Y)."""
        return self.poisson.evaluate(cM, X, Y)


__all__ = ["PlateBending"]
