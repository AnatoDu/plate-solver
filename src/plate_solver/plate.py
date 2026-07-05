r"""plate.py — изгиб пластины через расщепление бигармоники на две Пуассона.

Постановка (классика, шарнир в упрощении «мягкого шарнира»):
    D·Δ²w = q̃   в Ω,    w = 0 и M = 0 на ∂Ω,    q̃ = q0 − r.

Ввод ``M = −D·Δw`` даёт две задачи Дирихле с ОДНИМ оператором −Δ (NOTES.md §0):

    (P1)   −ΔM = q̃        в Ω,   M = 0 на ∂Ω
    (P2)   −Δw = M / D     в Ω,   w = 0 на ∂Ω

Обе решаются ОДНИМ :class:`~plate_solver.poisson.PoissonSolver` (матрица ``A`` и её
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
        M_nodes = self.poisson.evaluate_at_quad(cM)                  # M в узлах (кэш, GEMV)
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

    def load_vector(self, f_values) -> np.ndarray:
        """Вектор нагрузки (P1) ``b[k] = ∫ f ψ_k`` (внешние правые части, A4)."""
        return self.poisson.load_vector(f_values)

    def solve_from_b(self, b1):
        r"""Решить изгиб по ГОТОВОМУ вектору нагрузки (P1) (A4: пара пластин).

        Нужен, когда нагрузка содержит вклад реакции, интегрируемый по ЧУЖОЙ
        квадратуре (∫ r ψ_k по узлам первой пластины) — интерполяции r нет.
        """
        cM = self.poisson.solve_b(b1)
        M_nodes = self.poisson.evaluate_at_quad(cM)
        cw = self.poisson.solve(M_nodes / self.D)
        return cM, cw

    def structure_at(self, X, Y) -> np.ndarray:
        """Матрица структуры ψ_k = ω·T_k в произвольных точках: (N, len(X)) (A4)."""
        Phi = self.basis.values(X, Y)
        return self.domain.omega(X, Y) * Phi

    # -- протокол контакта (общий с ClampedPlate; фаза 3, A3.3) ------------ #
    def w_at_quad(self, state) -> np.ndarray:
        """Прогиб в узлах квадратуры; state = (cM, cw) из :meth:`solve`."""
        return self.poisson.evaluate_at_quad(state[1])

    def lap_w_at_quad(self, state) -> np.ndarray:
        r"""Кривизна ``Δw = −M/D`` в узлах квадратуры (из (P1), без
        численного дифференцирования); state = (cM, cw)."""
        return -self.poisson.evaluate_at_quad(state[0]) / self.D

    @staticmethod
    def coeffs_w(state) -> np.ndarray:
        """Коэффициенты прогиба из состояния решения (state = (cM, cw))."""
        return state[1]


__all__ = ["PlateBending"]
