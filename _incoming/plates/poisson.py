r"""poisson.py — решатель одной задачи Дирихле −Δv = f методом Ритца.

Решает ``−Δv = f`` в Ω, ``v = 0`` на ``∂Ω``, аппроксимация ``v = ω·Φ``,
``Φ = Σ_k c_k T_k``. Матрица Ритца ``A`` (assembler.py) собирается и факторизуется
(Холецкий) ОДИН раз при создании решателя; затем каждое решение для новой правой
части ``f`` — это лишь сборка ``b`` и две треугольные подстановки.

Зачем факторизация один раз: ``A`` общая для (P1), (P2) и всех итераций МОР
(NOTES.md §§1, 5) ⇒ контактный цикл получается дешёвым.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .assembler import assemble_load, assemble_stiffness


class PoissonSolver:
    """Решатель ``−Δv = f`` с предвычисленной факторизацией ``A``.

    Parameters
    ----------
    domain : область (Domain) — даёт ω и ∇ω.
    basis : тензорный базис Чебышёва (ChebyshevBasis).
    quad : узлы и веса квадратуры внутри Ω (QuadNodes).
    """

    def __init__(self, domain, basis, quad):
        self.domain = domain
        self.basis = basis
        self.quad = quad
        self.A = assemble_stiffness(domain, basis, quad)
        self.chol = sla.cho_factor(self.A)          # факторизация ОДИН раз

    @property
    def cond(self) -> float:
        """Число обусловленности cond(A) — диагностика устойчивости (NOTES.md §2)."""
        return float(np.linalg.cond(self.A))

    def solve(self, f_values) -> np.ndarray:
        """Коэффициенты ``c`` разложения Φ: решить ``A c = b(f)``.

        ``f_values`` — значения правой части в узлах квадратуры (self.quad).
        """
        b = assemble_load(self.domain, self.basis, self.quad, f_values)
        return sla.cho_solve(self.chol, b)

    def evaluate(self, c, X, Y) -> np.ndarray:
        """Значения ``v = ω·Σ_k c_k T_k`` в точках (X, Y)."""
        Phi = self.basis.values(X, Y)               # (N, *shape)
        v = np.tensordot(np.asarray(c, float), Phi, axes=(0, 0))
        return self.domain.omega(X, Y) * v

    def solve_field(self, f_values, X, Y) -> np.ndarray:
        """Удобный фасад: решить и сразу вычислить поле в точках (X, Y)."""
        return self.evaluate(self.solve(f_values), X, Y)


__all__ = ["PoissonSolver"]
