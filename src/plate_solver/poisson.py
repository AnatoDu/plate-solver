r"""poisson.py — решатель одной задачи Дирихле −Δv = f методом Ритца.

Решает ``−Δv = f`` в Ω, ``v = 0`` на ``∂Ω``, аппроксимация ``v = ω·Φ``,
``Φ = Σ_k c_k T_k``. Матрица Ритца ``A`` (assembler.py) собирается и факторизуется
(Холецкий) ОДИН раз при создании решателя; затем каждое решение для новой правой
части ``f`` — это лишь сборка ``b`` и две треугольные подстановки.

Зачем факторизация один раз: ``A`` общая для (P1), (P2) и всех итераций МОР
(NOTES.md §§1, 5) ⇒ контактный цикл получается дешёвым.

Кэш матриц структуры (P2.1). При ``cache_fields=True`` в конструкторе один раз
сохраняются матрицы N×M в узлах квадратуры, потребляемые итерацией МОР:
``psiW = ψ·diag(w)`` (ψ = ω·T) и ``Φ`` с ``ω`` по отдельности. Тогда на каждой
итерации:

* сборка нагрузки — один GEMV: ``b = psiW @ f`` (вместо пересчёта базиса);
* значения поля в узлах квадратуры — :meth:`evaluate_at_quad`, один GEMV:
  ``v = ψᵀc = ω·(Φᵀc)`` (реализовано как ``ω·(Φᵀc)``, чтобы порядок округления
  совпадал с :meth:`evaluate` БИТ-В-БИТ — golden не должен меняться).

Матрицы ``psi``, ``psi_x``, ``psi_y`` НЕ хранятся: после сборки ``A`` у них нет
потребителей (ψ = ω·Φ восстановим из кэша), а их хранение утроило бы память.
Память кэша ~2·N·M·8 байт: для L-формы (p=10, Q=120: N·M≈1.3e6) это ~21 МБ.
Для «больших» квадратур (N·M > ``CACHE_NM_MAX`` = 5e7, т.е. кэш > ~0.8 ГБ;
напр. круг Q=1024, p=10: N·M ≈ 1e8) кэш по умолчанию ВЫКЛЮЧЕН — поведение
и арифметика прежние.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .assembler import (
    assemble_load,
    assemble_stiffness,
    assemble_stiffness_from_fields,
    structure_fields,
)

# Порог включения кэша: N·M ≤ CACHE_NM_MAX (≈ 0.4 ГБ на одну матрицу float64).
CACHE_NM_MAX = 50_000_000


class PoissonSolver:
    """Решатель ``−Δv = f`` с предвычисленной факторизацией ``A``.

    Parameters
    ----------
    domain : область (Domain) — даёт ω и ∇ω.
    basis : тензорный базис Чебышёва (ChebyshevBasis).
    quad : узлы и веса квадратуры внутри Ω (QuadNodes).
    cache_fields : кэшировать ли матрицы структуры N×M (см. докстринг модуля);
        ``None`` (по умолчанию) — автоматически: True при N·M ≤ CACHE_NM_MAX.
    """

    def __init__(self, domain, basis, quad, cache_fields: bool | None = None):
        self.domain = domain
        self.basis = basis
        self.quad = quad
        if cache_fields is None:
            cache_fields = basis.N * quad.x.size <= CACHE_NM_MAX
        self.cache_fields = bool(cache_fields)
        if self.cache_fields:
            psi, psi_x, psi_y, W = structure_fields(domain, basis, quad)
            self.A = assemble_stiffness_from_fields(psi_x, psi_y, W)
            self.psiW = psi * W                                  # для b = psiW @ f
            del psi, psi_x, psi_y                                # без потребителей — не держим
            # Φ и ω отдельно — чтобы evaluate_at_quad повторял evaluate бит-в-бит.
            self._phi_quad = basis.values(quad.x, quad.y)        # (N, M)
            self._om_quad = domain.omega(quad.x, quad.y)         # (M,)
        else:
            self.psiW = self._phi_quad = self._om_quad = None
            self.A = assemble_stiffness(domain, basis, quad)
        self.chol = sla.cho_factor(self.A)          # факторизация ОДИН раз

    @property
    def cond(self) -> float:
        """Число обусловленности cond(A) — диагностика устойчивости (NOTES.md §2)."""
        return float(np.linalg.cond(self.A))

    def solve(self, f_values) -> np.ndarray:
        """Коэффициенты ``c`` разложения Φ: решить ``A c = b(f)``.

        ``f_values`` — значения правой части в узлах квадратуры (self.quad).
        При включённом кэше сборка ``b = psiW @ f`` — один GEMV.
        """
        if self.cache_fields:
            b = self.psiW @ np.asarray(f_values, dtype=float)
        else:
            b = assemble_load(self.domain, self.basis, self.quad, f_values)
        return sla.cho_solve(self.chol, b)

    def evaluate(self, c, X, Y) -> np.ndarray:
        """Значения ``v = ω·Σ_k c_k T_k`` в точках (X, Y)."""
        Phi = self.basis.values(X, Y)               # (N, *shape)
        v = np.tensordot(np.asarray(c, float), Phi, axes=(0, 0))
        return self.domain.omega(X, Y) * v

    def evaluate_at_quad(self, c) -> np.ndarray:
        r"""Значения ``v = ψᵀc`` в узлах квадратуры через кэш (один GEMV).

        Тождественно ``evaluate(c, quad.x, quad.y)`` (бит-в-бит): считается как
        ``ω·(Φᵀc)`` по кэшированным Φ и ω. Без кэша — просто вызывает evaluate.
        Главный потребитель — итерация МОР (contact.py) и (P2) в plate.solve.
        """
        if not self.cache_fields:
            return self.evaluate(c, self.quad.x, self.quad.y)
        v = np.tensordot(np.asarray(c, float), self._phi_quad, axes=(0, 0))
        return self._om_quad * v

    def solve_field(self, f_values, X, Y) -> np.ndarray:
        """Удобный фасад: решить и сразу вычислить поле в точках (X, Y)."""
        return self.evaluate(self.solve(f_values), X, Y)


__all__ = ["PoissonSolver", "CACHE_NM_MAX"]
