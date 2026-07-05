r"""assembler.py — сборка системы Ритца для оператора −Δ.

Структура ``v = ω·Φ``, ``Φ = Σ_k c_k T_k``; базисные функции структуры
``ψ_k = ω·T_k`` зануляются на ``∂Ω`` автоматически. Слабая форма ``−Δv = f``,
``v|_{∂Ω} = 0`` даёт

    A[k,l] = ∫_Ω ∇ψ_k · ∇ψ_l dΩ,        b[k] = ∫_Ω f · ψ_k dΩ,

где ``∇ψ_k = (∇ω)·T_k + ω·(∇T_k)`` — поэтому нужны и ``domain.grad_omega``, и
``basis.grads``. Интегралы берём по узлам квадратуры внутри Ω
(:class:`~plates.quadrature.QuadNodes`): ``∫_Ω g dΩ ≈ Σ_m w_m g(x_m, y_m)``.

Матричная запись (M узлов, N базисных функций):
    Ψ_x = (∇ω)_x·T + ω·T_x,   Ψ_y = (∇ω)_y·T + ω·T_y     (формы N×M)
    A = Ψ_x diag(w) Ψ_xᵀ + Ψ_y diag(w) Ψ_yᵀ,             b = (ω·T) diag(w) f

``A`` симметрична и положительно определена (билинейная форма ∫∇·∇ на ψ_k,
зануляющихся на ∂Ω) ⇒ годится для Холецкого (poisson.py).

ВАЖНО: ``A`` зависит только от геометрии и базиса ⇒ собирается ОДИН раз. ``b``
зависит от ``f`` ⇒ пересобирается на каждой итерации МОР (это дёшево).
"""

from __future__ import annotations

import numpy as np


def _structure_fields(domain, basis, quad):
    """Промежуточные матрицы N×M: ω·T и компоненты ∇ψ в узлах квадратуры."""
    X, Y, W = quad.x, quad.y, quad.w
    om = domain.omega(X, Y)                 # (M,)
    omx, omy = domain.grad_omega(X, Y)      # (M,), (M,)
    T = basis.values(X, Y)                  # (N, M)
    Tx, Ty = basis.grads(X, Y)              # (N, M), (N, M)
    psi = om * T                            # ψ_k = ω·T_k                 (N, M)
    psi_x = omx * T + om * Tx               # ∂ψ_k/∂x                     (N, M)
    psi_y = omy * T + om * Ty               # ∂ψ_k/∂y                     (N, M)
    return psi, psi_x, psi_y, W


def assemble_stiffness(domain, basis, quad) -> np.ndarray:
    r"""Матрица жёсткости ``A[k,l] = ∫_Ω ∇ψ_k·∇ψ_l dΩ`` (N×N, СИММ., полож. опр.).

    Собирается один раз: не зависит от правой части.
    """
    _, psi_x, psi_y, W = _structure_fields(domain, basis, quad)
    A = (psi_x * W) @ psi_x.T + (psi_y * W) @ psi_y.T
    return 0.5 * (A + A.T)                   # подавить несимметрию округления


def assemble_load(domain, basis, quad, f_values) -> np.ndarray:
    r"""Вектор нагрузки ``b[k] = ∫_Ω f·ψ_k dΩ`` (N,).

    ``f_values`` — значения правой части в узлах квадратуры (для (P1) это
    ``q̃ = q0 − r``, для (P2) — ``M/D``). Пересобирается на каждой итерации МОР.
    """
    X, Y, W = quad.x, quad.y, quad.w
    psi = domain.omega(X, Y) * basis.values(X, Y)   # (N, M)
    f_values = np.asarray(f_values, dtype=float)
    return (psi * W) @ f_values


__all__ = ["assemble_stiffness", "assemble_load"]
