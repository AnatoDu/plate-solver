"""Подпакет analytic — одномерные аналитические решения.

ВАЖНО по специальности 1.2.2 (физмат): эти решения подаются НЕ как
самостоятельный «аналитический вклад» (это технический п. 7 паспорта),
а как средство ВЕРИФИКАЦИИ двумерного численного метода (п. 5, валидация).
Сравнение 1D ↔ 2D — положение на защиту № 4 и автотест в tests/.
"""

from .circular import (
    clamped_uniform,
    clamped_uniform_wmax,
    simply_supported_uniform,
    simply_supported_uniform_wmax,
)
from .strip_contact import W_MAPLE, X_MAPLE, deflection_maple

__all__ = [
    "clamped_uniform",
    "clamped_uniform_wmax",
    "simply_supported_uniform",
    "simply_supported_uniform_wmax",
    "X_MAPLE",
    "W_MAPLE",
    "deflection_maple",
]
