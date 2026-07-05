r"""ktn.py — поправки уточнённой теории (тип Кармана–Тимошенко–Нагди).

Перенос поправок КТН из ``lab/Задачи Python/2/code.py`` (1D) в 2D. Уточнённая
теория добавляет к классике (Кирхгоф) учёт поперечного сдвига (Тимошенко/Рейснер,
функция Ψ) и поперечного обжатия (Нагди), т.е. поправки порядка ``(h/L)²``.

Характеристические длины поправок (по code.py):
    h_Ψ²  = h² / (6(1−ν))                 (сдвиг, hp2)
    h_*²  = ν h² / (8(1−ν))               (обжатие, hl2)
    h_z²  = h_Ψ² − h_*²                   (hz2)
    μ, λ  — постоянные Ламе.

Поправки входят в КОНТАКТНОЕ условие (обновление r) и в выражение прогиба. В 1D
``code.py`` это вторая разность ``w''``; в 2D кривизна берётся ПРЯМО из расщепления:
``Δw = −M/D`` (поле M уже считается в (P1), :meth:`PlateBending.moment`). Поэтому
КТН-поправка не требует численного дифференцирования — берём ``M`` в узлах.

Контактное смещение (для условия контакта, ``r``-обновление; code.py стр. 75–79):
    u_c = w + (2h_*²−h_Ψ²)·Δw − A·q0 − B·D·r,
    A = h/(8(λ+2μ)) − h_*²/(μh) + h_*²h_z²/D,
    B = 3h/(8(λ+2μ)) + h_*²/(μh) − h_*²h_z²/D.

КТН-прогиб срединной поверхности (для поправки w_max; code.py стр. 68–72):
    w_KTN = w + (2h_*²−h_Ψ²)·Δw + a·q0 + b·D·r,
    a = (3ν−1)h/(8(1−ν)E) − h_*²h_z²/D,
    b = (3−ν)h/(8(1−ν)E) + h_*²h_z²/D.

Член ``(2h_*²−h_Ψ²)·Δw`` действует как диффузия по полю реакции ⇒ СГЛАЖИВАЕТ
контактную реакцию у кромки пятна/входящего угла; члены ∝q0 и ∝r дают поправку
эффективного зазора и обратной связи (отсюда «поправки в q̃ и в обновление r»).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KTNParams:
    """Параметры и коэффициенты поправок КТН (выводятся из E, ν, h)."""

    E: float
    nu: float
    h: float

    # -- базовые величины ------------------------------------------------ #
    @property
    def D(self) -> float:
        return self.E * self.h**3 / (12.0 * (1.0 - self.nu**2))

    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lamb(self) -> float:
        return self.nu * self.E / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    @property
    def h_psi2(self) -> float:
        """h_Ψ² — длина поправки поперечного сдвига (hp2)."""
        return self.h**2 / (6.0 * (1.0 - self.nu))

    @property
    def h_star2(self) -> float:
        """h_*² — длина поправки поперечного обжатия (hl2)."""
        return self.nu * self.h**2 / (8.0 * (1.0 - self.nu))

    @property
    def h_z2(self) -> float:
        return self.h_psi2 - self.h_star2

    @property
    def c_curv(self) -> float:
        """Коэффициент кривизны (2h_*² − h_Ψ²) при Δw."""
        return 2.0 * self.h_star2 - self.h_psi2

    # -- коэффициенты контактного смещения (r-обновление) --------------- #
    @property
    def cq_contact(self) -> float:
        h, hl2, hz2, D = self.h, self.h_star2, self.h_z2, self.D
        return -(h / (8.0 * (self.lamb + 2.0 * self.mu)) - hl2 / (self.mu * h) + hl2 * hz2 / D)

    @property
    def cr_contact(self) -> float:
        h, hl2, hz2, D = self.h, self.h_star2, self.h_z2, self.D
        return -((3.0 * h) / (8.0 * (self.lamb + 2.0 * self.mu)) + hl2 / (self.mu * h) - hl2 * hz2 / D)

    # -- коэффициенты КТН-прогиба (срединная поверхность) --------------- #
    @property
    def cq_defl(self) -> float:
        h, nu, E, hl2, hz2, D = self.h, self.nu, self.E, self.h_star2, self.h_z2, self.D
        return (3.0 * nu - 1.0) * h / (8.0 * (1.0 - nu) * E) - hl2 * hz2 / D

    @property
    def cr_defl(self) -> float:
        h, nu, E, hl2, hz2, D = self.h, self.nu, self.E, self.h_star2, self.h_z2, self.D
        return (3.0 - nu) * h / (8.0 * (1.0 - nu) * E) + hl2 * hz2 / D

    # -- поля -------------------------------------------------------------- #
    def contact_displacement(self, w, lap_w, q0, r) -> np.ndarray:
        """Смещение контактной поверхности u_c (вход в условие контакта)."""
        r = np.asarray(r, float)
        return (
            np.asarray(w, float)
            + self.c_curv * np.asarray(lap_w, float)
            + self.cq_contact * q0
            + self.cr_contact * self.D * r
        )

    def corrected_deflection(self, w, lap_w, q0, r) -> np.ndarray:
        """КТН-поправленный прогиб срединной поверхности (для w_max)."""
        r = np.asarray(r, float)
        return (
            np.asarray(w, float)
            + self.c_curv * np.asarray(lap_w, float)
            + self.cq_defl * q0
            + self.cr_defl * self.D * r
        )

    @classmethod
    def from_config(cls, cfg) -> KTNParams:
        return cls(E=cfg.E, nu=cfg.nu, h=cfg.h)


# --------------------------------------------------------------------------- #
#  Классические соотношения (бывший plate_solver.model.ktn)
# --------------------------------------------------------------------------- #


def flexural_rigidity(E: float, h: float, nu: float) -> float:
    r"""Классическая цилиндрическая жёсткость пластины.

    .. math:: D = \frac{E\,h^3}{12\,(1-\nu^2)}

    Parameters
    ----------
    E : модуль Юнга.
    h : толщина пластины.
    nu : коэффициент Пуассона (0 <= nu < 0.5).
    """
    if h <= 0:
        raise ValueError("Толщина h должна быть положительной.")
    if not (-1.0 < nu < 0.5):
        raise ValueError("Коэффициент Пуассона nu вне допустимого диапазона.")
    return E * h**3 / (12.0 * (1.0 - nu**2))


@dataclass(frozen=True)
class PlateMaterial:
    """Изотропный упругий материал пластины.

    Attributes
    ----------
    E : модуль Юнга.
    nu : коэффициент Пуассона.
    h : толщина.
    """

    E: float
    nu: float
    h: float

    @property
    def D(self) -> float:
        """Цилиндрическая жёсткость D (классическая)."""
        return flexural_rigidity(self.E, self.h, self.nu)


__all__ = ["KTNParams", "PlateMaterial", "flexural_rigidity"]
