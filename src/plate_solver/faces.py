r"""faces.py — лицевые величины уточнённой теории ПЕРВЫМ КЛАССОМ (§6).

Прогиб лицевой поверхности ``u_c`` и напряжения на верхней/нижней гранях —
самостоятельный, переиспользуемый слой, доступный ЛЮБОЙ теории (для
``classic``/``karman`` лицевые сводятся к тривиальным/мембранным поправкам,
для ``ktn_linear``/``ktn_full`` — к полным). Всё считается из ``w, Δw, M, N, q``
и параметров толщины §3.2 — БЕЗ поля сдвига ``ψ`` (оно нужно только для
распределения поперечных касательных напряжений, §3.6).

КАНОНИЧЕСКИЕ параметры толщины (§3.2, единые имена по всему коду — §12):

.. math::
    h_\psi^2 = \frac{h^2}{6(1-\nu)},\quad
    h_*^2 = \frac{(4-3\nu)h^2}{24(1-\nu)},\quad
    h_c^2 = \frac{\nu h^2}{8(1-\nu)} = h_\psi^2 - h_*^2.

Все КТН-поправки — порядка ``O(h²/L²)``: при ``h/L → 0`` гаснут (Gate R4).

Соответствие устаревшим именам ``ktn.py`` (историческая путаница, §12):
``ktn.KTNParams.h_psi2 = h_psi_sq`` (h_ψ²), ``h_star2 = h_c_sq`` (h_c²),
``h_z2 = h_star_sq`` (h_*²). Чтобы линейные лицевые величины ``ktn_linear``
воспроизводились ЧИСЛО-В-ЧИСЛО (регресс не сдвигается), выверенные
коэффициенты ``c_curv, κ_q, κ_r`` берутся из :class:`~plate_solver.ktn.KTNParams`
(мост :meth:`FaceParams.ktn`); тождество коэффициентов проверяется тестом.

Знак члена обжатия ``h_*²Δw`` в напряжениях ПРОТИВОПОЛОЖЕН на верхней и нижней
гранях — это и есть подпись КТН: ``Δw = −(M_x+M_y)/(D(1+ν))`` меняется по
области даже под РАВНОМЕРНОЙ нагрузкой, поэтому лицевые напряжения КТН
отличаются от классических, тогда как срединный прогиб не меняется (Δq = 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceParams:
    """Параметры лицевых величин уточнённой теории (канонические имена §3.2)."""

    E: float
    nu: float
    h: float

    def __post_init__(self):
        # Тождество §3.2 как assert в коде: h_c² = h_ψ² − h_*².
        assert abs(self.h_c_sq - (self.h_psi_sq - self.h_star_sq)) < 1e-12 * self.h**2, (
            "нарушено тождество h_c² = h_ψ² − h_*² (§3.2)")

    # -- базовые упругие величины --------------------------------------- #
    @property
    def D(self) -> float:
        """Цилиндрическая жёсткость ``D = E h³ / [12(1−ν²)]``."""
        return self.E * self.h**3 / (12.0 * (1.0 - self.nu**2))

    @property
    def mu(self) -> float:
        """Модуль сдвига ``μ = E / [2(1+ν)]``."""
        return self.E / (2.0 * (1.0 + self.nu))

    # -- параметры толщины (канонические имена §3.2) -------------------- #
    @property
    def h_psi_sq(self) -> float:
        """``h_ψ² = h²/[6(1−ν)]`` — длина поправки поперечного СДВИГА."""
        return self.h**2 / (6.0 * (1.0 - self.nu))

    @property
    def h_star_sq(self) -> float:
        """``h_*² = (4−3ν)h²/[24(1−ν)]`` — длина поправки поперечного ОБЖАТИЯ."""
        return (4.0 - 3.0 * self.nu) * self.h**2 / (24.0 * (1.0 - self.nu))

    @property
    def h_c_sq(self) -> float:
        """``h_c² = νh²/[8(1−ν)] = h_ψ² − h_*²`` (связной параметр)."""
        return self.nu * self.h**2 / (8.0 * (1.0 - self.nu))

    @property
    def c_curv(self) -> float:
        """Коэффициент кривизны при Δw: ``h_c² − h_*² = 2h_c² − h_ψ²`` (§6.1)."""
        return self.h_c_sq - self.h_star_sq

    # -- мост к выверенным коэффициентам ktn_linear --------------------- #
    def ktn(self):
        """:class:`~plate_solver.ktn.KTNParams` с теми же (E, ν, h).

        Источник ЗАМОРОЖЕННЫХ линейных коэффициентов ``κ_q, κ_r`` (лицевой и
        срединный варианты): гарантирует, что ``ktn_linear`` считается
        число-в-число (регресс не сдвигается). Тождество ``c_curv`` этого
        класса и ``KTNParams.c_curv`` проверяется тестом Gate R5.
        """
        from .ktn import KTNParams

        return KTNParams(E=self.E, nu=self.nu, h=self.h)

    @classmethod
    def from_config(cls, cfg) -> FaceParams:
        """Собрать из ``Config`` (E, ν, h)."""
        return cls(E=cfg.E, nu=cfg.nu, h=cfg.h)

    # -- интроспекция (§6.3) -------------------------------------------- #
    def introspection(self, length: float | None = None) -> dict:
        """Словарь вычисленных параметров толщины (для Result/лога/ноутбука).

        ``length`` — характерный размер L (радиус/полусторона); при задании
        добавляется безразмерное ``h/L`` и оценка порядка эффекта ``(h/L)²`` —
        видно, как КТН-поправка гаснет с утоньшением пластины (Gate R4).
        """
        out = {
            "h_psi_sq": self.h_psi_sq,
            "h_star_sq": self.h_star_sq,
            "h_c_sq": self.h_c_sq,
            "c_curv": self.c_curv,
        }
        if length is not None and length > 0.0:
            out["h_over_L"] = self.h / length
            out["order_h2_L2"] = (self.h / length) ** 2
        return out

    # -- лицевой прогиб (§6.1) ------------------------------------------ #
    def face_deflection(self, w, lap_w, q_n, r=0.0, *, surface: str = "bottom") -> np.ndarray:
        r"""Прогиб лицевой поверхности ``u_c`` (§6.1, КАНОН пакета NOTES §21.1).

        .. math:: u_c = w + (h_c^2 - h_*^2)\,\Delta w - \kappa_q q_n - \kappa_r D r,

        (без контакта ``r = 0``). ``surface='bottom'`` — контактирующая (нижняя)
        грань: полная коррекция уточнённой теории (:meth:`ktn.contact_displacement`,
        число-в-число ``ktn_linear``). ``surface='top'`` — верхняя грань: по
        канону пакета совпадает со срединной (``≡ w``); выделяется формулой
        только контактирующая лицевая (NOTES §21.1, решение автора).
        Смещение контактирующей лицевой ``dh = u_c^{bot} − w`` — подпись КТН в
        кинематике (в зоне контакта ``dh < 0``).
        """
        w = np.asarray(w, float)
        if surface == "top":
            return w.copy()
        if surface == "bottom":
            return self.ktn().contact_displacement(w, lap_w, q_n, r)
        raise ValueError(f"surface: ожидалось 'bottom' | 'top', получено {surface!r}")

    def mid_corrected(self, w, lap_w, q_n, r=0.0) -> np.ndarray:
        """КТН-поправленный СРЕДИННЫЙ прогиб (для w_max; §6.1, ``corrected_deflection``)."""
        return self.ktn().corrected_deflection(w, lap_w, q_n, r)


def membrane_face_stress(Nx, Ny, Nxy, h: float) -> dict:
    r"""Мембранная составляющая лицевых напряжений: ``σ^m = N/h`` (одинакова на гранях).

    Для ``karman``/``ktn_full`` мембранные усилия ``N`` дают вклад ``N/h`` в
    напряжения обеих лицевых поверхностей (растяжение срединной плоскости). Для
    ``classic``/``ktn_linear`` ``N ≡ 0`` ⇒ вклад нулевой. Возвращает
    ``sx_m, sy_m, txy_m`` (формы входных массивов).
    """
    inv_h = 1.0 / h
    return {
        "sx_m": np.asarray(Nx, float) * inv_h,
        "sy_m": np.asarray(Ny, float) * inv_h,
        "txy_m": np.asarray(Nxy, float) * inv_h,
    }


def face_stresses(Mx, My, Mxy, *, h: float, nu: float, q_top=0.0, q_bottom=0.0,
                  Nx=None, Ny=None, Nxy=None) -> dict:
    r"""Полные напряжения на лицевых поверхностях (изгиб + обжатие + мембрана).

    Изгибная и обжимная (член ``ν/(1−ν)·q_n``) части — канон Ермоленко–Турковой
    (:func:`plate_solver.ktn.stresses_faces`); мембранная часть ``N/h`` (§6.1)
    добавляется при заданных ``N`` (нелинейные теории). Подпись КТН —
    ненулевой член обжатия даже под равномерной нагрузкой (через ``q_n``, а в
    ``ktn_full`` — и через регуляризованную кривизну). Возвращает словарь
    полей ``sx_top, sx_bot, sy_top, sy_bot, txy_top, txy_bot``.
    """
    from .ktn import stresses_faces

    s = stresses_faces(Mx, My, Mxy, h=h, nu=nu, q_top=q_top, q_bottom=q_bottom)
    if Nx is not None:
        m = membrane_face_stress(Nx, Ny, Nxy, h)
        for face in ("top", "bot"):
            s[f"sx_{face}"] = s[f"sx_{face}"] + m["sx_m"]
            s[f"sy_{face}"] = s[f"sy_{face}"] + m["sy_m"]
            s[f"txy_{face}"] = s[f"txy_{face}"] + m["txy_m"]
    return s


__all__ = ["FaceParams", "face_stresses", "membrane_face_stress"]
