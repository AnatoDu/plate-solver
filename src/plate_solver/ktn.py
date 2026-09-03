r"""ktn.py — поправки уточнённой теории (тип Кармана–Тимошенко–Нагди).

Уточнённая теория добавляет к классике (Кирхгоф) учёт поперечного сдвига
(Тимошенко/Рейснер, функция Ψ) и поперечного обжатия (Нагди), т.е. поправки
порядка ``(h/L)²``.

Характеристические длины поправок (обозначения этого модуля — исторические,
словарь соответствий каноническим именам ``faces.py`` — в NOTES §12):
    h_Ψ²  = h² / (6(1−ν))                 (сдвиг,  hp2 = h_psi_sq)
    h_*²  = ν h² / (8(1−ν))               (обжатие, hl2 ≡ канон. h_c_sq)
    h_z²  = h_Ψ² − h_*²                   (hz2 ≡ канон. h_star_sq)
    μ, λ  — постоянные Ламе.

Поправки входят в КОНТАКТНОЕ условие (обновление r) и в выражение прогиба. В 1D
это вторая разность ``w''``; в 2D кривизна берётся ПРЯМО из расщепления:
``Δw = −M/D`` (поле M уже считается в (P1), :meth:`PlateBending.moment`). Поэтому
КТН-поправка не требует численного дифференцирования — берём ``M`` в узлах.

Контактное смещение (для условия контакта, ``r``-обновление) — формула (9)
публикации (Дуркин, Ермоленко, 2025), NOTES §21.1:
    u_c = w + (2h_*²−h_Ψ²)·Δw − κ_q·q⁺ − κ_r·r,
    κ_q = h/(8(λ+2μ)) − h_*²/(μh) + h_*²h_z²/D,
    κ_r = 3h/(8(λ+2μ)) + h_*²/(μh) − h_*²h_z²/D
        = 3(1+ν)(2−4ν+ν²)·h / (16E(1−ν))            (тождество проверено sympy).

КТН-прогиб срединной поверхности (для поправки w_max):
    w_KTN = w + (2h_*²−h_Ψ²)·Δw + a·q⁺ + b·r,
    a = (3ν−1)h/(8(1−ν)E) − h_*²h_z²/D,
    b = (3−ν)h/(8(1−ν)E) + h_*²h_z²/D.

РАЗМЕРНОСТЬ (v0.8.0, исправление): κ_q, κ_r — ПОДАТЛИВОСТИ [длина/давление],
поэтому ``κ_r·r`` и ``κ_q·q`` — длины. До v0.8.0 член реакции умножался ещё и
на цилиндрическую жёсткость D (``κ_r·D·r``, дословный перенос 1D-скрипта),
что давало величину размерности Па·м⁴ и делало модель ЗАВИСИМОЙ ОТ СИСТЕМЫ
ЕДИНИЦ: при (E, q) → (E/s, q/s) классика воспроизводится бит-в-бит, а КТН-контакт
менялся качественно. Ворота — ``tests/test_units_invariance.py``; история и
следствия — CHANGELOG [0.8.0], NOTES §21.

Безразмерная мера эффекта в контакте — ``κ_r/‖G‖ ∝ (h/a)⁴`` (‖G‖ ~ податливость
изгиба ``a⁴/D``): поправка гаснет в тонком пределе, как и положено O(h²)-теории.
Член ``(2h_*²−h_Ψ²)·Δw`` (кривизна лицевой) сглаживает реакцию у кромки пятна;
член ``−κ_r·r`` даёт положительную диагональную добавку к оператору задачи
дополнительности (МОР становится строго монотонным ⇒ сходится там, где
классическая итерация лишь «полусходится», NOTES §11).
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
        return -(
            (3.0 * h) / (8.0 * (self.lamb + 2.0 * self.mu)) + hl2 / (self.mu * h) - hl2 * hz2 / D
        )

    # -- коэффициенты КТН-прогиба (срединная поверхность) --------------- #
    @property
    def cq_defl(self) -> float:
        h, nu, E, hl2, hz2, D = self.h, self.nu, self.E, self.h_star2, self.h_z2, self.D
        return (3.0 * nu - 1.0) * h / (8.0 * (1.0 - nu) * E) - hl2 * hz2 / D

    @property
    def cr_defl(self) -> float:
        h, nu, E, hl2, hz2, D = self.h, self.nu, self.E, self.h_star2, self.h_z2, self.D
        return (3.0 - nu) * h / (8.0 * (1.0 - nu) * E) + hl2 * hz2 / D

    # -- κ-податливости лицевого условия (формула (9) публикации 2025) ---- #
    @property
    def kappa_q(self) -> float:
        r"""``κ_q = −cq_contact`` — податливость лицевой по нагрузке ДАЛЬНЕЙ грани [м/Па].

        Знак НЕ фиксирован: ``κ_q > 0`` при ``ν < (√10 − 2)/3 ≈ 0.3874`` и
        меняется на противоположный выше (при ν = 0.45 это −6.7e-3·h/E).
        Физически это законно: давление на ПРОТИВОПОЛОЖНОЙ грани смещает
        рассматриваемую лицевую через сумму обжатия и «эффекта Пуассона»
        поперечного расширения, и у почти несжимаемого материала второй вклад
        перевешивает. Ближняя податливость ``κ_r`` положительна при всех
        допустимых ν — именно она регуляризует контакт (v0.8.0: прежний
        докстринг объявлял положительными обе).
        """
        return -self.cq_contact

    @property
    def kappa_r(self) -> float:
        r"""``κ_r = −cr_contact > 0`` — податливость лицевой по реакции [м/Па].

        .. math:: \kappa_r = \frac{3(1+\nu)(2-4\nu+\nu^2)\,h}{16E(1-\nu)}

        Диагональная добавка ``κ_r`` в операторе задачи дополнительности:
        именно она делает МОР строго монотонным (сходимость вместо
        «полусходимости» классики) и задаёт масштаб эффекта ``κ_r/‖G‖ ∝ (h/a)⁴``.
        """
        return -self.cr_contact

    # -- поля -------------------------------------------------------------- #
    def contact_displacement(self, w, lap_w, q0, r, *, terms=None) -> np.ndarray:
        r"""Прогиб контактирующей (НИЖНЕЙ) лицевой поверхности u_c по КТН.

        Этой величиной проверяется зазор в контактном условии (обновление r):

        .. math:: u_c = w + (2h_*^2-h_\Psi^2)\,\Delta w - \kappa_q q^+ - \kappa_r r

        КАНОН ПАКЕТА (NOTES §21.1) — формула (9) публикации 2025. Классическое
        3D-восстановление — независимая диагностика (§21.2): кривизный блок
        тождественен (поверхность и знаки подтверждены), q,r-члены различаются
        множителем порядка единицы при одинаковой размерности (при ν = 0.3
        канон/восстановление = 0.763) — в пределах O(h²)-точности теории.
        Синоним: :meth:`w_face_bottom`.

        ``terms`` (:class:`~plate_solver.faces.FaceTerms`) включает слагаемые
        по отдельности (лестница слагаемых); ``None`` — все три (штатный путь,
        арифметика прежняя).

        v0.8.0: член реакции — ``κ_r·r`` (ранее ошибочно ``κ_r·D·r``, см.
        докстринг модуля и ворота ``tests/test_units_invariance.py``).
        """
        return self._face_sum(w, lap_w, q0, r, self.cq_contact, self.cr_contact, terms)

    @staticmethod
    def _terms_flags(terms) -> tuple[bool, bool, bool]:
        """(кривизна, нагрузка, реакция); ``None`` ⇒ все включены."""
        if terms is None:
            return True, True, True
        return bool(terms.curvature), bool(terms.load), bool(terms.reaction)

    def _face_sum(self, w, lap_w, q0, r, c_q, c_r, terms) -> np.ndarray:
        """Общая сборка лицевой/срединной поправки с переключателями слагаемых."""
        use_curv, use_q, use_r = self._terms_flags(terms)
        if use_curv and use_q and use_r:                 # штатный путь: без ветвлений
            return (
                np.asarray(w, float)
                + self.c_curv * np.asarray(lap_w, float)
                + c_q * q0
                + c_r * np.asarray(r, float)
            )
        out = np.array(np.asarray(w, float), copy=True)
        if use_curv:
            out = out + self.c_curv * np.asarray(lap_w, float)
        if use_q:
            out = out + c_q * q0
        if use_r:
            out = out + c_r * np.asarray(r, float)
        return out

    def w_face_bottom(self, w, lap_w, q0, r, *, terms=None) -> np.ndarray:
        """Прогиб нижней лицевой (синоним :meth:`contact_displacement`)."""
        return self.contact_displacement(w, lap_w, q0, r, terms=terms)

    def corrected_deflection(self, w, lap_w, q0, r, *, terms=None) -> np.ndarray:
        """КТН-поправленный прогиб срединной поверхности (для w_max).

        ``w_KTN = w + (2h_*²−h_Ψ²)·Δw + a·q⁺ + b·r`` (v0.8.0: член реакции без
        множителя D — та же размерная поправка, что в :meth:`contact_displacement`).
        ``terms`` — переключатели слагаемых (см. :meth:`contact_displacement`).
        """
        return self._face_sum(w, lap_w, q0, r, self.cq_defl, self.cr_defl, terms)

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




# --------------------------------------------------------------------------- #
#  Напряжения на лицевых поверхностях (NOTES §19)
# --------------------------------------------------------------------------- #
def stresses_faces(Mx, My, Mxy, h: float, nu: float, q_top=0.0, q_bottom=0.0,
                   Nx=0.0, Ny=0.0, Nxy=0.0):
    r"""Напряжения на лицевых поверхностях z = ±h/2 (канон Ермоленко–Туркова, b=0).

    Формула (11) при b = 0 — суперпозиция мембранной и изгибной частей:

    .. math::
        \sigma_{ii}^{\pm h/2} = \frac{T_{ii}}{h} \pm\,6 M_{ii}/h^2
            - \frac{\nu}{1-\nu}\, q_n^{\pm}, \qquad
        \sigma_{12}^{\pm h/2} = \frac{T_{12}}{h} \pm\,6 M_{12}/h^2 .

    ``Nx, Ny, Nxy`` — мембранные усилия ``T_ij`` (нелинейные теории; по
    умолчанию 0 — линейный изгиб, T_ij ≡ 0, прежнее поведение число-в-число).

    ЗНАК ЧЛЕНА ОБЖАТИЯ (исправлено в v0.8.0, аудит P10). Обобщённый закон Гука
    в плоском деформированном слое даёт
    ``σ_x = E(ε_x + νε_y)/(1−ν²) + ν/(1−ν)·σ_z``. Величина ``q_n ≥ 0`` — это
    ДАВЛЕНИЕ на грань, а не ``σ_z``: по конвенции NOTES §0 (ось z вниз,
    ``q > 0`` вниз) нормальное напряжение на нагруженной грани равно
    ``σ_z = −q_n``, откуда вклад обжатия ``−ν/(1−ν)·q_n`` — СЖАТИЕ. До v0.8.0
    член входил со знаком «+», то есть давление давало растяжение: контроль —
    стеснённый слой (``ε_x = ε_y = 0`` под давлением p) обязан дать
    ``σ_x = −ν/(1−ν)·p`` (независимые ворота в ``tests/test_stresses.py``).

    ``q_n^{+}`` — давление на ВЕРХНЕЙ лицевой (внешняя нагрузка), ``q_n^{-}`` —
    на НИЖНЕЙ (в контактной зоне — реакция r ≥ 0, вне зоны — 0). ОСИ: прогиб и
    нагрузка положительны «вниз», ось z сонаправлена ⇒ ВЕРХНЯЯ лицевая
    (сторона внешней нагрузки) — это z = −h/2, НИЖНЯЯ (сторона основания) —
    z = +h/2. Контроль физики: в пролёте при q > 0 низ растянут (σ_bot > 0),
    верх сжат. Знаки зафиксированы 1D-тождеством (B3-т1) и таблицей NOTES §19.

    Возвращает словарь шести полей: sx_top, sx_bot, sy_top, sy_bot,
    txy_top, txy_bot (top = z=−h/2, bot = z=+h/2; формы входных массивов).
    """
    Mx = np.asarray(Mx, float)
    My = np.asarray(My, float)
    Mxy = np.asarray(Mxy, float)
    k = 6.0 / h**2
    # обжатие: σ_z = −q_n на грани под давлением q_n ≥ 0 (см. докстринг)
    c = -nu / (1.0 - nu)
    # мембранная часть T_ij/h (формула (11) при T_ij ≠ 0, b = 0): постоянна по
    # толщине ⇒ входит в ОБЕ лицевые с ОДНИМ знаком (v0.6.6; для линейных
    # теорий N ≡ 0 — прежние числа не сдвигаются)
    mNx = np.asarray(Nx, float) / h
    mNy = np.asarray(Ny, float) / h
    mNxy = np.asarray(Nxy, float) / h
    return {
        "sx_top": mNx - k * Mx + c * np.asarray(q_top, float),
        "sx_bot": mNx + k * Mx + c * np.asarray(q_bottom, float),
        "sy_top": mNy - k * My + c * np.asarray(q_top, float),
        "sy_bot": mNy + k * My + c * np.asarray(q_bottom, float),
        "txy_top": mNxy - k * Mxy,
        "txy_bot": mNxy + k * Mxy,
    }


def von_mises(sx, sy, txy) -> np.ndarray:
    r"""Эквивалентное напряжение фон Мизеса (плоское напряжённое состояние).

    .. math:: \sigma_{vm} = \sqrt{\sigma_x^2 - \sigma_x\sigma_y + \sigma_y^2
        + 3\,\tau_{xy}^2}

    Тривиальная алгебра над уже посчитанной σ-шестёркой лицевых (v0.6.6) —
    первый вопрос прочниста к любому расчёту.
    """
    sx = np.asarray(sx, float)
    sy = np.asarray(sy, float)
    txy = np.asarray(txy, float)
    return np.sqrt(sx**2 - sx * sy + sy**2 + 3.0 * txy**2)


__all__ = ["KTNParams", "PlateMaterial", "flexural_rigidity", "stresses_faces",
           "von_mises"]
