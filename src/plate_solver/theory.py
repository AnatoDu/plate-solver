r"""theory.py — единая параметрическая модель теорий (архитектурный стержень v0.6.0).

Все четыре теории пластины — именованные ПРЕСЕТЫ параметров ОДНОГО решателя
полной КТН (`ktn_solver.KTNSolver`) с управляемыми членами; редукции точны ПО
ПОСТРОЕНИЮ (тот же путь исполнения), а не «совпадением до машинной точности»
(усиление теоремы T6, §16 ТЗ).

Две оси и решётка 2×2 (§3.3):

|                     | уточнение ВЫКЛ | уточнение ВКЛ |
|---------------------|----------------|---------------|
| нелинейность ВЫКЛ   | ``classic``    | ``ktn_linear``|
| нелинейность ВКЛ    | ``karman``     | ``ktn_full``  |

* ось геометрической нелинейности — флаг ``membrane`` (член ``L(Φ, w)``);
* ось уточнения — параметры толщины ``h_psi_sq`` (регуляризация связи
  ``(I − h_ψ²Δ)``), ``h_star_sq`` (нагрузочная поправка ``−h_*²Δq`` и лицевой
  сдвиг/обжатие).

Физические значения (прил. A, ν=0.3 → ``h_ψ²=0.2381h²``, ``h_*²=0.1845h²``,
``h_c²=0.0536h²``) авто-вычисляются из ``(ν, h)`` через ``faces.FaceParams``
(единый источник формул, §12). Непрерывный морфинг ``refinement_scale`` α
масштабирует уточнение (``h_ψ²→α h_ψ²``, ``h_*²→α h_*²``): α=0 — Карман, α=1 —
полная КТН, плавно между (§3.5) — «одна теория с крутилкой».
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .faces import FaceParams

#: канонические имена пресетов (обратная совместимость v0.5.0); "ktn" — алиас.
PRESET_NAMES = ("classic", "karman", "ktn_linear", "ktn_full")


@dataclass(frozen=True)
class TheoryParams:
    r"""Управляемые параметры теории (§3.2) — конфигурация ОДНОГО решателя КТН.

    Attributes
    ----------
    membrane : геометрическая нелинейность (член ``L(Φ, w)``). Выкл ⇒ бигармоника
        линейна (нет мембранной связи).
    h_psi_sq : регуляризация связи ``(I − h_ψ²Δ)`` (член B). Физический
        ``h²/[6(1−ν)]`` или 0. Активен в РЕШЕНИИ только при ``membrane``.
    h_star_sq : нагрузочная поправка ``−h_*²Δq`` (член A) и лицевой сдвиг.
        Физический ``(4−3ν)h²/[24(1−ν)]`` или 0.
    compression : обжатие Нагди в ЛИЦЕВЫХ величинах (``h_c²``, постобработка).
    shear_field : решать поле поперечного сдвига ``ψ`` (постобработка §3.6 v0.5.0).
    """

    membrane: bool
    h_psi_sq: float
    h_star_sq: float
    compression: bool
    shear_field: bool = False

    @property
    def h_c_sq(self) -> float:
        """``h_c² = h_ψ² − h_*²`` (связной параметр обжатия)."""
        return self.h_psi_sq - self.h_star_sq

    @property
    def refined(self) -> bool:
        """Активны ли уточняющие члены толщины (h_ψ² или h_*² ненулевые)."""
        return self.h_psi_sq > 0.0 or self.h_star_sq > 0.0

    @property
    def solve_ktn_terms(self) -> bool:
        """Активны ли КТН-члены (A), (B) в РЕШЕНИИ (только нелинейный режим)."""
        return self.membrane and self.refined

    @property
    def face_curv_coeff(self) -> float:
        r"""Коэффициент при Δw в лицевом прогибе ``u_c = w + (h_c²−h_*²)Δw`` (§4.1).

        Масштабируется теорией: для ``classic``/``karman`` (уточнение выкл) —
        ноль ⇒ контакт «щупает» СРЕДИННУЮ поверхность (``u_c = w``); для полной
        КТН — физическая лицевая кривизна (подпись КТН в контакте).
        """
        return self.h_c_sq - self.h_star_sq

    def with_refinement_scale(self, alpha: float) -> TheoryParams:
        r"""Непрерывный морфинг (§3.5): ``h_ψ²→α h_ψ²``, ``h_*²→α h_*²``, α∈[0,1].

        Из физической КТН (``α=1``) плавно в Карман (``α=0``, уточнение снято).
        Даёт графики «решение vs α» — как ОДНА теория непрерывно переходит в
        подтеорию (демонстрация §6.2, коммутативность редукций).
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"refinement_scale α: ожидалось [0, 1], получено {alpha}")
        return replace(self, h_psi_sq=alpha * self.h_psi_sq,
                       h_star_sq=alpha * self.h_star_sq)


# --------------------------------------------------------------------------- #
#  Физические значения параметров толщины (прил. A) — единый источник faces.py
# --------------------------------------------------------------------------- #
def physical_h_psi_sq(nu: float, h: float) -> float:
    """``h_ψ² = h²/[6(1−ν)]`` (прил. A)."""
    return FaceParams(E=1.0, nu=nu, h=h).h_psi_sq


def physical_h_star_sq(nu: float, h: float) -> float:
    """``h_*² = (4−3ν)h²/[24(1−ν)]`` (прил. A)."""
    return FaceParams(E=1.0, nu=nu, h=h).h_star_sq


# --------------------------------------------------------------------------- #
#  Именованные пресеты (§3.4) — фабрики TheoryParams
# --------------------------------------------------------------------------- #
def classic() -> TheoryParams:
    """Линейная теория Кирхгофа: нелинейность и уточнение выключены."""
    return TheoryParams(membrane=False, h_psi_sq=0.0, h_star_sq=0.0,
                        compression=False, shear_field=False)


def karman() -> TheoryParams:
    """Геометрически-нелинейная теория Фёппля–Кармана: нелинейность вкл, уточнение выкл."""
    return TheoryParams(membrane=True, h_psi_sq=0.0, h_star_sq=0.0,
                        compression=False, shear_field=False)


def ktn_linear(nu: float, h: float) -> TheoryParams:
    """Линейные поправки сдвига/обжатия постобработкой: нелинейность выкл, уточнение вкл."""
    return TheoryParams(membrane=False, h_psi_sq=0.0,
                        h_star_sq=physical_h_star_sq(nu, h),
                        compression=True, shear_field=False)


def ktn_full(nu: float, h: float) -> TheoryParams:
    """Полная нелинейная КТН: нелинейность и уточнение включены (физические h_ψ², h_*²)."""
    return TheoryParams(membrane=True, h_psi_sq=physical_h_psi_sq(nu, h),
                        h_star_sq=physical_h_star_sq(nu, h),
                        compression=True, shear_field=True)


_PRESETS = {"classic": classic, "karman": karman,
            "ktn_linear": ktn_linear, "ktn_full": ktn_full}


def from_preset(name: str, nu: float, h: float) -> TheoryParams:
    """Собрать :class:`TheoryParams` по имени пресета (``[model] theory``).

    Алиас ``ktn`` → ``ktn_linear`` (обратная совместимость v0.5.0).
    Линейные пресеты (``classic``/``karman``) не зависят от (ν, h).
    """
    canonical = "ktn_linear" if name == "ktn" else name
    if canonical not in _PRESETS:
        raise ValueError(f"пресет теории: ожидалось {' | '.join(PRESET_NAMES)}, "
                         f"получено {name!r}")
    factory = _PRESETS[canonical]
    return factory(nu, h) if canonical in ("ktn_linear", "ktn_full") else factory()


__all__ = [
    "TheoryParams",
    "classic",
    "karman",
    "ktn_linear",
    "ktn_full",
    "from_preset",
]
