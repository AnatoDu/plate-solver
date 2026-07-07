"""plate_solver — комплекс программ для расчёта изгиба и контактного
взаимодействия упругих пластин произвольного очертания.

Триада (по Самарскому): Модель — Алгоритм — Программа. Метод: R-функции
(геометрия + краевые условия, Рвачёв) + метод Ритца (решатель на базисе
``ω·Φ``, Φ — полиномы Чебышёва) + метод обобщённой реакции (контакт).
Изгиб считается расщеплением бигармоники ``D·Δ²w = q̃`` на две задачи
Пуассона с одним оператором ``−Δ`` и однородным Дирихле — см.
``plate_solver.plate``. Поправки уточнённой теории типа
Кармана–Тимошенко–Нагди (поперечный сдвиг + обжатие) — ``plate_solver.ktn``.

Пакет ПЛОСКИЙ (обоснование — docs/ARCHITECTURE.md); «нелоскость» — этим
фасадом: публичные точки входа сгруппированы секциями ниже и доступны
как ``plate_solver.<имя>`` (ленивая подгрузка — ``import plate_solver``
не тянет sympy/matplotlib/scikit-fem раньше времени). Полный справочник —
docs/API.md.
"""

from __future__ import annotations

from . import config
from .config import Config
from .ktn import KTNParams, PlateMaterial, flexural_rigidity
from .problem import CaseError, Problem

__version__ = "0.5.0"

#: фасад: имя → модуль; секции — комментариями
_FACADE: dict[str, str] = {
    # -- постановка и диспетчер ------------------------------------------- #
    "solve": "dispatch",
    "Result": "dispatch",
    "build_domain": "dispatch",
    # -- геометрия (R-функции) -------------------------------------------- #
    "Domain": "geometry",
    "make_circle": "geometry",
    "make_rectangle": "geometry",
    "make_L": "geometry",
    "make_annulus": "geometry",
    "make_compose": "geometry",
    # -- решатели изгиба --------------------------------------------------- #
    "PlateBending": "plate",
    "ClampedPlate": "clamped",
    "MixedRectPlate": "clamped",
    # -- геометрическая нелинейность (теория Кармана) ---------------------- #
    "KarmanPlate": "membrane",
    "KarmanResult": "membrane",
    # -- полная нелинейная КТН --------------------------------------------- #
    "KTNPlate": "ktn_full",
    # -- контакт (МОР) ------------------------------------------------------ #
    "solve_contact": "contact",
    "ContactMOR": "contact",
    "ContactResult": "contact",
    "TwoPlateMOR": "contact",
    "TwoPlateResult": "contact",
    # -- верификация -------------------------------------------------------- #
    "resolve_reference": "references",
    "verify_result": "references",
    # -- напряжения и лицевые поверхности ----------------------------------- #
    "stresses_faces": "ktn",
    # -- лицевые величины первым классом (уточнённая теория) ---------------- #
    "FaceParams": "faces",
    "face_stresses": "faces",
}


def __getattr__(name: str):
    """Ленивый фасад: plate_solver.solve → plate_solver.dispatch.solve."""
    if name in _FACADE:
        import importlib

        mod = importlib.import_module(f".{_FACADE[name]}", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # ядро (импортируется сразу)
    "config",
    "Config",
    "KTNParams",
    "PlateMaterial",
    "flexural_rigidity",
    "CaseError",
    "Problem",
    "__version__",
    # фасад (лениво)
    *sorted(_FACADE),
]
