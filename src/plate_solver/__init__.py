"""plate_solver — комплекс программ для расчёта изгиба и контактного
взаимодействия упругих пластин произвольного очертания.

Триада (по Самарскому): Модель — Алгоритм — Программа. Метод: R-функции
(геометрия + краевые условия, Рвачёв) + метод Ритца (решатель на базисе
``ω·Φ``, Φ — полиномы Чебышёва) + метод обобщённой реакции (контакт).
Изгиб считается расщеплением бигармоники ``D·Δ²w = q̃`` на две задачи
Пуассона с одним оператором ``−Δ`` и однородным Дирихле — см.
``plate_solver.plate``. Поправки уточнённой теории типа
Кармана–Тимошенко–Нагди (поперечный сдвиг + обжатие) — ``plate_solver.ktn``.

Модули лежат плоско:
  * geometry, basis, quadrature, assembler — область (R-функции) и дискретизация;
  * poisson, plate, clamped, radial        — решатели изгиба (Ритц);
  * contact, ktn                           — контакт (МОР) и поправки КТН;
  * mor1d, green1d, stamp, stamp_ritz      — 1D-задел (балка-полоса, штамп);
  * analytic, ladder, penalty, verify_fem  — эталоны и верификация;
  * rfunctions, problems                   — примитивы R0 и постановки задач;
  * viz                                    — графика.

Подмодули импортируются лениво (по требованию), чтобы ``import plate_solver``
не тянул за собой sympy/matplotlib/scikit-fem раньше времени.
"""

from __future__ import annotations

from . import config
from .config import Config
from .ktn import KTNParams, PlateMaterial, flexural_rigidity
from .problem import CaseError, Problem

__version__ = "0.1.0"

__all__ = [
    "config",
    "Config",
    "KTNParams",
    "PlateMaterial",
    "flexural_rigidity",
    "CaseError",
    "Problem",
    "__version__",
]
