r"""benchmarks.py — независимые эталоны большого прогиба (теория Кармана).

Единый источник эталонных чисел для тестов (``tests/test_karman.py``) и
ноутбука (``notebooks/06_theory_comparison.ipynb``): числа заданы ЧИСТЫМИ
формулами либо замороженными таблицами со ссылкой на первоисточник в
docstring — в коде тестов таблиц НЕТ. Эталоны подобраны РАЗНОЙ
математической природы (точное аналитическое решение мембраны, степенные
ряды, энергия Ритца, ряды Фурье, МКЭ/IGA): их взаимные совпадения дают
невырожденную валидацию (§7, §9 ТЗ).

Нормировки (приложение B ТЗ)
---------------------------
Безразмерная нагрузка ``P̄ ≡ p a⁴/(E h⁴)`` (круг: ``a`` — радиус; квадрат:
``a`` — сторона); безразмерный прогиб — ``w/h`` (кроме Hencky, где удобна
``w/a``); жёсткость ``D = E h³/[12(1-ν²)]``. Пересчёт:

.. math:: \bar P = \frac{p a^4}{E h^4}
          = \frac{1}{12(1-\nu^2)}\,\frac{p a^4}{D h}.

⚠️ РИСК №1 (§13): разные эталоны используют разные ν (0.30 у
Way/Тимошенко/Hencky, 0.316 у Levy) — каждая функция ЯВНО принимает ν;
эталоны с разными ν не смешивать в одном допуске.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
#  Приложение B — пересчёт нормировок
# --------------------------------------------------------------------------- #
def pbar(p: float, a: float, E: float, h: float) -> float:
    r"""Безразмерная нагрузка ``P̄ = p a⁴/(E h⁴)`` (прил. B)."""
    return p * a**4 / (E * h**4)


def pbar_to_pa4_over_Dh(P_bar: float, nu: float) -> float:
    r"""``p a⁴/(D h) = 12(1-ν²)·P̄`` (прил. B): переход к нормировке жёсткости."""
    return 12.0 * (1.0 - nu**2) * P_bar


def pbar_to_pa4_over_64Dh(P_bar: float, nu: float) -> float:
    r"""``p a⁴/(64 D h) = [12(1-ν²)/64]·P̄`` (прил. B; для круга ν=0.3 → 0.1706·P̄)."""
    return pbar_to_pa4_over_Dh(P_bar, nu) / 64.0


# --------------------------------------------------------------------------- #
#  Gate L — линейные (Кирхгоф) наклоны w/h ~ P̄ при w/h ≪ 1
# --------------------------------------------------------------------------- #
def _kirchhoff_wh(alpha: float, nu: float, P_bar: float) -> float:
    r"""``w/h`` линейной теории по коэффициенту ``α`` из ``w = α·p a⁴/D``.

    .. math:: w/h = \alpha\,\frac{p a^4}{D h} = \alpha\,12(1-\nu^2)\,\bar P.
    """
    return alpha * pbar_to_pa4_over_Dh(P_bar, nu)


def kirchhoff_clamped_circle(P_bar: float, nu: float = 0.3) -> float:
    r"""Защемлённый круг, центр: ``w_0/h = p a⁴/(64 D h)`` (``α = 1/64``)."""
    return _kirchhoff_wh(1.0 / 64.0, nu, P_bar)


def kirchhoff_hinge_circle(P_bar: float, nu: float = 0.3) -> float:
    r"""Шарнирный круг, центр: ``α = (5+ν)/[(1+ν)·64]`` (ν=0.3 → коэф. 4.077)."""
    return _kirchhoff_wh((5.0 + nu) / ((1.0 + nu) * 64.0), nu, P_bar)


def kirchhoff_clamped_square(P_bar: float, nu: float = 0.316) -> float:
    r"""Защемлённый квадрат, центр: ``w_c = 0.001263·p a⁴/D`` (Levy, ν=0.316)."""
    return _kirchhoff_wh(0.001263, nu, P_bar)


def kirchhoff_hinge_square(P_bar: float, nu: float = 0.3) -> float:
    r"""Шарнирный квадрат (Навье), центр: ``w_max = 0.00406·q a⁴/D`` (ν=0.3)."""
    return _kirchhoff_wh(0.00406, nu, P_bar)


# --------------------------------------------------------------------------- #
#  A.1 Hencky — круглая мембрана, защемлённая, immovable, предел D→0 (ν=0.3)
# --------------------------------------------------------------------------- #
#: ведущий коэффициент прогиба центра мембраны Hencky (ν=0.3), ВОСПРОИЗВОДИМ.
HENCKY_W_COEFF = 0.653          # w_0/a = 0.653·(p a/(E h))^{1/3}
HENCKY_SIGMA_COEFF = 0.431      # σ_r(0) = 0.431·E·(p a/(E h))^{2/3}


def hencky_center_deflection(P_bar: float, nu: float = 0.3) -> float:
    r"""Центральный прогиб круглой мембраны Hencky (предел ``D→0``, §A.1).

    В нормировке пластины ``w_0/h = 0.653·P̄^{1/3}`` (ν=0.3). Пластина ЖЁСТЧЕ
    мембраны, поэтому решатель подходит к этой асимптоте СНИЗУ с ростом ``P̄``.
    Число 0.653 воспроизводимо (ведущий коэф. ряда ``b_0 = 1.7244``).
    Источник: H. Hencky, Z. Math. Phys. 63 (1915) 311–317; уточнение
    W. B. Fichter, NASA TP-3658 (1997).
    """
    if abs(nu - 0.3) > 1e-9:
        raise ValueError("hencky_center_deflection: коэффициент 0.653 табулирован "
                         "для ν=0.3 (см. §A.1); иное ν — направление развития")
    return HENCKY_W_COEFF * P_bar ** (1.0 / 3.0)


# --------------------------------------------------------------------------- #
#  A.2 Way — круглая, защемлённая, immovable (ν=0.3), ряды. Табл. 3.
# --------------------------------------------------------------------------- #
#: (P̄, w_0/h); источник: S. Way, Trans. ASME 56 (1934), APM-56-12, 627–636.
_WAY_CLAMPED_CIRCLE = (
    (1.818, 0.296), (3.196, 0.482), (4.561, 0.637),
    (6.321, 0.800), (8.635, 0.970), (11.71, 1.152),
)


def way_clamped_circle() -> np.ndarray:
    r"""Таблица Way (ряды): столбцы ``[P̄, w_0/h]`` (защемлённый круг, ν=0.3, §A.2).

    Пример из первоисточника: ``P̄=10 → w_0/h=1.055``. Решатель (многомодовый)
    обязан ложиться на ряды Way ТОЧНЕЕ, чем на одночлен Тимошенко.
    """
    return np.array(_WAY_CLAMPED_CIRCLE, dtype=float)


# --------------------------------------------------------------------------- #
#  A.3 Тимошенко — круглая, защемлённая, immovable, одночлен. ВОСПРОИЗВОДИМ.
# --------------------------------------------------------------------------- #
#: кубический коэффициент ``w/h + k(ν)(w/h)³ = p a⁴/(64 D h)``.
_TIMOSHENKO_K = {0.3: 0.488, 0.25: 0.477}


def timoshenko_clamped_circular(w_over_h: float, nu: float = 0.3) -> float:
    r"""Прямая формула Тимошенко: по ``w_0/h`` → ``P̄`` (защемлённый круг, §A.3).

    .. math:: w_0/h + k(\nu)\,(w_0/h)^3 = \frac{p a^4}{64 D h}
              \;\Longleftrightarrow\;
              \bar P = \frac{64}{12(1-\nu^2)}\big[(w_0/h) + k\,(w_0/h)^3\big].

    ``k=0.488`` (ν=0.3), ``0.477`` (ν=0.25). Для ν=0.3 →
    ``P̄ = 5.86(w_0/h) + 2.86(w_0/h)³`` (сверка с Way: ``w_0/h=0.8 → P̄=6.15``
    vs Way 6.321, ~2.7 %). Источник: Timoshenko & Woinowsky-Krieger, Theory
    of Plates and Shells, 2nd ed., McGraw-Hill (1959), гл. IX; константа —
    Way, Author's Closure, Trans. ASME 56 (1934) 636.
    """
    if nu not in _TIMOSHENKO_K:
        raise ValueError(f"timoshenko: k(ν) табулирован для ν ∈ {sorted(_TIMOSHENKO_K)}")
    k = _TIMOSHENKO_K[nu]
    scale = 64.0 / (12.0 * (1.0 - nu**2))
    x = float(w_over_h)
    return scale * (x + k * x**3)


def timoshenko_clamped_circular_inverse(P_bar: float, nu: float = 0.3) -> float:
    r"""Обратная формула Тимошенко: по ``P̄`` → ``w_0/h`` (единственный корень > 0)."""
    if nu not in _TIMOSHENKO_K:
        raise ValueError(f"timoshenko: k(ν) табулирован для ν ∈ {sorted(_TIMOSHENKO_K)}")
    k = _TIMOSHENKO_K[nu]
    scale = 64.0 / (12.0 * (1.0 - nu**2))
    # scale·(x + k x³) = P̄  ⇒  k x³ + x − P̄/scale = 0
    roots = np.roots([k * scale, 0.0, scale, -float(P_bar)])
    real = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 0]
    return float(min(real))                                  # единственный положительный


# --------------------------------------------------------------------------- #
#  A.4 Levy — квадрат, защемлённый, immovable (ν=0.316); кросс-чек IGA.
# --------------------------------------------------------------------------- #
#: (P̄, w_c/h, σ_x a²/(E h²) в центре); Levy NACA TN 847 (Report 740, 1942),
#: кросс-чек — L. V. Tran, J. Lee et al., arXiv:1411.3508, Табл. 1.
_LEVY_SQUARE_CLAMPED = (
    (17.8, 0.237, 2.6), (38.3, 0.471, 5.2), (63.4, 0.695, 8.0),
    (95.0, 0.912, 11.1), (134.9, 1.121, 13.3), (184.0, 1.323, 15.9),
    (245.0, 1.521, 19.2), (318.0, 1.714, 21.9), (402.0, 1.902, 25.1),
)


def levy_square_clamped() -> np.ndarray:
    r"""Таблица Levy (ряды Фурье): ``[P̄, w_c/h, σ_x a²/(E h²)]`` (квадрат, §A.4).

    Защемлённый квадрат, immovable, ν=0.316; столбец прогиба кросс-проверен
    IGA (Levy↔IGA < 1 %). Пример отчёта: ``P̄=320 → w_c/h=1.72``.
    """
    return np.array(_LEVY_SQUARE_CLAMPED, dtype=float)


# --------------------------------------------------------------------------- #
#  A.5 Levy — квадрат, шарнирный, immovable (ν=0.316).
# --------------------------------------------------------------------------- #
#: секция «Edge Displacement Zero» (immovable!): P̄≈278.5 → w_c/h≈1.83.
LEVY_SQUARE_SS_IMMOVABLE = (278.5, 1.83)
#: ⚠️ секция «Edge Compression Zero» (movable) — для контраста Gate B:
#: P̄=247 → w_c/h≈2.72 (≈ 25 % больше). НЕ путать с immovable выше.
LEVY_SQUARE_SS_MOVABLE = (247.0, 2.72)


def levy_square_ss_immovable() -> tuple[float, float]:
    r"""Точка Levy: шарнирный квадрат, immovable — ``(P̄, w_c/h) = (278.5, 1.83)``.

    Секция «Edge Displacement Zero» (immovable), между 1.827 и 1.846. НЕ путать
    с «Edge Compression Zero» (movable): там ``P̄=247 → w_c/h≈2.70–2.745``
    (≈ 25 % больше), см. :data:`LEVY_SQUARE_SS_MOVABLE`. Источник: S. Levy,
    NACA TN 846 (= Report 737, 1942).
    """
    return LEVY_SQUARE_SS_IMMOVABLE


__all__ = [
    "pbar",
    "pbar_to_pa4_over_Dh",
    "pbar_to_pa4_over_64Dh",
    "kirchhoff_clamped_circle",
    "kirchhoff_hinge_circle",
    "kirchhoff_clamped_square",
    "kirchhoff_hinge_square",
    "hencky_center_deflection",
    "way_clamped_circle",
    "timoshenko_clamped_circular",
    "timoshenko_clamped_circular_inverse",
    "levy_square_clamped",
    "levy_square_ss_immovable",
    "HENCKY_W_COEFF",
    "HENCKY_SIGMA_COEFF",
    "LEVY_SQUARE_SS_IMMOVABLE",
    "LEVY_SQUARE_SS_MOVABLE",
]
