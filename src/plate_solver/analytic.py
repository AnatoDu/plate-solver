r"""analytic.py — точные эталоны для верификации (reuse + замыкание Пуассона).

Две группы эталонов:

1. **Изгиб круглой пластины** (теория Кирхгофа) — классические формулы
   Тимошенко («Пластинки и оболочки»), радиус ``a``, жёсткость ``D``:
       защемление:        w = q (a² − r²)² / (64 D),  w_max = q a⁴ / (64 D)
       шарнир (опирание):  w = q/(64 D)(a² − r²)[(5+ν)/(1+ν) a² − r²]
   Для нашей постановки «мягкого шарнира» эталон — шарнирно опёртая пластина.

2. **Задача Пуассона на диске** — нужна как промежуточная сверка решателя
   ``poisson.py`` (каждый из кирпичей (P1)/(P2) — это −Δv = f, v|∂Ω = 0).
   Для постоянной правой части ``−Δv = c`` на круге радиуса ``a``:

   .. math:: v(r) = \frac{c\,(a^2 - r^2)}{4}, \qquad v(0) = \frac{c\,a^2}{4}.

   (Осесимметрия: ``−(1/r)(r v')' = c`` ⇒ ``v = c(a²−r²)/4`` при ``v(a)=0`` и
   ограниченности в нуле.)
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
#  Изгиб круглой пластины (Кирхгоф; бывший plate_solver.analytic.circular)
# --------------------------------------------------------------------------- #


def clamped_uniform(r, a: float, q: float, D: float):
    """Прогиб защемлённой круглой пластины (равномерная нагрузка)."""
    r = np.asarray(r, float)
    return q * (a**2 - r**2) ** 2 / (64.0 * D)


def clamped_uniform_wmax(a: float, q: float, D: float) -> float:
    """Максимальный прогиб (в центре) защемлённой круглой пластины."""
    return q * a**4 / (64.0 * D)


def simply_supported_uniform(r, a: float, q: float, D: float, nu: float):
    """Прогиб шарнирно опёртой круглой пластины (равномерная нагрузка)."""
    r = np.asarray(r, float)
    k = (5.0 + nu) / (1.0 + nu)
    return q / (64.0 * D) * (a**2 - r**2) * (k * a**2 - r**2)


def simply_supported_uniform_wmax(a: float, q: float, D: float, nu: float) -> float:
    """Максимальный прогиб (в центре) шарнирно опёртой круглой пластины."""
    return (5.0 + nu) / (1.0 + nu) * q * a**4 / (64.0 * D)


def disk_poisson_uniform(r, a: float, c: float = 1.0):
    r"""Решение ``−Δv = c`` на круге радиуса ``a``, ``v|_{r=a} = 0``.

    .. math:: v(r) = c\,(a^2 - r^2)/4.
    """
    r = np.asarray(r, float)
    return c * (a**2 - r**2) / 4.0


def disk_poisson_uniform_center(a: float, c: float = 1.0) -> float:
    """Значение в центре ``v(0) = c·a²/4`` (контроль знака и масштаба решателя)."""
    return c * a**2 / 4.0


def disk_poisson_unit(X, Y):
    r"""Эталон для test_poisson_disk: ``−Δv = 1`` на единичном круге.

    .. math:: v(x, y) = (1 - x^2 - y^2)/4.

    Принимает декартовы координаты (X, Y) — удобно сверять прямо в узлах
    квадратуры/на сетке, без перехода к ρ.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    return (1.0 - X**2 - Y**2) / 4.0


def circular_plate_clamped(rho, q: float, a: float, D: float):
    r"""Формула (4.1): прогиб защемлённой круглой пластины (равномерная q).

    .. math:: w(\rho) = q\,(a^2 - \rho^2)^2 / (64 D).
    """
    return clamped_uniform(rho, a, q, D)


def circular_plate_simply_supported(rho, q: float, a: float, nu: float, D: float):
    r"""Формула (4.2): прогиб шарнирно опёртой круглой пластины (равномерная q).

    .. math:: w(\rho) = \frac{q}{64 D}(a^2 - \rho^2)
              \Big[\frac{5+\nu}{1+\nu} a^2 - \rho^2\Big].
    """
    return simply_supported_uniform(rho, a, q, D, nu)


def circular_plate_soft_hinge(rho, q: float, a: float, D: float):
    r"""ТОЧНОЕ решение расщепления «мягкого шарнира» (M=0 на ∂Ω) на круге.

    Из (P1) ``−ΔM=q, M|_{r=a}=0`` ⇒ ``M=q(a²−ρ²)/4``; из (P2) ``−Δw=M/D`` ⇒

    .. math:: w(\rho) = \frac{q}{64 D}(a^2 - \rho^2)(3a^2 - \rho^2).

    Это формула (4.2) с множителем ``(5+ν)/(1+ν)``, заменённым на 3 (т.е. (4.2)
    при ν=1). Эталон для test_plate_circle: проверяет ЧИСЛЕННЫЙ метод против
    аналитики РЕАЛИЗОВАННОЙ модели. Расхождение с (4.2) при ν≠1 — модельная
    погрешность мягкого шарнира на криволинейной границе, ∝ (1−ν)·кривизна;
    отношение w_soft/w_SS = 3(1+ν)/(5+ν) (NOTES.md §8).
    """
    rho = np.asarray(rho, float)
    return q * (a**2 - rho**2) * (3.0 * a**2 - rho**2) / (64.0 * D)


def circular_plate_soft_hinge_wmax(q: float, a: float, D: float) -> float:
    """Максимальный прогиб (центр) мягкого шарнира на круге: ``3 q a⁴ / (64 D)``."""
    return 3.0 * q * a**4 / (64.0 * D)


# --------------------------------------------------------------------------- #
#  Кольцо b < r < a, равномерная нагрузка (P3.2 фазы 2)
# --------------------------------------------------------------------------- #
ANNULUS_BCS = ("clamped", "soft", "true_ss")


def _annulus_coeffs(a: float, b: float, q: float, D: float, bc: str, nu: float) -> np.ndarray:
    r"""Константы C1..C4 общего осесимметричного решения бигармоники на кольце.

    .. math:: w(r) = \frac{q r^4}{64 D} + C_1 + C_2 r^2
              + C_3 \ln\frac{r}{a} + C_4 r^2 \ln\frac{r}{a}

    (каждое слагаемое однородной части бигармонично). Краевые условия на
    r = a и r = b (по два на край) дают систему 4×4 (numpy.linalg.solve):

    * ``clamped``: w = 0, w' = 0;
    * ``soft`` (модель расщепления, «мягкий шарнир»): w = 0, Δw = 0;
    * ``true_ss`` (истинное опирание Кирхгофа, только для model_gap):
      w = 0, M_r = −D (w'' + ν w'/r) = 0.
    """
    if not 0.0 < b < a:
        raise ValueError("Кольцо требует 0 < b < a.")
    if bc not in ANNULUS_BCS:
        raise ValueError(f"Неизвестное закрепление {bc!r}; ожидается {ANNULUS_BCS}.")
    rows: list[list[float]] = []
    rhs: list[float] = []
    for r0 in (a, b):
        L = np.log(r0 / a)
        # строка w: [1, r², ln(r/a), r² ln(r/a)] и частное решение q r⁴/(64D)
        rows.append([1.0, r0**2, L, r0**2 * L])
        rhs.append(-q * r0**4 / (64.0 * D))
        if bc == "clamped":                      # w' = 0
            rows.append([0.0, 2.0 * r0, 1.0 / r0, 2.0 * r0 * L + r0])
            rhs.append(-q * r0**3 / (16.0 * D))
        elif bc == "soft":                       # Δw = w'' + w'/r = 0
            rows.append([0.0, 4.0, 0.0, 4.0 * L + 4.0])
            rhs.append(-q * r0**2 / (4.0 * D))
        else:                                    # true_ss: w'' + ν w'/r = 0
            rows.append([0.0, 2.0 * (1.0 + nu), (nu - 1.0) / r0**2,
                         (2.0 * L + 3.0) + nu * (2.0 * L + 1.0)])
            rhs.append(-(12.0 + 4.0 * nu) * q * r0**2 / (64.0 * D))
    return np.linalg.solve(np.array(rows), np.array(rhs))


def annulus_uniform(r, a: float, b: float, q: float, D: float,
                    bc: str = "clamped", nu: float = 0.3):
    r"""Прогиб кольца b < r < a под равномерной нагрузкой q (Кирхгоф, осесимметрия).

    Эталон P3.2: общее решение с константами из :func:`_annulus_coeffs`;
    ``bc = clamped | soft | true_ss`` (см. там же). Для ``soft`` эталон
    модельно-согласован с расщеплением (NOTES §8): на кольце обе границы
    криволинейны, поэтому soft ≠ true_ss при ν ≠ 1.
    """
    C = _annulus_coeffs(a, b, q, D, bc, nu)
    r = np.asarray(r, float)
    L = np.log(r / a)
    return q * r**4 / (64.0 * D) + C[0] + C[1] * r**2 + C[2] * L + C[3] * r**2 * L


def annulus_uniform_wmax(a: float, b: float, q: float, D: float,
                         bc: str = "clamped", nu: float = 0.3) -> float:
    """Максимальный |w| кольца (плотная выборка по радиусу; максимум не в центре)."""
    r = np.linspace(b, a, 4001)
    return float(np.max(np.abs(annulus_uniform(r, a, b, q, D, bc, nu))))


__all__ = [
    "clamped_uniform",
    "clamped_uniform_wmax",
    "simply_supported_uniform",
    "simply_supported_uniform_wmax",
    "disk_poisson_uniform",
    "disk_poisson_uniform_center",
    "disk_poisson_unit",
    "circular_plate_clamped",
    "circular_plate_simply_supported",
    "circular_plate_soft_hinge",
    "circular_plate_soft_hinge_wmax",
    "ANNULUS_BCS",
    "annulus_uniform",
    "annulus_uniform_wmax",
]
