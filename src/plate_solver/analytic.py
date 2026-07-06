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
#  Кольцо b < r < a, равномерная нагрузка
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

    Эталон: общее решение с константами из :func:`_annulus_coeffs`;
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


# --------------------------------------------------------------------------- #
#  Точечная сила P в центре круга (формулы НЕ выводятся заново)
# --------------------------------------------------------------------------- #
def circle_point_clamped(r, a: float, P: float, D: float):
    r"""Прогиб защемлённого круга под центральной силой P (Кирхгоф, Тимошенко).

    .. math:: w(r) = \frac{P}{16\pi D}\big[a^2 - r^2 + 2 r^2 \ln(r/a)\big],
              \qquad w(0) = \frac{P a^2}{16\pi D}.

    Вне точки r = 0 бигармоника однородна (сила — δ в центре);
    ``r² ln r → 0`` при r → 0, поэтому в центре значение конечно.
    """
    r = np.asarray(r, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(r > 0.0, 2.0 * r**2 * np.log(r / a), 0.0)
    return P / (16.0 * np.pi * D) * (a**2 - r**2 + term)


def circle_point_clamped_wmax(a: float, P: float, D: float) -> float:
    """Максимальный прогиб (центр): ``w(0) = P a² / (16 π D)``."""
    return P * a**2 / (16.0 * np.pi * D)


def circle_point_soft_moment(r, a: float, P: float):
    r"""Поле M расщепления под центральной силой: ``M(r) = (P/2π)·ln(a/r)``.

    Решение (P1): ``−ΔM = P·δ`` с ``M(a) = 0`` — логарифм точечного источника.
    """
    r = np.asarray(r, float)
    with np.errstate(divide="ignore"):
        return P / (2.0 * np.pi) * np.log(a / r)


def circle_point_soft(r, a: float, P: float, D: float):
    r"""Прогиб «мягкого шарнира» (модель расщепления) под центральной силой P.

    Из (P2) ``−Δw = M/D`` с ``M = (P/2π)ln(a/r)`` и ``w(a) = 0``:

    .. math:: w(r) = \frac{P}{8\pi D}\big[a^2 - r^2(1 + \ln(a/r))\big],
              \qquad w(0) = \frac{P a^2}{8\pi D}.

    Контроль (тот же паттерн, что NOTES §8 для равномерной нагрузки):
    совпадает с пределом ν → 1 формулы Тимошенко для опёртой пластины
    ``w = P/(16πD)[(3+ν)/(1+ν)(a²−r²) + 2r²ln(r/a)]``.
    """
    r = np.asarray(r, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(r > 0.0, r**2 * (1.0 + np.log(a / r)), 0.0)
    return P / (8.0 * np.pi * D) * (a**2 - term)


def circle_point_soft_wmax(a: float, P: float, D: float) -> float:
    """Максимальный прогиб (центр) мягкого шарнира: ``w(0) = P a² / (8 π D)``."""
    return P * a**2 / (8.0 * np.pi * D)




# --------------------------------------------------------------------------- #
#  Прямоугольник: Навье (SSSS) и Леви (SCSC)
# --------------------------------------------------------------------------- #
def navier_rect_uniform(x, y, x1: float, x2: float, y1: float, y2: float,
                        q: float, D: float, tol: float = 1e-12):
    r"""Ряд Навье для SSSS-прямоугольника [x1,x2]×[y1,y2] с КОНТРОЛЕМ остатка.

    .. math:: w = \\frac{16 q}{\\pi^6 D} \\sum_{m,n\\ нечёт.}
              \\frac{\\sin(m\\pi\\xi/L_x)\\,\\sin(n\\pi\\eta/L_y)}
                   {mn\\,[(m/L_x)^2 + (n/L_y)^2]^2}

    Число членов удваивается, пока изменение поля не станет < tol·|w|
    (мажоранта хвоста ~1/M⁴ гарантирует сходимость контроля).
    """
    from .ladder import navier_uniform

    Lx, Ly = x2 - x1, y2 - y1
    X = np.asarray(x, float) - x1
    Y = np.asarray(y, float) - y1
    n_terms = 25
    w_prev = navier_uniform(X, Y, Lx, Ly, D, q, n_terms=n_terms)
    w = w_prev
    while n_terms <= 1600:
        n_terms *= 2
        w = navier_uniform(X, Y, Lx, Ly, D, q, n_terms=n_terms)
        scale = float(np.max(np.abs(w))) or 1.0
        if float(np.max(np.abs(w - w_prev))) < tol * scale:
            return w
        w_prev = w
    return w


def levy_rect_uniform(x, y, x1: float, x2: float, y1: float, y2: float,
                      q: float, D: float, n_terms: int = 60):
    r"""Ряд Леви: x-края шарнир (hinge), y-края защемление (clamped).

    .. math:: w = \\sum_{m\\ нечёт.} Y_m(\\eta)\\,
              \\sin(\\alpha_m (x - x_1)), \\qquad \\alpha_m = m\\pi/L_x,

    где η = y − (y₁+y₂)/2; Y_m — решение ОДУ четвёртого порядка
    Y⁗ − 2α²Y″ + α⁴Y = q_m/D (q_m = 4q/(πm)) с Y(±L_y/2) = Y′(±L_y/2) = 0:
    частное q_m/(Dα⁴) плюс ЧЁТНАЯ однородная часть A·ch(αη) + B·η·sh(αη);
    A, B — из 2×2 (симметрия по η). Sympy-проверка ОДУ и КУ —
    tests/test_mixed_bc.py.
    """
    Lx, Ly = x2 - x1, y2 - y1
    X = np.asarray(x, float) - x1
    eta = np.asarray(y, float) - 0.5 * (y1 + y2)
    c = Ly / 2.0
    out = np.zeros(np.broadcast(X, eta).shape)
    for m in range(1, 2 * n_terms, 2):
        al = m * np.pi / Lx
        qm = 4.0 * q / (np.pi * m)
        yp = qm / (D * al**4)
        ch, sh = np.cosh(al * c), np.sinh(al * c)
        a11, a12, b1 = ch, c * sh, -yp
        a21, a22, b2 = al * sh, sh + al * c * ch, 0.0
        det = a11 * a22 - a12 * a21
        aa = (b1 * a22 - a12 * b2) / det
        bb = (a11 * b2 - b1 * a21) / det
        ym = yp + aa * np.cosh(al * eta) + bb * eta * np.sinh(al * eta)
        out = out + ym * np.sin(al * X)
    return out


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
    "circle_point_clamped",
    "circle_point_clamped_wmax",
    "circle_point_soft",
    "circle_point_soft_moment",
    "circle_point_soft_wmax",
    "navier_rect_uniform",
    "levy_rect_uniform",
]
