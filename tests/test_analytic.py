"""Проверка аналитических решений круглой пластины.

Тесты опираются на известные точные свойства (Тимошенко): значение
максимального прогиба, нулевой прогиб на крае, нулевой наклон на
защемлённом крае, монотонность. Это эталон для верификации 2D-метода.
"""

import numpy as np

from plate_solver import analytic

A = 1.0       # радиус
Q = 1.0e4     # нагрузка
NU = 0.3


def _D():
    from plate_solver import flexural_rigidity

    return flexural_rigidity(E=2.0e11, h=0.01, nu=NU)


def test_clamped_wmax_matches_closed_form():
    D = _D()
    w0 = analytic.clamped_uniform(0.0, A, Q, D)
    assert np.isclose(w0, analytic.clamped_uniform_wmax(A, Q, D))
    assert np.isclose(w0, Q * A**4 / (64.0 * D))


def test_clamped_zero_deflection_and_slope_at_edge():
    D = _D()
    # прогиб на крае = 0
    assert np.isclose(analytic.clamped_uniform(A, A, Q, D), 0.0)
    # наклон dw/dr на защемлённом крае = 0 (численная производная)
    r = A - 1e-6
    dwdr = (analytic.clamped_uniform(A, A, Q, D)
            - analytic.clamped_uniform(r, A, Q, D)) / (A - r)
    assert abs(dwdr) < 1e-3 * analytic.clamped_uniform_wmax(A, Q, D)


def test_simply_supported_wmax_matches_timoshenko():
    r"""Шарнирный круг — замкнутая форма Тимошенко.

    .. math:: w_{\max} = \frac{5+\nu}{1+\nu}\,\frac{q a^4}{64 D}

    (в публикации: Timoshenko & Woinowsky-Krieger, Theory of Plates and Shells,
    2-е изд., гл. 3). Проверяется не только значение при ν = 0.3, но и ВЕСЬ
    множитель ``(5+ν)/(1+ν)``: при нескольких ν сверяются и абсолютное
    значение, и отношение к защемлённой пластине ``qa⁴/(64D)`` — ошибка в
    зависимости от ν (частая при переписывании формулы) не проходит.
    """
    D = _D()
    w0 = analytic.simply_supported_uniform(0.0, A, Q, D, NU)
    assert np.isclose(w0, analytic.simply_supported_uniform_wmax(A, Q, D, NU))
    w_clamped = analytic.clamped_uniform_wmax(A, Q, D)
    for nu in (0.0, 0.15, NU, 0.45):
        w_ref = (5.0 + nu) / (1.0 + nu) * Q * A**4 / (64.0 * D)
        w_num = analytic.simply_supported_uniform_wmax(A, Q, D, nu)
        assert np.isclose(w_num, w_ref, rtol=1e-14), (nu, w_num, w_ref)
        # отношение к защемлённой = тот же множитель (при ν=0.3 — 4.0769)
        assert np.isclose(w_num / w_clamped, (5.0 + nu) / (1.0 + nu), rtol=1e-14)
    # шарнирная пластина прогибается сильнее защемлённой
    assert w0 > w_clamped
    # прогиб на крае = 0
    assert np.isclose(analytic.simply_supported_uniform(A, A, Q, D, NU), 0.0)


def test_clamped_profile_monotonic_decreasing():
    D = _D()
    r = np.linspace(0.0, A, 50)
    w = analytic.clamped_uniform(r, A, Q, D)
    assert np.all(np.diff(w) <= 1e-12)  # невозрастает от центра к краю
