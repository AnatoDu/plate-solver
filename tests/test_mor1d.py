"""Тесты МОР для 1D контактной задачи (балка-полоса).

Проверяются:
  • физические инварианты решения (r ≥ 0, зона реакции, комплементарность);
  • верификация функции Грина по точному решению свободной балки и
    первый порядок аппроксимации квадратуры O(1/n);
  • верификация МОР по точному решению Maple (положение на защиту № 4).
"""

import numpy as np
import pytest

from plate_solver.contact.mor1d import ContactStrip1D, solve_mor_1d


@pytest.fixture
def quick_problem():
    """Умеренная сходимость — достаточно для проверки инвариантов (быстро)."""
    return ContactStrip1D(beta=0.02, max_iter=100_000, tol=1e-9)


def test_reaction_nonnegative(quick_problem):
    _, _, r = solve_mor_1d(quick_problem)
    assert np.all(r >= -1e-10)


def test_no_reaction_outside_foundation(quick_problem):
    p = quick_problem
    x, _, r = solve_mor_1d(p)
    outside = x <= p.foundation_start
    assert np.all(r[outside] < 1e-10)


def test_complementarity(quick_problem):
    """Там, где реакция положительна, прогиб должен быть ≈ gap."""
    p = quick_problem
    _, w, r = solve_mor_1d(p)
    active = r > 1e-3
    assert active.any(), "Ожидалась непустая зона контакта"
    assert np.allclose(w[active], p.gap, atol=0.03)


def test_no_contact_when_gap_large():
    """При очень большом зазоре контакта нет — реакция нулевая."""
    p = ContactStrip1D(gap=1e9, max_iter=10, tol=0.0)
    _, _, r = solve_mor_1d(p)
    assert np.all(r < 1e-10)


def test_free_beam_matches_analytic():
    """Без контакта МОР сводится к квадратуре функции Грина.

    Точное решение для балки «шарнир слева — симметрия справа» под
    равномерной нагрузкой:  w(x) = (q₀L⁴/D)(x⁴/24 − x³/6 + x/3),
    максимум в плоскости симметрии:  w(L) = 5 q₀ L⁴ / (24 D).
    Расхождение — лишь погрешность квадратуры (≈0.8 % при n=100).
    """
    p = ContactStrip1D(gap=1e9, n=100, max_iter=10, tol=0.0)
    x, w, r = solve_mor_1d(p)
    assert np.all(r == 0.0)

    w_max_analytic = 5.0 * p.q0 * p.L ** 4 / (24.0 * p.D)
    assert abs(w.max() - w_max_analytic) / w_max_analytic < 0.01
    assert np.isclose(x[w.argmax()], p.L)   # максимум — на правом конце (симметрия)


def test_green_quadrature_first_order():
    """Порядок аппроксимации квадратуры функции Грина: ошибка ~ O(1/n).

    При удвоении числа разбиений ошибка свободной балки уменьшается вдвое.
    """
    def free_err(n: int) -> float:
        p = ContactStrip1D(gap=1e9, n=n, max_iter=10, tol=0.0)
        _, w, _ = solve_mor_1d(p)
        xs = np.linspace(0.0, 1.0, n + 1)
        w_exact = p.q0 * p.L ** 4 / p.D * (xs ** 4 / 24 - xs ** 3 / 6 + xs / 3)
        return np.max(np.abs(w - w_exact)) / w_exact.max()

    e100, e200 = free_err(100), free_err(200)
    assert e200 < e100
    assert 1.7 < e100 / e200 < 2.3   # первый порядок: коэффициент ≈ 2


def test_verification_vs_maple():
    """Верификация МОР по точному решению Maple (положение на защиту № 4).

    Сходимость МОР к эталону — первого порядка O(1/n): дискретизационный
    порог при n=100 ≈ 3.9 % (проверено: 0.47 % при n=800). На 5·10⁵ итерациях
    β-схемы (β=0.02) остаётся ≈5 %. Допуск 6 % отражает этот порог, а не
    подгонку: уточнение сетки/итераций монотонно снижает расхождение.
    """
    from plate_solver.analytic.strip_contact import W_MAPLE, X_MAPLE

    p = ContactStrip1D(beta=0.02, n=100, max_iter=500_000, tol=1e-9)
    x, w, _ = solve_mor_1d(p)

    w_ref = np.interp(x, X_MAPLE, W_MAPLE)
    rel_err = np.max(np.abs(w - w_ref)) / np.max(np.abs(w_ref))
    assert rel_err < 0.06, (
        f"Расхождение МОР vs. Maple = {rel_err:.2%}. Это итерационно-"
        "дискретизационный порог; уточняйте n/число итераций, не допуск."
    )
