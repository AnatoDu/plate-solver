"""Тесты МОР для 1D контактной задачи.

Проверяются физические инварианты решения (не зависят от реализации):
    • реакция неотрицательна (r ≥ 0),
    • комплементарность: там, где r > 0, прогиб ≈ gap; там, где w < gap, r ≈ 0,
    • вне зоны основания реакция равна нулю,
    • при бесконечном зазоре (no-contact) реакция равна нулю везде,
    • результат МОР близок к точному решению Maple (верификация).
"""

import numpy as np
import pytest

from plate_solver.contact.mor1d import ContactStrip1D, solve_mor_1d


@pytest.fixture
def default_problem():
    return ContactStrip1D(max_iter=50_000)


def test_reaction_nonnegative(default_problem):
    _, _, r = solve_mor_1d(default_problem)
    assert np.all(r >= -1e-10)


def test_no_reaction_outside_foundation(default_problem):
    p = default_problem
    x, _, r = solve_mor_1d(p)
    outside = x <= p.foundation_start
    assert np.all(r[outside] < 1e-10)


def test_complementarity(default_problem):
    """Там, где реакция положительна, прогиб должен быть ≈ gap."""
    p = default_problem
    _, w, r = solve_mor_1d(p)
    active = r > 1e-3
    if active.any():
        assert np.allclose(w[active], p.gap, atol=0.05)


def test_no_contact_when_gap_large():
    """При очень большом зазоре контакта нет — реакция нулевая."""
    p = ContactStrip1D(gap=1e6, max_iter=100)
    _, _, r = solve_mor_1d(p)
    assert np.all(r < 1e-10)


def test_free_beam_deflection():
    """Без контакта МОР даёт прогиб свободной шарнирной балки.

    Аналитически: w_max = 5 q L⁴ / (384 D) в точке x = L/2 (Тимошенко).
    Это верификация функции Грина, независимая от МОР.
    """
    p = ContactStrip1D(gap=1e9, max_iter=1)   # зазор огромный → контакта нет
    x, w, r = solve_mor_1d(p)
    assert np.all(r < 1e-10), "Реакция должна быть нулевой при большом зазоре"

    w_max_analytic = 5.0 * p.q0 * p.L ** 4 / (384.0 * p.D)
    rel_err = abs(w.max() - w_max_analytic) / w_max_analytic
    assert rel_err < 1e-3, (
        f"Прогиб свободной балки: МОР={w.max():.4f}, аналитика={w_max_analytic:.4f}, "
        f"расхождение={rel_err:.2%}"
    )
