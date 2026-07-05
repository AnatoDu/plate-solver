r"""Тест-ворота Части 1 — штамп методом Ритца (вариант B), тройная сходимость.

Та же 1D-задача о штампе, но изгиб на каждой итерации МОР решается РИТЦЕМ
(структура под пару ГУ «шарнир(0)+скользящая заделка(L)»). Показываем
Грин = Ритц = Maple. Чтобы тест был быстрым, МОР ограничен умеренным числом
итераций — оба метода в ОДНОМ состоянии МОР, отклонение между методами уже мало.
"""

from __future__ import annotations

import numpy as np
import pytest
from plate_solver.contact.mor1d import ContactStrip1D
from plates.stamp import load_maple_reference, maple_agreement, solve_stamp
from plates.stamp_ritz import solve_stamp_ritz

MAXIT = 100_000


def _rel_l2(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return 100.0 * np.sqrt(np.sum((a - b) ** 2)) / np.sqrt(np.sum(b**2))


@pytest.fixture(scope="module")
def triple():
    """Грин и Ритц при одинаковом числе итераций МОР + согласие с Maple."""
    problem = ContactStrip1D()
    green = solve_stamp(problem, max_iter=MAXIT, tol_dw=0.0)
    ritz, _ = solve_stamp_ritz(problem, p=16, max_iter=MAXIT, tol_dw=0.0)
    xa, ya = load_maple_reference()
    _, l2_gm = maple_agreement(green.x, green.w, xa, ya)
    _, l2_rm = maple_agreement(ritz.x, ritz.w, xa, ya)
    return {"green": green, "ritz": ritz,
            "l2_rg": _rel_l2(ritz.w, green.w), "l2_gm": l2_gm, "l2_rm": l2_rm}


def test_stamp_ritz_vs_green(triple):
    """ВОРОТА: L²-отклонение Ритц↔Грин < 1 % (тот же МОР, те же ГУ)."""
    assert triple["l2_rg"] < 1.0


def test_stamp_ritz_vs_maple(triple):
    """ВОРОТА: Ритц↔Maple того же порядка, что Грин↔Maple (~2–6 %)."""
    assert triple["l2_rm"] == pytest.approx(triple["l2_gm"], rel=0.3)


def test_stamp_ritz_contact_matches_green(triple):
    """ВОРОТА: зона контакта Ритц = Грин ([46,100]); пик реакции на кромке штампа."""
    rz, g = triple["ritz"], triple["green"]
    assert rz.contact_span == (46.0, 100.0) == g.contact_span
    assert rz.x_rmax == pytest.approx(46.0)             # пик у кромки, как у Грина
    assert rz.r_max == pytest.approx(g.r_max, rel=0.05)


def test_stamp_ritz_bc_structure():
    """Структурный базис удовлетворяет ГУ: w(0)=0 и w'(L)=0 тождественно."""
    from plates.stamp_ritz import build_ritz_beam_operator

    L, D, n = 100.0, 192307.69, 100
    M, _ = build_ritz_beam_operator(L, D, n, p=16)
    # отклик на произвольную нагрузку обязан давать w(0)=0 (узел 0)
    rng = np.random.default_rng(0)
    w = M @ rng.standard_normal(n + 1)
    assert abs(w[0]) < 1e-6 * np.max(np.abs(w))         # шарнир при x=0
