r"""Тест-ворота Части 1 — 1D задача о ШТАМПЕ (метод обобщённой реакции).

Проверяем:
  • точное ВОСПРОИЗВЕДЕНИЕ ``fix_base2.py`` при тех же настройках (50000 итераций,
    β=0.01): w_max, зона контакта и сам решатель совпадают с каноническим
    ``plate_solver.contact.mor1d.solve_mor_1d`` — физика и числа сохранены;
  • СОГЛАСИЕ с аналитикой Maple: отн. L²-отклонение мало (единицы %), и УБЫВАЕТ с
    числом итераций (МОР сходится именно к аналитическому решению);
  • рисунок ``stamp_1d.png`` создаётся и читаем.

Замечание: в ``fix_base2.py`` фиксированы 50000 итераций — это НЕДОсходимость
(пик реакции у кромки жёсткого штампа досходится медленно). Поэтому 50000-шаговый
прогон воспроизводится точно (контроль метода), а согласие с Maple снимается на
сошедшемся решении (критерий ``max|Δw|``), где L² минимально.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from plate_solver.contact.mor1d import ContactStrip1D, solve_mor_1d
from plates.stamp import load_maple_reference, maple_agreement, solve_stamp


# --------------------------------------------------------------------------- #
#  Воспроизведение fix_base2.py
# --------------------------------------------------------------------------- #
def test_stamp_reproduces_fix_base2():
    """ВОРОТА: 50000 итераций воспроизводят прямой прогон fix_base2.py."""
    res = solve_stamp(ContactStrip1D(), max_iter=50_000, tol_dw=0.0)
    # документированный результат fix_base2.py (50000 шагов, β=0.01):
    assert res.iters == 50_000 and not res.converged
    assert res.w_max == pytest.approx(1.41452, abs=1e-3)
    assert res.x_wmax == pytest.approx(27.0)
    assert res.contact_span == (46.0, 100.0)
    assert res.n_contact == 40


def test_stamp_matches_canonical_solver():
    """Тот же решатель, что и переиспользуемый ``solve_mor_1d`` (как есть)."""
    res = solve_stamp(ContactStrip1D(), max_iter=50_000, tol_dw=0.0)
    x2, w2, r2 = solve_mor_1d(ContactStrip1D(max_iter=50_000, tol=0.0))
    assert np.allclose(res.w, w2, atol=1e-9)
    assert np.allclose(res.r, r2, atol=1e-9)


# --------------------------------------------------------------------------- #
#  Согласие с аналитикой Maple
# --------------------------------------------------------------------------- #
def test_stamp_matches_maple():
    """ВОРОТА: отн. L²-отклонение от Maple мало и УБЫВАЕТ к сошедшемуся решению."""
    xa, ya = load_maple_reference()
    res_a = solve_stamp(ContactStrip1D(), max_iter=50_000, tol_dw=0.0)
    res_b = solve_stamp(ContactStrip1D(), max_iter=500_000, tol_dw=0.0)
    _, l2_a = maple_agreement(res_a.x, res_a.w, xa, ya)
    _, l2_b = maple_agreement(res_b.x, res_b.w, xa, ya)
    assert l2_b < l2_a    # МОР сходится ИМЕННО к аналитике Maple
    assert l2_b < 5.0     # единицы % (факт ~3.3 % при 500k; ~2.4 % к полной сходимости)


def test_stamp_physics_one_sided_contact():
    """Реакция односторонняя (r≥0) и только под штампом (x>m); w→Δ в контакте."""
    p = ContactStrip1D()
    res = solve_stamp(p, max_iter=200_000, tol_dw=0.0)
    assert res.r.min() >= 0.0                                  # r ≥ 0
    assert np.all(res.r[res.x <= p.foundation_start] == 0.0)   # нет реакции вне штампа
    # в зоне контакта прогиб близок к зазору Δ
    assert res.w[res.contact].min() == pytest.approx(p.gap, abs=0.05)
    assert res.x_rmax == pytest.approx(p.foundation_start + 1.0)  # пик у кромки штампа


# --------------------------------------------------------------------------- #
#  Рисунок
# --------------------------------------------------------------------------- #
def test_stamp_png_created(tmp_path):
    """ВОРОТА: ``stamp_1d.png`` создаётся и непустой."""
    from run_stamp_1d import make_figure

    res = solve_stamp(ContactStrip1D(), max_iter=3000, tol_dw=0.0)  # коротко: важен сам файл
    xa, ya = load_maple_reference()
    out = tmp_path / "stamp_1d.png"
    path = make_figure(res, (xa, ya), save=str(out))
    assert os.path.exists(path) and os.path.getsize(path) > 5000
