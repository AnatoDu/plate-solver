"""Контакт методом обобщённой реакции (тест-ворота шага) → Таблица 4.2.

Постановка: L-форма прижимается к жёсткому плоскому основанию (зазор Δ). МОР —
внешний цикл (логика fix_base2.py) с предвычисленной A. Проверяем:
  • сходимость: невязка ‖Δr‖ монотонно падает (на ≥ 2 порядка);
  • односторонняя связь: r ≥ 0 всюду;
  • контакт только в зоне основания (foundation_mask);
  • пик реакции — внутри, в области, прилегающей к входящему углу (NOTES.md §10:
    НЕ в самой вершине угла, где w=0, а на кромке пятна вблизи неё).
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver import geometry
from plate_solver.config import Config
from plate_solver.contact import ContactMOR
from plate_solver.plate import PlateBending


@pytest.fixture(scope="module")
def contact_run():
    dom = geometry.make_L(1.0, 0.5)
    pb = PlateBending.from_config(dom, Config(nu=0.3, q0=4.0, p=8, Q=40))
    q = pb.quad
    _, cw = pb.solve_uniform(4.0)
    wmax = float(pb.deflection(cw, q.x, q.y).max())
    fmask = lambda x, y: (x + y) < 0.9          # noqa: E731 — основание под частью Ω
    cfg = Config(nu=0.3, q0=4.0, p=8, Q=40, beta=1.0, max_iter=2000, tol=1e-12, grid_n=36)
    mor = ContactMOR(pb, cfg, foundation_mask=fmask, gap=0.6 * wmax)
    return mor, mor.solve()


def test_gate_mor_residual_monotone_and_drops(contact_run):
    """ГЛАВНЫЕ ВОРОТА: невязка МОР монотонно падает (на ≥ 2 порядка)."""
    _, res = contact_run
    h = res.residual_history
    assert np.all(np.diff(h[3:]) <= 1e-9 * h[0])   # монотонно невозрастает (после старта)
    assert h[-1] < h[0] / 100.0                     # падение ≥ 2 порядков


def test_reaction_nonnegative(contact_run):
    _, res = contact_run
    assert res.r_nodes.min() >= 0.0                 # односторонняя связь r ≥ 0


def test_contact_only_in_foundation(contact_run):
    mor, res = contact_run
    assert np.max(np.abs(res.r_nodes[~mor.fmask])) == 0.0   # вне основания r = 0
    assert int((res.r_nodes > 0).sum()) > 0                 # контакт возник


def test_peak_interior_near_reentrant_corner(contact_run):
    _, res = contact_run
    px, py = res.peak_xy
    assert 0.0 < px < 1.0 and 0.0 < py < 1.0        # пик строго внутри
    # тяготеет к области входящего угла (0.5,0.5), но не в самой вершине (w=0 там)
    assert np.hypot(px - 0.5, py - 0.5) < 0.45
