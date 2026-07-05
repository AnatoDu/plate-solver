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


def test_complementarity_metrics_small_run(contact_run):
    """Метрики Синьорини заполнены и имеют правильный (малый) масштаб.

    Малый прогон останавливается по max_iter=2000 (tol=1e-12 недостижим), поэтому
    метрики отличны от нуля: факт comp_residual ≈ 2.7e-2, gap_overshoot ≈ 2.6e-3.
    Проверяем безразмерный масштаб, а не точную сходимость.
    """
    _, res = contact_run
    assert np.isfinite(res.comp_residual) and res.comp_residual >= 0.0
    assert np.isfinite(res.gap_overshoot)
    assert res.comp_residual < 0.1       # r·(w−Δ) мало́ по сравнению с q0·Δ
    assert abs(res.gap_overshoot) < 0.01  # прогиб в контакте ≈ Δ (доли процента)


# --------------------------------------------------------------------------- #
#  Критерий останова stop="dr"|"comp" (P1.2)
# --------------------------------------------------------------------------- #
def test_stop_comp_certifies_kkt():
    """stop='comp': остановка сертифицирует условия Синьорини с точностью tol.

    KKT-невязка η = max( max|r·(u−Δ)|/(q0·Δ), max(u−Δ)₊/Δ ) — безразмерная;
    факт: tol=5e-2 достигается за ~184 итерации (β=1.0, Q=40). Старт r≡0
    критерий НЕ проходит (проникание u>Δ), поэтому остановка нетривиальна.
    """
    dom = geometry.make_L(1.0, 0.5)
    pb = PlateBending.from_config(dom, Config(nu=0.3, q0=4.0, p=8, Q=40))
    q = pb.quad
    _, cw = pb.solve_uniform(4.0)
    wmax = float(pb.deflection(cw, q.x, q.y).max())
    cfg = Config(nu=0.3, q0=4.0, p=8, Q=40, beta=1.0, max_iter=50_000, tol=5e-2,
                 stop="comp", grid_n=36)
    res = ContactMOR(pb, cfg, gap=0.6 * wmax).solve()
    assert res.converged
    assert 1 < res.iters < 1000                 # не мгновенно и задолго до max_iter
    assert res.comp_residual <= cfg.tol         # комплементарность в допуске
    assert abs(res.gap_overshoot) <= cfg.tol    # проникание в допуске


def test_config_defaults_mor():
    """Дефолты МОР в Config: β безразмерна и внутри условия теоремы 4 (0 < β < 2)."""
    cfg = Config()
    assert cfg.beta == 1.2                      # согласован с золотым конфигом
    assert 0.0 < cfg.beta < 2.0                 # условие сходимости МОР


def test_stop_default_dr_and_validation():
    """Дефолт stop='dr' — поведение прежнее; неизвестный критерий отвергается."""
    assert Config().stop == "dr"
    dom = geometry.make_L(1.0, 0.5)
    pb = PlateBending.from_config(dom, Config(p=4, Q=16))
    with pytest.raises(ValueError, match="stop"):
        ContactMOR(pb, Config(p=4, Q=16, stop="bogus"))


# --------------------------------------------------------------------------- #
#  Ворота комплементарности золотой серии (P1.1) — честность Табл. 4.2
# --------------------------------------------------------------------------- #
def test_gate_golden_complementarity():
    """ВОРОТА: метрики Синьорини золотой контактной серии — фиксированный диапазон.

    Факт золотого прогона (L-форма, h=h_ktn, Q=120, p=10, β=1.2, 8000 итераций):
    comp_residual = 8.59e-2, gap_overshoot = 3.05e-3 — честное недосхождение МОР
    при лимите итераций. Выход из диапазона В ЛЮБУЮ сторону означает изменение
    метода/параметров и требует пересмотра golden (числа golden неприкосновенны).
    """
    from golden_config import GoldenConfig
    from run_lshape_contact import compute_w_free_lshape, lshape_lab_config

    g = GoldenConfig()
    delta = g.gap_factor * compute_w_free_lshape(g)
    dom = geometry.make_L(g.L_side, g.L_cut)
    lab = lshape_lab_config(g)
    pb = PlateBending.from_config(dom, lab)
    res = ContactMOR(pb, lab, gap=delta).solve()
    assert 0.07 <= res.comp_residual <= 0.10        # факт 8.585e-2
    assert 2.0e-3 <= res.gap_overshoot <= 4.0e-3    # факт 3.050e-3
