r"""Смешанные КУ для геометрически-нелинейной теории Кармана (v0.6.5, снят задел v0.7).

Структура прогиба ``w = ∏(сторона)^{2|1|0}·Φ`` (``clamped`` → фактор², ``hinge`` →
фактор¹, ``free`` → без фактора) обобщает защемление/шарнир на СМЕШАННЫЕ кромки
прямоугольника. Мембранная связь Фёппля–Кармана — та же (immovable по контуру).

Верификация:
* ЛИНЕЙНЫЙ предел (малая нагрузка) ⇒ Карман-mixed = КЛАССИЧЕСКИЙ mixed (ряд Леви)
  МАШИННО (та же ∏-структура) — сильнейший сертификат;
* нелинейный режим ⇒ мембранное ужесточение (Карман < классика);
* редукция к clamped/soft_hinge на уровне дискретизации (иная R-функция ω²/ω).
"""

from __future__ import annotations

import numpy as np

from plate_solver.clamped import MixedRectPlate
from plate_solver.config import Config
from plate_solver.geometry import make_rectangle
from plate_solver.membrane import KarmanPlate

_X = (-1.0, 1.0, -0.6, 0.6)
_SCSC = {"x1": "hinge", "x2": "hinge", "y1": "clamped", "y2": "clamped"}   # смешанные (Леви)


def _cfg(q0, p=12, Q=160):
    return Config(E=1.0, nu=0.3, h=1.0, q0=q0, a=1.0, p=p, Q=Q, n_load_steps=2,
                  karman_tol=1e-10, karman_max_iter=300)


def _karman_mixed(q0, sides=_SCSC, **kw):
    return KarmanPlate.from_config(make_rectangle(*_X), _cfg(q0, **kw), bc_type="mixed",
                                   inplane_bc="immovable", sides=sides).solve_uniform()


def _classic_mixed(q0, sides=_SCSC):
    mp = MixedRectPlate(*_X, sides, _cfg(q0))
    c = mp.solve(np.full(mp.quad.x.size, q0))
    return float(np.max(np.abs(mp.deflection(c, mp.quad.x, mp.quad.y))))


def test_karman_mixed_linear_limit_matches_classic_levy():
    """Малая нагрузка: Карман-mixed = классический mixed (ряд Леви) МАШИННО — сертификат."""
    r = _karman_mixed(1e-4)
    w_classic = _classic_mixed(1e-4)
    assert abs(r.w_max - w_classic) / w_classic < 1e-8    # та же ∏-структура, линейный предел


def test_karman_mixed_nonlinear_stiffening():
    """Умеренная нагрузка: мембранное натяжение УЖЕСТОЧАЕТ — Карман-mixed < классика."""
    r = _karman_mixed(3.0)
    w_classic = _classic_mixed(3.0)
    assert r.converged
    assert r.w_max < w_classic                            # геом. нелинейность (N > 0)
    assert abs(r.w_max - w_classic) / w_classic > 1e-3    # эффект заметен


def test_karman_mixed_reduces_to_clamped_and_soft_hinge():
    """Все стороны clamped/hinge ⇒ Карман-mixed ≈ Карман clamped/soft_hinge (дискретизация)."""
    allc = dict.fromkeys(("x1", "x2", "y1", "y2"), "clamped")
    allh = dict.fromkeys(("x1", "x2", "y1", "y2"), "hinge")
    dom, cfg = make_rectangle(*_X), _cfg(2.0)
    r_c = KarmanPlate.from_config(dom, cfg, bc_type="clamped",
                                  inplane_bc="immovable").solve_uniform()
    r_h = KarmanPlate.from_config(dom, cfg, bc_type="soft_hinge",
                                  inplane_bc="immovable").solve_uniform()
    assert abs(_karman_mixed(2.0, allc).w_max - r_c.w_max) / r_c.w_max < 1e-3
    assert abs(_karman_mixed(2.0, allh).w_max - r_h.w_max) / r_h.w_max < 1e-3


def test_karman_mixed_intermediate_between_clamped_and_hinge():
    """Смешанная (2 clamped + 2 hinge) — прогиб МЕЖДУ полностью защемлённой и шарнирной."""
    dom, cfg = make_rectangle(*_X), _cfg(1.0)
    w_c = KarmanPlate.from_config(dom, cfg, bc_type="clamped",
                                  inplane_bc="immovable").solve_uniform().w_max
    w_h = KarmanPlate.from_config(dom, cfg, bc_type="soft_hinge",
                                  inplane_bc="immovable").solve_uniform().w_max
    w_mix = _karman_mixed(1.0).w_max
    assert w_c < w_mix < w_h                               # мягче защемления, жёстче шарнира


def test_karman_mixed_free_sides_rejected():
    """FREE-стороны для нелинейного Кармана — отклонены (не верифицированы; аудит v0.6.5).

    Глубоко-нелинейный режим free+movable не сходится Пикаром, а результат
    возвращался бы с warning'ом — честнее отказ; для free — theory = classic.
    """
    import pytest

    from plate_solver.problem import CaseError, Problem
    d = {
        "geometry": {"kind": "rectangle", "x1": -1.0, "x2": 1.0, "y1": -0.6, "y2": 0.6},
        "bc": {"type": "mixed", "sides": [
            {"side": "x1", "type": "hinge"}, {"side": "x2", "type": "free"},
            {"side": "y1", "type": "hinge"}, {"side": "y2", "type": "free"}]},
        "load": {"type": "uniform", "q0": 1.0},
        "model": {"theory": "karman", "h": 1.0},
        "discretization": {"p": 8, "Q": 96, "grid_n": 16},
        "verify": {"reference": "none"},
    }
    with pytest.raises(CaseError, match="free"):
        Problem.from_dict(d)
    d["model"]["theory"] = "classic"                       # классика с free — как раньше
    Problem.from_dict(d)
