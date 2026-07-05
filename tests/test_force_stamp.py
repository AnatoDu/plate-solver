"""Силовой штамп (фаза 3, A2): задана сила P, ищется уровень.

Ворота самозамкнуты: позиционное решение (level*) → P* = ∫r dΩ → силовой
режим с P* обязан вернуть level*. Требует ГЛУБОКОЙ сходимости МОР
(tol=1e-11 реально достигается: недосошедшие прогоны смещают F(level)
сильнее потолков — диагностика в PROGRESS). Допуски заморожены факт × 3.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from plate_solver.dispatch import solve
from plate_solver.problem import CaseError, Problem

_ROOT = Path(__file__).resolve().parents[1]

BASE = {
    "geometry": {"kind": "L", "side": 1.0, "cut": 0.5},
    "bc": {"type": "soft_hinge"},
    "load": {"type": "uniform", "q0": 4.0},
    "model": {"h": 0.06},
    "discretization": {"p": 8, "Q": 40, "grid_n": 16},
}
ZONE = {"kind": "rectangle", "x1": 0.15, "x2": 0.45, "y1": 0.15, "y2": 0.45}


def _problem(contact) -> Problem:
    d = copy.deepcopy(BASE)
    d["contact"] = copy.deepcopy(contact)
    return Problem.from_dict(d)


def test_force_schema():
    p = _problem({"enabled": True, "force": 0.5, "zone": ZONE})
    assert p.contact.force == 0.5 and p.contact.gap is None
    # force + gap допустимы (gap игнорируется с warning в диспетчере)
    p2 = _problem({"enabled": True, "force": 0.5, "gap": 1e-4, "zone": ZONE})
    assert p2.contact.force == 0.5 and p2.contact.gap == 1e-4
    with pytest.raises(CaseError, match="force"):
        _problem({"enabled": True, "force": -1.0, "zone": ZONE})
    # без force действует прежнее правило «ровно одно из»
    with pytest.raises(CaseError, match="ровно одно"):
        _problem({"enabled": True, "zone": ZONE})


def test_force_ignores_scalar_gap_with_warning():
    res = solve(_problem({"enabled": True, "force": 0.05, "gap": 1e-4,
                          "beta": 1.0, "max_iter": 2000, "tol": 1e-10,
                          "zone": ZONE}))
    assert any("игнорируется" in w for w in res.warnings)
    # лёгкий прогон (2000 итер., МОР не сходится): ∫r шумит ~1e-3 — точный
    # баланс гейтится big-замыканием на глубокой сходимости
    assert res.level is not None and res.force_total == pytest.approx(0.05, rel=1e-2)


def test_force_limits():
    """Предельные случаи: P → 0⁺ ⇒ level → w_free; P сверх максимума — ошибка."""
    free = solve(_problem({"enabled": False}))
    w_free = free.w_max
    tiny = solve(_problem({"enabled": True, "force": 1e-7, "beta": 1.0,
                           "max_iter": 2000, "tol": 1e-10, "zone": ZONE}))
    assert tiny.level == pytest.approx(w_free, rel=1e-3)   # уровень уходит к w_free
    with pytest.raises(CaseError, match="максимум"):
        solve(_problem({"enabled": True, "force": 1e6, "beta": 1.0,
                        "max_iter": 500, "tol": 1e-10, "zone": ZONE}))


@pytest.mark.big
def test_gate_force_position_closure():
    """ВОРОТА (big): позиционный ↔ силовой замыкаются на глубокой сходимости.

    Факты (Q=40, tol=1e-11, converged): |Δlevel|/w_free = 2.253e-10,
    ∫r rel = 1.063e-10; потолки 1e-6 / 1e-8 пройдены — заморожено факт×3.
    """
    mor = {"enabled": True, "beta": 1.0, "max_iter": 500_000, "tol": 1e-11,
           "zone": ZONE}
    free = solve(_problem({"enabled": False}))
    w_free = free.w_max
    level_star = 0.5 * w_free
    pos = solve(_problem({**mor, "gap": level_star}))
    assert pos.contact.converged                     # глубокая сходимость обязана
    q = pos.contact.plate.quad
    p_star = float(np.sum(q.w * pos.contact.r_nodes))

    frc = solve(_problem({**mor, "force": p_star}))
    assert abs(frc.level - level_star) / w_free <= 7.0e-10       # факт 2.253e-10 × 3
    assert abs(frc.force_total - p_star) / p_star <= 3.2e-10     # факт 1.063e-10 × 3


@pytest.mark.big
def test_gate_lshape_stamp_force_regression():
    """ВОРОТА (big): регресс-снимок case lshape_stamp_force против baselines."""
    base = json.loads((_ROOT / "cases" / "baselines.json").read_text(encoding="utf-8"))
    b = base["lshape_stamp_force"]
    res = solve(Problem.from_toml(_ROOT / "cases" / "ladder" / "lshape_stamp_force.toml"))
    assert res.level == pytest.approx(b["level"], rel=1e-6)
    assert float(res.contact.r_nodes.max()) == pytest.approx(b["r_max"], rel=1e-6)
    assert int((res.contact.r_nodes > 0).sum()) == b["n_contact"]
    # ∫r отклоняется от P на протокольное недосхождение (~8e-4 при 8000 итер.);
    # регресс — против снимка, точный баланс — big-замыкание выше
    assert res.force_total == pytest.approx(b["force_total"], rel=1e-9)
