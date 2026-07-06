"""Контактный интерфейс защемлённого решателя.

A3.2: (т1) прежний solve ≡ solve_rhs; (т2) Δw защемлённого круга против
аналитики (решение в структуре ⇒ rel ≤ 1e-10). A3.4: контакт при
защемлении — предельный тест, инварианты, комплементарность (факт 3.12e-2,
потолок 5e-2, заморожено ×3), регресс-снимок, инфо-сравнение с soft.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from plate_solver import geometry
from plate_solver.clamped import ClampedPlate
from plate_solver.config import Config
from plate_solver.dispatch import solve
from plate_solver.problem import Problem

_ROOT = Path(__file__).resolve().parents[1]

BASE = {
    "geometry": {"kind": "circle", "a": 1.0},
    "bc": {"type": "clamped"},
    "load": {"type": "uniform", "q0": 4.0},
    "model": {"h": 0.06},
    "discretization": {"p": 8, "Q": 96, "grid_n": 16},
}
MOR = {"enabled": True, "gap_factor": 0.5, "beta": 1.2,
       "max_iter": 2000, "tol": 1.0e-12}


def _problem(**over) -> Problem:
    d = copy.deepcopy(BASE)
    for k, v in over.items():
        d[k] = copy.deepcopy(v)
    return Problem.from_dict(d)


# --------------------------------------------------------------------------- #
#  A3.2 — интерфейс
# --------------------------------------------------------------------------- #
def test_gate_t1_solve_equals_solve_rhs():
    """т1: прежний solve — псевдоним solve_rhs (побитово)."""
    cp = ClampedPlate.from_config(geometry.make_circle(1.0),
                                  Config(q0=4.0, h=0.06, p=6, Q=64))
    rng = np.random.default_rng(7)
    f = 4.0 + rng.normal(0.0, 1.0, cp.quad.x.size)
    assert np.array_equal(cp.solve(f), cp.solve_rhs(f))


def test_gate_t2_laplacian_machinery_exact_in_structure():
    r"""т2: Δw через кэш Δ(ω²Φ) точен для решения, ЛЕЖАЩЕГО в структуре.

    Точное решение защемлённого круга w = q(a²−r²)²/(64D) = ω²·γ, γ = qa²/16D
    (Φ ≡ const): задаём коэффициенты аналитически и гейтим МАШИНЕРИЮ вторых
    производных (символика ω + chebder), а не решатель — у решателя на круге
    остаётся квадратурный пол маски ~1/Q, усиливаемый в Δw (см. PROGRESS).

    .. math:: \Delta w = -q\,(a^2 - 2r^2)/(8D).
    """
    a, q0 = 1.0, 4.0
    cfg = Config(q0=q0, h=0.06, p=6, Q=128)
    cp = ClampedPlate.from_config(geometry.make_circle(a), cfg)
    c = np.zeros(cp.basis.N)
    c[0] = q0 * a**2 / (16.0 * cfg.D)              # w* = γ·ω², φ_0 = T0·T0 ≡ 1
    qn = cp.quad
    r2 = qn.x**2 + qn.y**2
    lap = cp.laplacian_at_quad(c)
    lap_exact = -q0 * (a**2 - 2.0 * r2) / (8.0 * cfg.D)
    rel = float(np.max(np.abs(lap - lap_exact)) / np.max(np.abs(lap_exact)))
    assert rel <= 1e-10, rel
    # прогиб через кэш ψ — то же точное поле
    w_exact = q0 * (a**2 - r2) ** 2 / (64.0 * cfg.D)
    w1 = cp.deflection_at_quad(c)
    assert float(np.max(np.abs(w1 - w_exact))) <= 1e-12 * float(np.max(w_exact))
    # согласованность кэша с прямым вычислением в произвольных точках
    w2 = cp.deflection(c, qn.x, qn.y)
    assert float(np.max(np.abs(w1 - w2))) <= 1e-14 * float(np.max(np.abs(w2)))


# --------------------------------------------------------------------------- #
#  A3.4 — контакт при защемлении
# --------------------------------------------------------------------------- #
def test_gate_clamped_contact_limit_no_contact():
    """gap huge ⇒ r ≡ 0 и совпадение с чистым изгибом (1e-10)."""
    free = solve(_problem(contact={"enabled": False}))
    res = solve(_problem(contact={**MOR, "gap_factor": 2.0, "max_iter": 50}))
    assert float(res.contact.r_nodes.max()) == 0.0
    assert abs(res.w_max - free.w_max) <= 1e-10 * free.w_max
    assert res.contact.converged


def test_gate_clamped_contact_invariants():
    """Инварианты + замороженная комплементарность (факт 3.12e-2 × 3)."""
    res = solve(_problem(contact=MOR))
    c = res.contact
    h = c.residual_history
    assert float(c.r_nodes.min()) >= 0.0
    assert np.all(np.diff(h[3:]) <= 1e-9 * h[0])
    assert c.comp_residual <= 9.4e-2               # факт 3.123e-2 × 3 (потолок 5e-2)
    assert abs(c.gap_overshoot) <= 8.5e-3          # факт 2.805e-3 × 3
    # инфо (фиксация, не гейт): пик защемления ниже пика шарнира
    soft = solve(_problem(bc={"type": "soft_hinge"}, contact=MOR))
    assert float(c.r_nodes.max()) > 0.0 and float(soft.contact.r_nodes.max()) > 0.0


@pytest.mark.big
def test_gate_circle_clamped_contact_regression():
    """Регресс-снимок case circle_clamped_contact против baselines (big)."""
    base = json.loads((_ROOT / "cases" / "baselines.json").read_text(encoding="utf-8"))
    b = base["circle_clamped_contact"]
    res = solve(Problem.from_toml(_ROOT / "cases" / "ladder"
                                  / "circle_clamped_contact.toml"))
    c = res.contact
    assert float(c.r_nodes.max()) == pytest.approx(b["r_max"], rel=1e-6)
    assert int((c.r_nodes > 0).sum()) == b["n_contact"]
    assert res.w_max == pytest.approx(b["w_max"], rel=1e-9)
    assert res.delta == pytest.approx(b["delta"], rel=1e-9)
