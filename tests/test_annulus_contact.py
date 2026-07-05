"""Контакт на кольце (фаза 2, P3.7): предельное тождество, инварианты, регресс.

Ворота, не требующие эталона: (а) gap_factor=2 ⇒ контакт не возникает —
математическое тождество МОР; (б) инварианты с замороженными допусками
(факт × 3, журнал PROGRESS.md); (в) регресс-снимок против
cases/baselines.json; топология зоны фиксируется, но не гейтится.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from plate_solver.problem import Problem

_ROOT = Path(__file__).resolve().parents[1]
_CASE = _ROOT / "cases" / "ladder" / "annulus_soft_contact.toml"
_BASE = json.loads((_ROOT / "cases" / "baselines.json").read_text(encoding="utf-8"))


def test_case_file_valid():
    p = Problem.from_toml(_CASE)
    assert p.contact.enabled and p.contact.gap_factor == 0.5
    cfg = p.to_config()
    assert (cfg.beta, cfg.max_iter, cfg.tol) == (1.2, 8000, 1.0e-8)  # как в golden


def test_gate_no_contact_limit():
    """ВОРОТА (а): gap_factor=2 ⇒ r ≡ 0 и w совпадает с чистым изгибом.

    Δ = 2·w_free > w всюду ⇒ r + β(w−Δ) < 0 на каждой итерации ⇒ проекция
    max(·, 0) держит r ≡ 0 — тождество, обязано выполняться точно.
    """
    import copy
    import tomllib

    from plate_solver.dispatch import solve

    data = tomllib.loads(_CASE.read_text(encoding="utf-8"))
    free = copy.deepcopy(data)
    free["contact"] = {"enabled": False}
    limit = copy.deepcopy(data)
    limit["contact"]["gap_factor"] = 2.0
    limit["contact"]["max_iter"] = 50
    res_free = solve(Problem.from_dict(free))
    res_lim = solve(Problem.from_dict(limit))
    q0 = res_lim.config.q0
    assert float(res_lim.contact.r_nodes.max()) < 1e-12 * q0     # контакта нет
    rel = abs(res_lim.w_max - res_free.w_max) / res_free.w_max
    assert rel <= 1e-10                                           # чистый изгиб
    assert res_lim.contact.converged                              # ‖Δr‖ = 0 сразу


@pytest.fixture(scope="module")
def contact_run():
    from plate_solver.dispatch import solve

    return solve(Problem.from_toml(_CASE))


@pytest.mark.big
def test_gate_invariants(contact_run):
    """ВОРОТА (б): r ≥ 0; ‖Δr‖ монотонно убывает; замороженные допуски."""
    c = contact_run.contact
    h = c.residual_history
    assert float(c.r_nodes.min()) >= 0.0
    assert np.all(np.diff(h[3:]) <= 1e-9 * h[0])       # монотонное убывание
    assert c.comp_residual <= 3.6e-2                    # факт 1.190e-2 × 3
    assert abs(c.gap_overshoot) <= 2.9e-3               # факт 9.62e-4 × 3


@pytest.mark.big
def test_gate_regression_baseline(contact_run):
    """ВОРОТА (в): регресс-снимок против baselines.json; топология — инфо."""
    from scipy import ndimage

    b = _BASE["annulus_soft_contact"]
    c = contact_run.contact
    assert float(c.r_nodes.max()) == pytest.approx(b["r_max"], rel=1e-6)
    assert int((c.r_nodes > 0).sum()) == b["n_contact"]
    assert contact_run.w_max == pytest.approx(b["w_max"], rel=1e-9)
    assert contact_run.delta == pytest.approx(b["delta"], rel=1e-9)
    # топологию зоны фиксируем (материал главы 4), порогом не гейтим
    n_comp = int(ndimage.label(c.contact_zone)[1])
    assert n_comp >= 1                                   # смысловой минимум
