"""Штамп-квадрат у входящего угла L-формы (фаза 2, P4.3: lshape_stamp).

Ворота: (i) тождество «zone = вся Ω» ≡ базовый путь (rel ≤ 1e-15 — та же
арифметика); (ii) r ≡ 0 вне зоны; (iii) предельный тест gap_factor=2 ⇒
r ≡ 0; (iv) инварианты; big — регресс против baselines.json (топология
зоны фиксируется, не гейтится).
"""

from __future__ import annotations

import copy
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from plate_solver.dispatch import build_domain, solve
from plate_solver.problem import Problem

_ROOT = Path(__file__).resolve().parents[1]
_CASE = _ROOT / "cases" / "ladder" / "lshape_stamp.toml"
_CI_CASE = _ROOT / "cases" / "ci" / "lshape_stamp.toml"


def _light(**over) -> dict:
    d = tomllib.loads(_CI_CASE.read_text(encoding="utf-8"))
    for k, v in over.items():
        d[k] = copy.deepcopy(v)
    return d


def test_gate_zone_whole_omega_identity():
    """ВОРОТА: зона «вся Ω» тождественна базовому пути без зоны (rel ≤ 1e-15)."""
    d = _light()
    base = copy.deepcopy(d)
    del base["contact"]["zone"]
    whole = copy.deepcopy(d)
    whole["contact"]["zone"] = {"kind": "rectangle", "x1": -1.0, "x2": 2.0,
                                "y1": -1.0, "y2": 2.0}      # накрывает всю Ω
    r1 = solve(Problem.from_dict(base))
    r2 = solve(Problem.from_dict(whole))
    assert abs(r2.w_max - r1.w_max) <= 1e-15 * r1.w_max
    assert np.array_equal(r1.contact.r_nodes, r2.contact.r_nodes)


def test_gate_reaction_only_in_zone():
    """ВОРОТА: r ≡ 0 вне зоны штампа; контакт в зоне возник."""
    res = solve(Problem.from_toml(_CI_CASE))
    p = Problem.from_toml(_CI_CASE)
    q = res.contact.plate.quad
    zone = build_domain(p.contact.zone)
    outside = zone.omega(q.x, q.y) <= 0.0
    assert float(np.max(np.abs(res.contact.r_nodes[outside]))) == 0.0
    assert int((res.contact.r_nodes > 0).sum()) > 0


def test_gate_stamp_limit_no_contact():
    """ВОРОТА: gap_factor=2 ⇒ r ≡ 0 (тождество МОР) и w == чистый изгиб."""
    d = _light()
    d["contact"]["gap_factor"] = 2.0
    d["contact"]["max_iter"] = 50
    res = solve(Problem.from_dict(d))
    assert float(res.contact.r_nodes.max()) == 0.0
    free = _light()
    free["contact"] = {"enabled": False}
    res_free = solve(Problem.from_dict(free))
    assert abs(res.w_max - res_free.w_max) <= 1e-10 * res_free.w_max


def test_invariants_light():
    res = solve(Problem.from_toml(_CI_CASE))
    c = res.contact
    h = c.residual_history
    assert float(c.r_nodes.min()) >= 0.0
    assert np.all(np.diff(h[3:]) <= 1e-9 * h[0])


@pytest.mark.big
def test_gate_regression_baseline_full():
    """ВОРОТА (big): полный lshape_stamp против регресс-снимка baselines.json."""
    from scipy import ndimage

    base = json.loads((_ROOT / "cases" / "baselines.json").read_text(encoding="utf-8"))
    b = base["lshape_stamp"]
    res = solve(Problem.from_toml(_CASE))
    c = res.contact
    assert float(c.r_nodes.max()) == pytest.approx(b["r_max"], rel=1e-6)
    assert int((c.r_nodes > 0).sum()) == b["n_contact"]
    assert res.w_max == pytest.approx(b["w_max"], rel=1e-9)
    n_comp = int(ndimage.label(c.contact_zone)[1])
    assert n_comp >= 1                       # топология — фиксируется, не гейтится
