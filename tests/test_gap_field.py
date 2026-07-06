"""Поле зазора Δ(x, y): схема, ворота-тождества т1–т4.

т1: kind=const ≡ скалярный путь v0.2 побитово; т2: Δ ≥ 2·w_free ⇒ r ≡ 0;
т3: steps-зона с огромным value ≡ исключение зоны из foundation_mask;
т4: paraboloid на круге — регресс + сходимость к плоскому при r_curv → ∞.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from plate_solver import geometry
from plate_solver.config import Config
from plate_solver.contact import ContactMOR
from plate_solver.dispatch import solve
from plate_solver.plate import PlateBending
from plate_solver.problem import CaseError, Problem

_ROOT = Path(__file__).resolve().parents[1]

BASE = {
    "geometry": {"kind": "circle", "a": 1.0},
    "bc": {"type": "soft_hinge"},
    "load": {"type": "uniform", "q0": 4.0},
    "model": {"h": 0.06},
    "contact": {"enabled": True, "gap_factor": 0.5, "beta": 1.2,
                "max_iter": 500, "tol": 1.0e-12},
    "discretization": {"p": 8, "Q": 96, "grid_n": 16},
}


def _problem(**over) -> Problem:
    d = copy.deepcopy(BASE)
    for k, v in over.items():
        d[k] = copy.deepcopy(v)
    return Problem.from_dict(d)


def _contact(gap_table=None, **contact_over) -> dict:
    c = copy.deepcopy(BASE["contact"])
    c.pop("gap_factor", None)
    if gap_table is not None:
        c["gap"] = gap_table
    c.update(contact_over)
    return c


def _w_free() -> float:
    res = solve(_problem(contact={"enabled": False}))
    return res.w_max


# --------------------------------------------------------------------------- #
#  Схема
# --------------------------------------------------------------------------- #
def test_gap_schema_valid_and_errors():
    p = _problem(contact=_contact({"kind": "plane", "a": 0.0, "b": 1e-5, "c": 1e-4}))
    assert p.contact.gap_field.kind == "plane" and p.contact.gap is None
    with pytest.raises(CaseError, match="contact.gap.kind"):
        _problem(contact=_contact({"kind": "wavy", "value": 1.0}))
    with pytest.raises(CaseError, match="ровно одно"):
        _problem(contact=_contact({"kind": "const", "value": 1e-4}, gap_factor=0.5))
    with pytest.raises(CaseError, match="r_curv"):
        _problem(contact=_contact({"kind": "paraboloid", "cx": 0.0, "cy": 0.0,
                                   "apex": 1e-4}))
    with pytest.raises(CaseError, match="value"):
        _problem(contact=_contact({"kind": "steps", "base": 1e-4,
                                   "zones": [{"kind": "circle", "a": 0.3}]}))
    # отрицательный зазор на основании ловится диспетчером
    with pytest.raises(CaseError, match="min Δ"):
        solve(_problem(contact=_contact({"kind": "plane", "a": 1.0, "b": 0.0,
                                         "c": -0.5})))


# --------------------------------------------------------------------------- #
#  т1: const ≡ скалярный gap (побитово)
# --------------------------------------------------------------------------- #
def test_gate_t1_const_identical_to_scalar():
    delta = 0.5 * _w_free()
    res_scalar = solve(_problem(contact=_contact() | {"gap": delta}))
    res_const = solve(_problem(contact=_contact({"kind": "const", "value": delta})))
    c1, c2 = res_scalar.contact, res_const.contact
    assert c1.iters == c2.iters
    assert np.array_equal(c1.r_nodes, c2.r_nodes)       # побитово
    assert np.array_equal(c1.w_nodes, c2.w_nodes)
    assert res_scalar.w_max == res_const.w_max


# --------------------------------------------------------------------------- #
#  т2: Δ ≥ 2·w_free всюду ⇒ r ≡ 0
# --------------------------------------------------------------------------- #
def test_gate_t2_large_plane_gap_no_contact():
    w_free = _w_free()
    res = solve(_problem(contact=_contact({"kind": "plane", "a": 0.1 * w_free,
                                           "b": 0.0, "c": 2.5 * w_free},
                                          max_iter=50)))
    assert float(res.contact.r_nodes.max()) == 0.0
    assert abs(res.w_max - w_free) <= 1e-10 * w_free


# --------------------------------------------------------------------------- #
#  т3: steps-зона с value = 10³·w_free ≡ исключение зоны из foundation_mask
# --------------------------------------------------------------------------- #
def test_gate_t3_steps_zone_equals_mask_exclusion():
    w_free = _w_free()
    delta = 0.5 * w_free
    zone = {"kind": "rectangle", "x1": -0.4, "x2": 0.1, "y1": -0.4, "y2": 0.1}
    res_steps = solve(_problem(contact=_contact(
        {"kind": "steps", "base": delta,
         "zones": [dict(zone, value=1e3 * w_free)]})))
    # прямой API: то же основание, но зона исключена из foundation_mask
    dom = geometry.make_circle(1.0)
    cfg = Config(q0=4.0, h=0.06, p=8, Q=96, grid_n=16, beta=1.2,
                 max_iter=500, tol=1.0e-12)
    pb = PlateBending.from_config(dom, cfg)
    from plate_solver.dispatch import build_domain
    from plate_solver.problem import GeometrySpec

    zdom = build_domain(GeometrySpec(kind="rectangle", **{k: zone[k]
                                                          for k in ("x1", "x2", "y1", "y2")}))
    outside = lambda X, Y: zdom.omega(X, Y) <= 0.0       # noqa: E731
    direct = ContactMOR(pb, cfg, foundation_mask=outside, gap=delta).solve()
    r1, r2 = res_steps.contact.r_nodes, direct.r_nodes
    scale = max(float(r2.max()), 1e-300)
    assert float(np.max(np.abs(r1 - r2))) <= 1e-12 * scale
    assert abs(res_steps.w_max - float(np.max(np.abs(direct.w_nodes)))) \
        <= 1e-12 * res_steps.w_max


# --------------------------------------------------------------------------- #
#  т4: paraboloid на круге — регресс + сходимость к плоскому при r_curv → ∞
# --------------------------------------------------------------------------- #
@pytest.mark.big
def test_gate_t4_paraboloid_regression_and_flat_limit():
    base = json.loads((_ROOT / "cases" / "baselines.json").read_text(encoding="utf-8"))
    b = base["circle_paraboloid_contact"]
    w_free = _w_free()
    apex = 0.5 * w_free

    def run(r_curv):
        return solve(_problem(contact=_contact(
            {"kind": "paraboloid", "r_curv": r_curv, "cx": 0.0, "cy": 0.0,
             "apex": apex}, max_iter=2000)))

    # регресс-снимок выразительно неплоского штампа
    res = run(b["r_curv"])
    assert float(res.contact.r_nodes.max()) == pytest.approx(b["r_max"], rel=1e-6)
    assert int((res.contact.r_nodes > 0).sum()) == b["n_contact"]
    assert res.w_max == pytest.approx(b["w_max"], rel=1e-9)

    # сходимость к плоскому: три r_curv, сближение монотонно (инфо) + ворота
    flat = solve(_problem(contact=_contact({"kind": "const", "value": apex},
                                           max_iter=2000)))
    rels = []
    for r_curv in (1.0e4, 1.0e5, 1.0e6):
        r = run(r_curv)
        rels.append(abs(r.w_max - flat.w_max) / flat.w_max)
    assert rels[0] > rels[1] > rels[2], rels             # монотонное сближение (инфо)
    assert rels[-1] <= 8.0e-6, rels                      # факт 2.666e-6 × 3 (потолок 1e-3)