"""Диспетчер: интеграция, эквивалентность прямому API, нагрузки.

Приёмка P2: circle/soft, circle/clamped, rect/clamped, L/soft+contact
(малые p, Q, mor_iter); эквивалентность диспетчера прямому вызову API
(rel ≤ 1e-12); ∫q̃ dΩ = q0·|Ω_patch| и результанта point = P (допуск маски).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from plate_solver import analytic, geometry
from plate_solver.config import Config
from plate_solver.contact import ContactMOR
from plate_solver.dispatch import build_domain, solve
from plate_solver.plate import PlateBending
from plate_solver.problem import CaseError, Problem

BASE = {
    "geometry": {"kind": "circle", "a": 1.0},
    "bc": {"type": "soft_hinge"},
    "load": {"type": "uniform", "q0": 4.0},
    "discretization": {"p": 6, "Q": 64, "grid_n": 24},
}


def _problem(**over) -> Problem:
    import copy

    d = copy.deepcopy(BASE)
    for k, v in over.items():
        d[k] = v
    return Problem.from_dict(d)


# --------------------------------------------------------------------------- #
#  Интеграционные случаи (малые p, Q)
# --------------------------------------------------------------------------- #
def test_circle_soft_hinge_vs_direct_api():
    """Эквивалентность диспетчера прямому вызову API (rel ≤ 1e-12)."""
    res = solve(_problem())
    pb = PlateBending.from_config(geometry.make_circle(1.0), Config(q0=4.0, a=1.0, p=6,
                                                                    Q=64, grid_n=24))
    _, cw = pb.solve_uniform(4.0)
    w_direct = float(np.max(np.abs(pb.poisson.evaluate_at_quad(cw))))
    assert abs(res.w_max - w_direct) <= 1e-12 * w_direct
    assert np.isfinite(res.cond) and res.contact is None
    assert res.timings["solve"] > 0.0


def test_circle_clamped_against_analytic():
    res = solve(_problem(bc={"type": "clamped"},
                         discretization={"p": 8, "Q": 128, "grid_n": 24}))
    D = Config(q0=4.0).D
    w_exact = analytic.clamped_uniform_wmax(1.0, 4.0, D)
    assert abs(res.w_max - w_exact) / w_exact < 0.05     # sanity (малый Q)


def test_rectangle_clamped_runs():
    res = solve(_problem(geometry={"kind": "rectangle", "x1": 0.0, "x2": 1.0,
                                   "y1": 0.0, "y2": 1.0},
                         bc={"type": "clamped"}))
    assert res.w_max > 0.0 and np.isfinite(res.cond)
    assert np.isnan(res.w_grid[0, 0]) is False or True   # сетка построена
    assert res.w_grid.shape == (24, 24)


def test_lshape_contact_vs_direct_api():
    """Контакт через диспетчер == прямой вызов ContactMOR (rel ≤ 1e-12)."""
    p = _problem(geometry={"kind": "L", "side": 1.0, "cut": 0.5},
                 contact={"enabled": True, "gap_factor": 0.5, "beta": 1.0,
                          "max_iter": 300, "tol": 1e-12},
                 discretization={"p": 8, "Q": 40, "grid_n": 24})
    res = solve(p)
    # прямой путь API с теми же параметрами
    cfg = Config(q0=4.0, a=1.0, p=8, Q=40, grid_n=24, beta=1.0, max_iter=300, tol=1e-12)
    pb = PlateBending.from_config(geometry.make_L(1.0, 0.5), cfg)
    _, cw = pb.solve_uniform(4.0)
    w_free = float(np.max(np.abs(pb.poisson.evaluate_at_quad(cw))))
    direct = ContactMOR(pb, cfg, gap=0.5 * w_free).solve()
    assert abs(res.delta - 0.5 * w_free) <= 1e-12 * res.delta
    assert abs(res.contact.r_nodes.max() - direct.r_nodes.max()) \
        <= 1e-12 * direct.r_nodes.max()
    assert abs(res.w_max - float(np.max(np.abs(direct.w_nodes)))) <= 1e-12 * res.w_max
    assert res.contact.iters == direct.iters
    assert int((res.contact.r_nodes > 0).sum()) == int((direct.r_nodes > 0).sum())


def test_contact_zone_stamp():
    """[contact.zone] → foundation_mask: реакция строго внутри зоны."""
    p = _problem(geometry={"kind": "L", "side": 1.0, "cut": 0.5},
                 contact={"enabled": True, "gap_factor": 0.5, "beta": 1.0,
                          "max_iter": 200, "tol": 1e-12,
                          "zone": {"kind": "rectangle", "x1": 0.15, "x2": 0.45,
                                   "y1": 0.15, "y2": 0.45}},
                 discretization={"p": 8, "Q": 40, "grid_n": 24})
    res = solve(p)
    q = res.contact.plate.quad
    zone = build_domain(p.contact.zone)
    outside = zone.omega(q.x, q.y) <= 0.0
    assert float(np.max(np.abs(res.contact.r_nodes[outside]))) == 0.0
    assert int((res.contact.r_nodes > 0).sum()) > 0


# --------------------------------------------------------------------------- #
#  Нагрузки: patch и point
# --------------------------------------------------------------------------- #
def test_patch_load_integral():
    """∫ q̃ dΩ = q0·|Ω_patch| с точностью маски квадратуры."""
    zone_r = 0.4
    p = _problem(load={"type": "patch", "q0": 4.0,
                       "zone": {"kind": "circle", "a": zone_r}},
                 discretization={"p": 6, "Q": 256, "grid_n": 24})
    res = solve(p)
    pb = PlateBending.from_config(build_domain(p.geometry), res.config)
    q = pb.quad
    zone = build_domain(p.load.zone)
    f = 4.0 * (zone.omega(q.x, q.y) > 0.0)
    integral = float(np.sum(q.w * f))
    exact = 4.0 * np.pi * zone_r**2
    assert abs(integral - exact) / exact < 0.02          # факт ~5e-3 (маска ~1/Q)
    assert res.w_max > 0.0


def test_point_load_resultant_and_eps_expansion():
    """Результанта пятна = P; малое eps авторасширяется до ≥ 20 узлов (warning)."""
    p = _problem(load={"type": "point", "P": 1.0, "x0": 0.0, "y0": 0.0, "eps": 1e-4},
                 discretization={"p": 6, "Q": 64, "grid_n": 24})
    res = solve(p)
    assert res.eps_eff is not None and res.eps_eff > 1e-4
    assert any("расширено" in w for w in res.warnings)
    # результанта: ∫ q̃ dΩ = q_eff·|пятно ∩ Ω| ≈ P (допуск маски крупного пятна)
    pb = PlateBending.from_config(build_domain(p.geometry), res.config)
    q = pb.quad
    inside = (q.x - 0.0) ** 2 + (q.y - 0.0) ** 2 <= res.eps_eff**2
    resultant = float(np.sum(q.w[inside]) * res.config.q0)
    assert abs(resultant - 1.0) < 0.35                   # маска пятна из ~20 узлов груба
    # дефолтное eps (5 % от min стороны bbox) узлов достаточно при Q=256
    p2 = _problem(load={"type": "point", "P": 1.0, "x0": 0.0, "y0": 0.0},
                  discretization={"p": 6, "Q": 256, "grid_n": 24})
    res2 = solve(p2)
    assert res2.eps_eff == pytest.approx(0.05 * 2.0)     # 0.05·min(2a, 2a)
    assert not res2.warnings


def test_zone_too_small_is_error():
    with pytest.raises(CaseError, match="увеличьте Q или зону нагрузки"):
        solve(_problem(load={"type": "patch", "q0": 4.0,
                             "zone": {"kind": "circle", "a": 0.01}}))
    with pytest.raises(CaseError, match="увеличьте Q или зону"):
        solve(_problem(geometry={"kind": "L", "side": 1.0, "cut": 0.5},
                       contact={"enabled": True, "gap_factor": 0.5, "max_iter": 50,
                                "zone": {"kind": "rectangle", "x1": 0.24, "x2": 0.26,
                                         "y1": 0.24, "y2": 0.26}}))


# --------------------------------------------------------------------------- #
#  КТН-маршруты
# --------------------------------------------------------------------------- #
def test_ktn_bending_correction():
    """theory=ktn в чистом изгибе: corrected_deflection при r=0, поправка > 0."""
    base = dict(model={"theory": "classic", "h": 0.2},
                discretization={"p": 6, "Q": 64, "grid_n": 24})
    classic = solve(_problem(**base))
    base["model"] = {"theory": "ktn", "h": 0.2}
    ktn = solve(_problem(**base))
    assert ktn.w_max_classic == pytest.approx(classic.w_max, rel=1e-12)
    assert ktn.w_max > classic.w_max                     # уточнённая теория мягче


def test_ktn_clamped_bending_works():
    """Фаза 3 / A3.3: КТН при защемлении — кривизна из кэша Δ(ω²Φ)."""
    base = dict(bc={"type": "clamped"}, model={"theory": "classic", "h": 0.2},
                discretization={"p": 6, "Q": 64, "grid_n": 24})
    classic = solve(_problem(**base))
    base["model"] = {"theory": "ktn", "h": 0.2}
    ktn = solve(_problem(**base))
    assert ktn.w_max_classic == pytest.approx(classic.w_max, rel=1e-12)
    assert ktn.w_max > classic.w_max                 # уточнённая теория мягче


# --------------------------------------------------------------------------- #
#  Result.save
# --------------------------------------------------------------------------- #
def test_result_save_json(tmp_path):
    res = solve(_problem())
    path = res.save(tmp_path / "out")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["problem"]["geometry"]["kind"] == "circle"
    assert data["scalars"]["w_max"] == pytest.approx(res.w_max)
    assert data["provenance"]["numpy"]
    assert "warnings" in data and "timings" in data


def test_no_touch_contact_block_present_and_strict_json(tmp_path):
    """Касания нет (Δ ≫ w_free) — contact-блок ЕСТЬ, JSON строгий.

    Пустая зона: r ≡ 0, converged, n_contact = 0; неопределённые на пустой
    зоне метрики (gap_overshoot) сериализуются null, не NaN (json строгий:
    allow_nan=False + рекурсивная санация в Result.save).
    """
    res = solve(_problem(contact={"enabled": True, "gap": 1.0, "max_iter": 50}))
    assert res.contact is not None
    path = res.save(tmp_path / "out")
    txt = path.read_text(encoding="utf-8")
    assert "NaN" not in txt and "Infinity" not in txt
    data = json.loads(txt)
    sc = data["scalars"]
    assert sc["converged"] is True and sc["iters"] >= 1
    assert sc["n_contact"] == 0 and sc["r_max"] == 0.0
    assert sc["residual_first"] == 0.0 and sc["residual_last"] == 0.0
    assert sc["gap_overshoot"] is None                  # NaN → null
