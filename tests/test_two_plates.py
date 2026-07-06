"""Контакт двух пластин (фаза 3, A4): тождества, смешанные закрепления, планформы.

Порядок ворот = порядок подзадач TODO: (a) равные планформы/soft/Δ=0;
(b) clamped+soft и вырождение «жёсткая вторая»; (c) круг над кольцом.

Отступление (журнал PROGRESS): при полном контакте УЗЛОВАЯ реакция
неединственна (стационарность фиксирует лишь проекцию ψᵀW(q₁−q₂−2r)=0,
узлов M ≫ N базисных функций), поэтому континуальные потолки (a)
проверяются на НЕПОДВИЖНОЙ ТОЧКЕ схемы: точное решение r ≡ (q₁−q₂)/2,
заданное стартом, обязано остаться на месте (это гейт знаков и сборки);
холодный старт гейтится по единственным величинам (w₁ ≡ w₂).
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
from plate_solver.contact import ContactMOR, TwoPlateMOR
from plate_solver.dispatch import solve
from plate_solver.plate import PlateBending
from plate_solver.problem import CaseError, Problem

_ROOT = Path(__file__).resolve().parents[1]
DOM = geometry.make_circle(1.0)


def _cfg(**over) -> Config:
    base = dict(q0=4.0, h=0.06, p=6, Q=64, grid_n=16, beta=1.0,
                max_iter=3000, tol=1e-12)
    base.update(over)
    return Config(**base)


# --------------------------------------------------------------------------- #
#  Схема
# --------------------------------------------------------------------------- #
def test_plate2_schema():
    d = {
        "geometry": {"kind": "circle", "a": 1.0},
        "bc": {"type": "soft_hinge"},
        "load": {"type": "uniform", "q0": 4.0},
        "contact": {"enabled": True, "target": "plate2", "gap": 0.0},
        "plate2": {"bc": {"type": "clamped"}, "load": {"type": "uniform", "q0": 1.0}},
    }
    p = Problem.from_dict(d)
    assert p.contact.target == "plate2" and p.plate2.bc.type == "clamped"
    assert p.contact.gap == 0.0                       # Δ=0 — касание, допустимо
    # [plate2] без target — ошибка; target без секции — ошибка; force — фаза 5
    bad = copy.deepcopy(d)
    bad["contact"]["target"] = "foundation"
    with pytest.raises(CaseError, match="plate2"):
        Problem.from_dict(bad)
    bad2 = copy.deepcopy(d)
    del bad2["plate2"]
    with pytest.raises(CaseError, match="plate2"):
        Problem.from_dict(bad2)
    bad3 = copy.deepcopy(d)
    bad3["contact"]["force"] = 0.5
    del bad3["contact"]["gap"]
    with pytest.raises(CaseError, match="фаза 5"):
        Problem.from_dict(bad3)


# --------------------------------------------------------------------------- #
#  (a) равные планформы, обе soft, Δ = 0
# --------------------------------------------------------------------------- #
def test_gate_a_exact_solution_is_fixed_point():
    """ВОРОТА (a): r ≡ (q₁−q₂)/2 — неподвижная точка; потолки континуума."""
    cfg = _cfg(max_iter=200)
    p1 = PlateBending.from_config(DOM, cfg)
    p2 = PlateBending.from_config(DOM, cfg)
    mor = TwoPlateMOR(p1, p2, cfg, q2=2.0, gap=0.0)
    res = mor.solve(r0=np.ones(p1.quad.x.size))       # точное решение стартом
    assert res.converged and res.iters == 1           # неподвижность
    rm = res.r_nodes[mor.mask]
    assert float(np.std(rm) / np.mean(rm)) <= 1e-8    # r ≡ const
    assert abs(float(np.mean(rm)) - 1.0) <= 1e-10     # (q₁−q₂)/2 = 1
    m = np.isfinite(res.w2_nodes)
    dw = float(np.max(np.abs(res.w_nodes[m] - res.w2_nodes[m])))
    assert dw <= 1e-10 * float(np.max(np.abs(res.w_nodes)))   # w₁ ≡ w₂


def test_gate_a_cold_start_converges_to_common_deflection():
    """(a) холодный старт: w₁ ≡ w₂ восстанавливается (узловое r неединственно)."""
    cfg = _cfg(max_iter=50_000, tol=1e-14)
    p1 = PlateBending.from_config(DOM, cfg)
    p2 = PlateBending.from_config(DOM, cfg)
    res = TwoPlateMOR(p1, p2, cfg, q2=2.0, gap=0.0).solve()
    m = np.isfinite(res.w2_nodes)
    dw = float(np.max(np.abs(res.w_nodes[m] - res.w2_nodes[m])))
    assert dw <= 5e-6 * float(np.max(np.abs(res.w_nodes)))
    # равнодействующая реакции близка к континуальной (инфо-мягкий гейт)
    q1 = p1.quad
    total = float(np.sum(q1.w * res.r_nodes))
    exact = 1.0 * float(np.sum(q1.w))                 # (q₁−q₂)/2 · |Ω|
    assert abs(total - exact) / exact < 0.15


def test_gate_a_reverse_order_no_contact():
    """(a) обратный порядок q₁ < q₂ ⇒ r ≡ 0 (нет прижатия) — точно."""
    cfg = _cfg(max_iter=100)
    p1 = PlateBending.from_config(DOM, cfg)
    p2 = PlateBending.from_config(DOM, cfg)
    res = TwoPlateMOR(p1, p2, cfg, q2=8.0, gap=0.0).solve()
    assert float(res.r_nodes.max()) == 0.0 and res.converged


# --------------------------------------------------------------------------- #
#  (b) разные закрепления + вырождение «жёсткая вторая»
# --------------------------------------------------------------------------- #
def test_gate_b_mixed_bc_invariants():
    """(b) soft + clamped: инварианты, комплементарность (факт 1.73e-2 × 3)."""
    cfg = _cfg(p=8, Q=96)
    p1 = PlateBending.from_config(DOM, cfg)
    p2 = ClampedPlate.from_config(DOM, cfg)
    res = TwoPlateMOR(p1, p2, cfg, q2=1.0, gap=0.0).solve()
    h = res.residual_history
    assert float(res.r_nodes.min()) >= 0.0
    assert np.all(np.diff(h[3:]) <= 1e-9 * h[0])
    assert res.comp_residual <= 5.2e-2                # факт 1.730e-2 × 3 (потолок 5e-2)
    assert abs(res.gap_overshoot) <= 6.1e-3           # факт 2.026e-3 × 3


def test_gate_b_rigid_second_degenerates_to_foundation():
    """(b) E₂ = 10⁹·E₁ ⇒ пара ≡ основание gap = Δ (факт ~1e-9, потолок 1e-6)."""
    import dataclasses

    cfg = _cfg(p=8, Q=96)
    p1 = PlateBending.from_config(DOM, cfg)
    s1 = p1.solve(np.full(p1.quad.x.size, cfg.q0))
    w_free = float(np.max(np.abs(p1.w_at_quad(s1))))
    delta = 0.5 * w_free
    p2r = PlateBending.from_config(DOM, dataclasses.replace(cfg, E=cfg.E * 1e9))
    pair = TwoPlateMOR(p1, p2r, cfg, q2=0.0, gap=delta).solve()
    found = ContactMOR(p1, cfg, gap=delta).solve()
    r_scale = float(found.r_nodes.max())
    w_scale = float(np.max(np.abs(found.w_nodes)))
    assert float(np.max(np.abs(pair.r_nodes - found.r_nodes))) / r_scale <= 3.1e-9
    assert float(np.max(np.abs(pair.w_nodes - found.w_nodes))) / w_scale <= 3.0e-9


# --------------------------------------------------------------------------- #
#  (c) [stretch] разные планформы: круг над кольцом
# --------------------------------------------------------------------------- #
@pytest.mark.big
def test_gate_c_circle_over_ring_regression():
    """(c) контакт на пересечении планформ; топология зоны — фиксация."""
    from scipy import ndimage

    base = json.loads((_ROOT / "cases" / "baselines.json").read_text(encoding="utf-8"))
    b = base["two_plates_ring"]
    res = solve(Problem.from_toml(_ROOT / "cases" / "ladder" / "two_plates_ring.toml"))
    c = res.contact
    h = c.residual_history
    q1 = c.plate.quad
    assert float(c.r_nodes.min()) >= 0.0
    assert np.all(np.diff(h[3:]) <= 1e-9 * h[0])
    hole = (q1.x**2 + q1.y**2) < (0.4 * 0.98) ** 2
    assert float(np.max(np.abs(c.r_nodes[hole]))) == 0.0   # над дыркой реакции нет
    assert float(c.r_nodes.max()) == pytest.approx(b["r_max"], rel=1e-6)
    assert int((c.r_nodes > 0).sum()) == b["n_contact"]
    n_comp = int(ndimage.label(c.contact_zone)[1])
    assert n_comp >= 1                                # топология фиксируется, не гейт


def test_f0_2_pair_summary_has_w2_panel(tmp_path):
    """F0.2: контактный планшет пары — 4 панели, среди них w₁ И w₂."""
    import matplotlib

    matplotlib.use("Agg")
    from plate_solver import viz

    res = solve(Problem.from_toml(_ROOT / "cases" / "ci" / "two_plates_ring.toml"))
    dest = tmp_path / "pair.png"
    fig = viz.plot_pair_summary(res.contact, save=str(dest))
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert any("w₁" in t for t in titles) and any("w₂" in t for t in titles)
    assert dest.stat().st_size > 10_000
    # штатный путь фигур: Result._save_figures выбирает планшет пары
    res.save_fields(tmp_path / "fields.npz")
    res._save_figures(tmp_path, formats=("png",))
    assert (tmp_path / "two_plates_ring_contact_summary.png").is_file()
